"""
llm_client.py
-------------
Unified LLM client supporting both HuggingFace (for XAI) and Ollama (for fast inference).

Architecture rationale:
- We need TWO backends because:
  * HuggingFace: exposes attention weights, gradients — required for XAI
  * Ollama:      fast, quantized inference — useful for large-scale baseline runs
- A common LLMResponse dataclass ensures all downstream modules are backend-agnostic.
- `PromptStrategy` controls explanation elicitation mode (CoT vs self-explain).

The client is the only module that touches the model directly.
All other modules receive LLMResponse objects — clean separation of concerns.
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Standardized output from any LLM backend.
    All downstream modules ONLY interact with this object — never with the model directly.
    """
    # Text outputs
    question: str
    answer: str
    explanation: str
    full_response: str                          # Raw model output

    # Token-level signals (populated in HuggingFace mode)
    tokens: List[str] = field(default_factory=list)
    input_tokens: List[str] = field(default_factory=list)
    output_tokens: List[str] = field(default_factory=list)
    token_log_probs: List[float] = field(default_factory=list)
    answer_confidence: float = 0.0              # exp(mean log-prob) ∈ [0, 1]

    # Attention (populated in HuggingFace mode with output_attentions=True)
    attention_rollout: Optional[np.ndarray] = None  # Shape: (n_input_tokens,)
    raw_attention: Optional[List[np.ndarray]] = None

    # Gradients (populated when compute_gradients=True)
    gradient_scores: Optional[np.ndarray] = None

    # Metadata
    model_id: str = ""
    backend: str = ""                           # "huggingface" | "ollama"
    latency_seconds: float = 0.0
    prompt_strategy: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified LLM client. Instantiate once, call generate() repeatedly.

    Example:
        client = LLMClient.from_config("config/model_config.yaml")
        response = client.generate(question="What is the capital of France?")
        print(response.answer)          # "Paris"
        print(response.answer_confidence)  # 0.87
    """

    def __init__(self, config: dict):
        self.config = config
        self.backend = config["model"]["backend"]
        self.device = config["model"]["device"]
        self.max_new_tokens = config["model"]["max_new_tokens"]
        self.temperature = config["model"]["temperature"]
        self.n_stability_runs = config["model"]["n_stability_runs"]

        self._model = None
        self._tokenizer = None
        self._extractor = None
        self._ollama_client = None

        self._load_backend()

    @classmethod
    def from_config(cls, config_path: str) -> "LLMClient":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ── Backend Initialization ────────────────────────────────────────────────

    def _load_backend(self):
        if self.backend == "huggingface":
            self._load_huggingface()
        elif self.backend == "ollama":
            self._load_ollama()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Choose 'huggingface' or 'ollama'.")

    def _load_huggingface(self):
        """
        Load Gemma via HuggingFace transformers.
        Requires: pip install transformers accelerate sentencepiece

        Note: Gemma requires accepting the license on HuggingFace Hub.
        Run: huggingface-cli login   (one time only)
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from src.model.token_extractor import TokenExtractor

        model_id = self.config["model"]["hf_model_id"]
        logger.info(f"Loading {model_id} via HuggingFace (device={self.device})...")

        # device_map="auto" only for CUDA multi-GPU setups; MPS and CPU load directly
        device_map = "auto" if self.device == "cuda" else None

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=device_map,
            attn_implementation="eager",  # Required for output_attentions=True (SDPA doesn't support it)
        )

        if device_map is None:
            self._model = self._model.to(self.device)

        self._extractor = TokenExtractor(self._model, self._tokenizer, self.device)
        logger.info(f"✓ HuggingFace model loaded: {model_id}")

    def _load_ollama(self):
        """
        Load Ollama client.
        Requires: pip install ollama + Ollama running locally (ollama serve)
        Pull model first: ollama pull gemma:2b
        """
        import ollama as ollama_lib
        self._ollama_client = ollama_lib
        self._ollama_model = self.config["model"]["ollama_model_id"]
        logger.info(f"✓ Ollama client ready: {self._ollama_model}")

    # ── Main Generate Method ──────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        strategy: str = "self_explain",
        compute_gradients: bool = False,
    ) -> LLMResponse:
        """
        Generate an answer + explanation for a question.

        Args:
            question:          The input question.
            strategy:          "self_explain" | "cot" | "direct"
            compute_gradients: Compute gradient×input attribution (HF only, slower).

        Returns:
            LLMResponse with populated fields based on backend.
        """
        from src.model.prompt_templates import PromptTemplates

        # Build prompts
        if strategy == "cot":
            prompt_out = PromptTemplates.chain_of_thought(question)
        elif strategy == "self_explain":
            prompt_out = PromptTemplates.self_explanation(question)
        else:
            prompt_out = PromptTemplates.direct(question)

        t0 = time.time()

        if self.backend == "huggingface":
            response = self._generate_hf(
                question, prompt_out, compute_gradients
            )
        else:
            response = self._generate_ollama(question, prompt_out)

        response.latency_seconds = time.time() - t0
        response.prompt_strategy = strategy
        return response

    def generate_multiple(
        self, question: str, n: int = 5, strategy: str = "self_explain"
    ) -> List[LLMResponse]:
        """
        Generate N responses for the same question.
        Used by ESS (Explanation Stability Score) computation.
        Uses higher temperature to get varied outputs.
        """
        responses = []
        for i in range(n):
            resp = self.generate(question, strategy=strategy)
            responses.append(resp)
        return responses

    # ── HuggingFace Backend ───────────────────────────────────────────────────

    def _generate_hf(self, question: str, prompt_out, compute_gradients: bool) -> LLMResponse:
        import torch

        model_id = self.config["model"]["hf_model_id"]

        # Step 1: Generate answer
        answer_enc = self._tokenizer(
            prompt_out.answer_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            answer_ids = self._model.generate(
                **answer_enc,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.config["model"]["do_sample"],
                top_p=self.config["model"]["top_p"],
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the generated part
        n_input = answer_enc["input_ids"].shape[1]
        answer_token_ids = answer_ids[0, n_input:]
        answer_text = self._tokenizer.decode(answer_token_ids, skip_special_tokens=True).strip()
        answer_text = self._extract_final_answer(answer_text)

        # Step 2: Generate explanation
        explanation_prompt = prompt_out.explanation_prompt_template.format(answer=answer_text)
        expl_enc = self._tokenizer(
            explanation_prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            expl_ids = self._model.generate(
                **expl_enc,
                max_new_tokens=256,
                temperature=0.5,        # Lower temp for more deterministic explanation
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        n_expl_input = expl_enc["input_ids"].shape[1]
        expl_text = self._tokenizer.decode(
            expl_ids[0, n_expl_input:], skip_special_tokens=True
        ).strip()

        # Step 3: Extract token-level signals via TokenExtractor
        full_generated = answer_text + " " + expl_text
        extraction = self._extractor.extract(
            input_text=prompt_out.answer_prompt,
            generated_text=full_generated,
            compute_gradients=compute_gradients,
        )

        return LLMResponse(
            question=question,
            answer=answer_text,
            explanation=expl_text,
            full_response=full_generated,
            tokens=extraction.all_tokens,
            input_tokens=extraction.input_tokens,
            output_tokens=extraction.output_tokens,
            token_log_probs=extraction.output_log_probs,
            answer_confidence=extraction.answer_confidence,
            attention_rollout=extraction.attention_rollout,
            raw_attention=extraction.raw_attention,
            gradient_scores=extraction.gradient_scores,
            model_id=model_id,
            backend="huggingface",
        )

    # ── Ollama Backend ────────────────────────────────────────────────────────

    def _generate_ollama(self, question: str, prompt_out) -> LLMResponse:
        """
        Ollama inference — fast but no internal signals (attention/gradients).
        Log-probs are approximated via token logprobs if Ollama supports it.
        """
        # Generate answer
        answer_resp = self._ollama_client.generate(
            model=self._ollama_model,
            prompt=prompt_out.answer_prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "top_p": self.config["model"]["top_p"],
            },
        )
        answer_text = self._extract_final_answer(answer_resp["response"].strip())

        # Generate explanation
        explanation_prompt = prompt_out.explanation_prompt_template.format(answer=answer_text)
        expl_resp = self._ollama_client.generate(
            model=self._ollama_model,
            prompt=explanation_prompt,
            options={"temperature": 0.5, "num_predict": 256},
        )
        expl_text = expl_resp["response"].strip()

        return LLMResponse(
            question=question,
            answer=answer_text,
            explanation=expl_text,
            full_response=answer_text + " " + expl_text,
            model_id=self._ollama_model,
            backend="ollama",
            # Note: attention/gradients not available in Ollama mode
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """
        Parse the final answer from CoT output.
        Handles patterns like: "Answer: Paris", "The answer is Paris"
        Falls back to returning the full text if no pattern found.
        """
        patterns = [
            r"Answer:\s*(.+?)(?:\n|$)",
            r"answer is[:\s]+(.+?)(?:\n|$)",
            r"answer[:\s]+(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # Return first sentence as fallback
        return text.split(".")[0].strip() if "." in text else text.strip()
