"""
prompt_templates.py
-------------------
Prompt templates for the FHI pipeline.

Design rationale:
- We use two prompting strategies:
  1. Chain-of-Thought (CoT): Elicit step-by-step reasoning before the answer.
     This is standard in LLM faithfulness literature (Wei et al., 2022).
  2. Self-Explanation: Ask the model to explain WHY it gave the answer after
     generating it. This separates generation from attribution.

- The explanation is solicited AFTER the answer to avoid priming effects
  (i.e., the model justifying a predetermined answer vs. actually reasoning).

References:
  - Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
  - Turpin et al. (2023) "Language Models Don't Always Say What They Think"
"""

from dataclasses import dataclass
from enum import Enum


class PromptStrategy(str, Enum):
    CHAIN_OF_THOUGHT = "cot"
    SELF_EXPLANATION = "self_explain"
    DIRECT = "direct"


@dataclass
class PromptOutput:
    answer_prompt: str
    explanation_prompt_template: str   # Takes {answer} as placeholder
    strategy: PromptStrategy


class PromptTemplates:
    """
    All prompts used in the FHI pipeline.
    Prompts are carefully designed to:
    1. Elicit factual answers (not hallucinations by design)
    2. Force the model to generate a separable explanation token sequence
    3. Keep explanation grounded in the question context
    """

    @staticmethod
    def chain_of_thought(question: str) -> PromptOutput:
        """
        CoT prompt: model reasons step-by-step then gives a final answer.
        The reasoning steps ARE the explanation in this mode.
        """
        answer_prompt = f"""Answer the following question by thinking step by step.
Show your reasoning clearly, then provide your final answer on a new line starting with "Answer:".

Question: {question}

Reasoning:"""

        explanation_prompt_template = f"""Given the question: {question}
And the answer: {{answer}}

Explain in 2-3 sentences why this answer is correct, citing specific facts or reasoning steps."""

        return PromptOutput(
            answer_prompt=answer_prompt,
            explanation_prompt_template=explanation_prompt_template,
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        )

    @staticmethod
    def self_explanation(question: str) -> PromptOutput:
        """
        Self-explanation prompt: model gives answer first, then explains.
        This is the primary mode for XAI analysis, as the explanation tokens
        are generated separately and can be causally intervened upon.
        """
        answer_prompt = f"""Answer the following question concisely and factually.

Question: {question}

Answer:"""

        explanation_prompt_template = f"""Question: {question}
Answer: {{answer}}

Now explain your reasoning for this answer. Be specific about which facts or knowledge led to this conclusion:
Explanation:"""

        return PromptOutput(
            answer_prompt=answer_prompt,
            explanation_prompt_template=explanation_prompt_template,
            strategy=PromptStrategy.SELF_EXPLANATION,
        )

    @staticmethod
    def direct(question: str) -> PromptOutput:
        """
        Baseline direct prompt: no CoT, no explanation.
        Used for baseline comparison and log-prob extraction.
        """
        answer_prompt = f"""Answer this question in one sentence.

Question: {question}

Answer:"""

        # Note: we avoid embedding {question} in the template because .format(answer=...)
        # would try to resolve it as a key. Store question as a closed-over literal instead.
        question_escaped = question.replace("{", "{{").replace("}", "}}")
        explanation_prompt_template = (
            f"Why did you answer \"{question_escaped}\" with: {{answer}}? Explain briefly."
        )

        return PromptOutput(
            answer_prompt=answer_prompt,
            explanation_prompt_template=explanation_prompt_template,
            strategy=PromptStrategy.DIRECT,
        )

    @staticmethod
    def adversarial_probe(question: str, false_premise: str) -> str:
        """
        Adversarial prompt: embeds a false premise to induce hallucination.
        Used for generating the adversarial dataset subset.

        The false premise is injected naturally to avoid obvious detection.
        Design follows: Shi et al. (2023) "Large Language Models Can Be Easily Distracted"
        """
        return f"""Given that {false_premise}, answer the following question:

Question: {question}

Answer:"""
