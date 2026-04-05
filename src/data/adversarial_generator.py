"""
adversarial_generator.py
------------------------
Generates adversarial inputs to strictly test hallucination detection.
Techniques include False Premise Injection and Counterfactual Prompting.
"""

from typing import Dict, Any

class AdversarialGenerator:
    """
    Injects realistic adversarial noise or false premises into questions.
    Forces the LLM into 'hallucination-prone' states to test if FHI can catch it.
    """
    
    @staticmethod
    def inject_false_premise(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a valid question and prepends a misleading context.
        E.g., "Given that Obama was born in Kenya, where was he born?"
        """
        # In a full research setup, an LLM would dynamically build the false premise.
        # For evaluation, we prepend a standardized hard-coded false constraint.
        false_premise = "Assume for the sake of this question that historical facts have been altered and the following is true: The moon is made entirely of cheese. "
        
        adversarial_sample = sample.copy()
        adversarial_sample["original_question"] = sample["question"]
        adversarial_sample["question"] = false_premise + sample["question"]
        adversarial_sample["is_adversarial"] = True
        return adversarial_sample

    @staticmethod
    def inject_conflicting_context(sample: Dict[str, Any], conflicting_text: str) -> Dict[str, Any]:
        """
        Injects context that conflicts directly with the gold answer.
        """
        conflict_prompt = f"Background Info: {conflicting_text}\n\nQuestion: {sample['question']}"
        
        adversarial_sample = sample.copy()
        adversarial_sample["original_question"] = sample["question"]
        adversarial_sample["question"] = conflict_prompt
        adversarial_sample["is_adversarial"] = True
        return adversarial_sample
