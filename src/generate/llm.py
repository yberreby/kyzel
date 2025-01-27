"""
Core LLM functionality: initialization and generation with constraints.
"""

import logging
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from .constrain import StructuredEnforcer
from ..types.chatml import Conversation

log = logging.getLogger(__name__)

class LLM:
    """
    Handles model initialization and generation with appropriate constraints.
    """

    def __init__(self, model_name: str = "unsloth/Phi-4", chat_template="phi-4", max_seq_length: int = 2048):
        log.info(f"Initializing LLM with model {model_name}, chat template {chat_template}, max_seq_length {max_seq_length}")
        # Initialize model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        self.tokenizer = get_chat_template(
            tokenizer, chat_template
        ) # CAREFUL wrt mistmatches...
        FastLanguageModel.for_inference(model)
        self.model = model

    def generate(self, messages: Conversation, max_new_tokens: int = 512) -> str:
        """
        Generate a response given a conversation history.
        Returns raw text (still needs to be parsed).
        """
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")

        n_input = inputs.shape[1]
        log.info(f"Completing {n_input} tokens...")

        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[StructuredEnforcer(self.tokenizer)],
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=1,
        )

        # Always BS = 1.
        assert outputs.shape[0] == 1

        # Remove input tokens
        outputs = outputs[0, n_input:]

        n_output = outputs.shape[0]
        log.info(f"Generated {n_output} tokens")

        return self.tokenizer.decode(outputs, skip_special_tokens=False)
