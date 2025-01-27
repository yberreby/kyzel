import torch
from enum import Enum, auto
from transformers import LogitsProcessor
from torch import Tensor, FloatTensor
import logging
from typing import List, Tuple, Optional
from collections import defaultdict

from .logit_utils import force_token

logger = logging.getLogger(__name__)

code_start = "```python\n"

class State(Enum):
    START = auto()
    THOUGHT_CONTENT = auto()
    ACTION_OPEN = auto()
    ACTION_CONTENT = auto()
    CODE_FENCE_START = auto()
    CODE_CONTENT = auto()
    DONE = auto()

def get_code_block_status(text: str) -> Tuple[bool, bool]:
    if not text or code_start not in text:
        return False, False

    start_loc = text.find(code_start)
    lines = text[start_loc:].split("\n")[1:]
    for i, line in enumerate(lines):
        if line.startswith("```") and not line[3:].strip():
            has_content = any(l.strip() for l in lines[:i])
            return has_content, True
    return True, False

def get_next_state(state: State, text: str) -> Tuple[State, Optional[str]]:
    match state:
        case State.START:
            return State.THOUGHT_CONTENT, "<thought>"
        case State.THOUGHT_CONTENT if "</thought>" in text:
            return State.ACTION_OPEN, "\n<action>"
        case State.ACTION_OPEN:
            return State.ACTION_CONTENT, None
        case State.ACTION_CONTENT if "</action>" in text:
            return State.CODE_FENCE_START, "\n" + code_start
        case State.CODE_FENCE_START:
            return State.CODE_CONTENT, None
        case State.CODE_CONTENT:
            has_content, should_end = get_code_block_status(text)
            if has_content and should_end:
                return State.DONE, "\n"
    return state, None

class StructuredEnforcer(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.state = State.START
        self.start_pos = None
        self.token_to_text = defaultdict(str, {
            token_id: tokenizer.decode([token_id], add_special_tokens=False)
            for token_id in tokenizer.get_vocab().values()
        })
        self.target_sequence = ""
        self.target_position = 0
        self.step = 0

    def __call__(self, input_ids: Tensor, scores: FloatTensor) -> FloatTensor:
        self.step += 1
        logger.debug(f"\n{'='*40} STEP {self.step} {'='*40}")

        if self.start_pos is None:
            self.start_pos = input_ids.shape[1]
            logger.debug(f"Initialized start position: {self.start_pos}")

        scores[0, self.eos_token_id] = float("-inf")

        generated_tokens = input_ids[0, self.start_pos:].tolist()
        generated_text = self.tokenizer.decode(generated_tokens, add_special_tokens=False)
        logger.debug(f"Generated text: {repr(generated_text)}")

        new_state, target_string = get_next_state(self.state, generated_text)
        if new_state != self.state:
            logger.debug(f"STATE TRANSITION: {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.target_sequence = target_string or ""
            self.target_position = 0
            logger.debug(f"New target sequence: {repr(self.target_sequence)}")

        if self.target_sequence:
            new_text = self.tokenizer.decode(input_ids[0][self.start_pos:], add_special_tokens=False)
            matched_length = 0
            for i in range(min(len(new_text), len(self.target_sequence))):
                if new_text[i] == self.target_sequence[i]:
                    matched_length += 1
                else:
                    break
            self.target_position = matched_length

            remaining_target = self.target_sequence[self.target_position:]
            logger.debug(f"Target progress: {self.target_position}/{len(self.target_sequence)}")
            logger.debug(f"Remaining target: {repr(remaining_target)}")

            valid_tokens = []
            for token_id in self.tokenizer.get_vocab().values():
                token_text = self.token_to_text[token_id]
                if remaining_target.startswith(token_text):
                    valid_tokens.append(token_id)
                    # logger.debug(f"Valid token: {token_id} '{token_text}'")

            if not valid_tokens:
                for token_id in self.tokenizer.get_vocab().values():
                    token_text = self.token_to_text[token_id]
                    if token_text.startswith(remaining_target):
                        valid_tokens.append(token_id)
                        # logger.debug(f"Partial match token: {token_id} '{token_text}'")

            if valid_tokens:
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask[0, valid_tokens] = False
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                logger.warning("No valid tokens found, releasing constraints")
                self.target_sequence = ""

            logger.debug(f"Updated target position: {self.target_position}")

            if self.target_sequence in new_text:
                logger.debug(f"Completed target: {repr(self.target_sequence)}")
                self.target_sequence = ""
                self.target_position = 0

        if self.state == State.DONE:
            logger.debug("Forcing EOS token")
            scores = force_token(scores, self.eos_token_id)

        if torch.all(scores == float("-inf")):
            logger.error("All tokens blocked! Allowing EOS as fallback")
            scores[0, self.eos_token_id] = 0

        return scores
