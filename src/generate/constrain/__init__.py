import torch
from enum import Enum, auto
from transformers import LogitsProcessor
from torch import Tensor, FloatTensor
import logging
from typing import List, Tuple, Optional, Set
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
            return State.THOUGHT_CONTENT, "<thought>\n"
        case State.THOUGHT_CONTENT:
            # First ensure the opening tag is there
            if "<thought>" not in text:
                return state, "<thought>\n"
            # Then look for closing tag
            if "</thought>" in text:
                return State.ACTION_OPEN, "\n<action>"
            # In between, allow free generation
            return state, ""
        case State.ACTION_OPEN if "<action>" in text:
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

    def _tokens_prefixing(self, text: str) -> Set[int]:
        """
        Returns the IDs of tokens that are prefixes of the given text (i.e. the given text starts with them).
        """
        # Slow.
        toks: Set[int] = set()
        for token_id in self.tokenizer.get_vocab().values():
            token_text = self.token_to_text[token_id]
            if text.startswith(token_text):
                toks.add(token_id)
        return toks

    def _tokens_starting_with(self, text: str) -> Set[int]:
        """
        Returns the IDs of tokens for which the given text is a prefix (i.e. tokens that start with the given text).
        """
        # Slow.
        toks: Set[int] = set()
        for token_id in self.tokenizer.get_vocab().values():
            token_text = self.token_to_text[token_id]
            if token_text.startswith(text):
                toks.add(token_id)
        return toks

    def _compute_matched_length(self, generated_text: str) -> int:
        """
        Compute how many characters of target_sequence are already present at the end of generated_text.
        """
        # No target sequence -> nothing to match
        if not self.target_sequence:
            return 0

        # Look at the end of generated_text, up to the length of target_sequence
        suffix = generated_text[-len(self.target_sequence):]

        # Now compute the match starting from this suffix
        matched_length = 0
        for i in range(min(len(suffix), len(self.target_sequence))):
            if suffix[i] == self.target_sequence[i]:
                matched_length += 1
            else:
                break

        return matched_length

    def _forbid_illegal_tokens_for_target(self, generated_text: str, scores: FloatTensor):
        self.target_position = self._compute_matched_length(generated_text)
        logger.debug(f"Target progress: {self.target_position}/{len(self.target_sequence)}")
        remaining_target = self.target_sequence[self.target_position:]

        # Not currently enforcing
        if not remaining_target:
            return set()

        logger.debug(f"Blocking illegal tokens for target: {repr(remaining_target)}")

        # If we have no constrained generation, all tokens would be allowed.
        allowed_tokens = set()

        # If a token is a prefix of the remaining target, allow it
        # That's unequivocally good: it is a possible step toward the target.
        allowed_tokens.update(self._tokens_prefixing(remaining_target))

        # If the remaining target is short, it's possible that we'll find no tokens that prefix it.
        # Then we hit this branch.
        if not allowed_tokens:
            # XXX: might allow illegal completions
            # The other way around: maybe the remaining target is the prefix of a token.
            # This, however, means we can get undesired trailing characters.
            # This is rather context-dependent, and there's more to it than just whitespace.
            # For example, what if you have a closing tag then an opening tag? Maybe '><' is one token.
            # But whether to allow it depends on whether we will want to consume a '<' in the future.
            #
            # For now, we'll just allow it, and count on the model to be coherent.
            allowed_tokens.update(self._tokens_starting_with(remaining_target))

        all_tokens = set(self.tokenizer.get_vocab().values())
        forbidden = all_tokens - allowed_tokens

        if forbidden:
            allowed_pairs = [(self.token_to_text[token_id], token_id) for token_id in allowed_tokens]
            logger.debug(f"Allowing {len(allowed_tokens)} tokens: {allowed_pairs} / blocking {len(forbidden)} forbidden tokens")
        for token_id in forbidden:
            scores[0, token_id] = float("-inf")

        logger.debug(f"Updated target position: {self.target_position}")

        if self.target_sequence in generated_text:
            logger.debug(f"Completed target: {repr(self.target_sequence)}")
            self.target_sequence = ""
            self.target_position = 0



    def __call__(self, input_ids: Tensor, scores: FloatTensor) -> FloatTensor:
        assert scores.shape[0] == 1, "Only batch size 1 is supported"

        self.step += 1
        logger.debug(f"\n{'='*40} STEP {self.step} {'='*40}, state = {self.state.name}")
        logger.debug(f"Target sequence: {repr(self.target_sequence)}")

        # First call.
        if self.start_pos is None:
            self.start_pos = input_ids.shape[1]

        # EOS <=> DONE.
        if self.state == State.DONE:
            return force_token(scores, self.eos_token_id)
        else:
            scores[0, self.eos_token_id] = float("-inf")

        generated_tokens = input_ids[0, self.start_pos:].tolist()

        if generated_tokens:
            last_generated_pair: Tuple[int, str] = (generated_tokens[-1], self.token_to_text[generated_tokens[-1]])
            logger.debug(f"Last generated token: {last_generated_pair}")

        generated_text = self.tokenizer.decode(generated_tokens, add_special_tokens=False)
        logger.debug(f"Generated text since beginning of completion: {repr(generated_text)}")

        # State transitions are string-based, not token-based.
        # When looking back, we don't care about token IDs.
        new_state, target_string = get_next_state(self.state, generated_text)

        if new_state != self.state:
            logger.debug(f"STATE TRANSITION: {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.target_position = 0
            if target_string:
                self.target_sequence = target_string
                logger.debug(f"New target sequence: {repr(self.target_sequence)}")
            else:
                self.target_sequence = ""
        elif target_string:
            self.target_sequence = target_string
            self.target_position = 0

        if self.target_sequence:
           self._forbid_illegal_tokens_for_target(generated_text, scores)

        # Sanity check.
        assert not torch.all(scores == float("-inf")), "All tokens are blocked. This is a bug."

        return scores
