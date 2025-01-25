from torch import FloatTensor


def force_token(scores: FloatTensor, token_id: int) -> FloatTensor:
    """
    Enforce generation of `token_id`: set its score to 0 and all others to -inf.
    """
    scores.fill_(float("-inf"))
    scores[0, token_id] = 0
    return scores
