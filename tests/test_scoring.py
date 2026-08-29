"""Tests for zero-shot scoring mechanics."""

import numpy as np
import pytest
import torch
from plmppi.scoring import AA_LIST, _aa_token_ids, mask_token_id


class MockTokenizer:
    def __init__(self):
        self.mask_token_id = 32
        self.vocab = {a: i for i, a in enumerate(AA_LIST)}

    def convert_tokens_to_ids(self, tok):
        if tok == "<mask>":
            return self.mask_token_id
        return self.vocab.get(tok, 0)

    def __call__(self, seq, return_tensors="pt", padding=True):
        ids = torch.tensor([[0] + [self.vocab.get(a, 1) for a in seq] + [2]])
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


def test_mask_token_id():
    tok = MockTokenizer()
    assert mask_token_id(tok) == 32


def test_aa_token_ids():
    tok = MockTokenizer()
    ids = _aa_token_ids(tok)
    assert len(ids) == 20
    assert ids[0] == tok.vocab["A"]
    assert ids[-1] == tok.vocab["Y"]
