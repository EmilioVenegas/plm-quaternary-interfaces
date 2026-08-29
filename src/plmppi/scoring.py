"""Masked-marginal zero-shot scoring for protein language models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def mask_token_id(tok: Any) -> int:
    """Extracts mask token ID from HuggingFace or ESMC tokenizer."""
    return getattr(tok, "mask_token_id", None) or tok.convert_tokens_to_ids("<mask>")


def _aa_token_ids(tok: Any) -> list[int]:
    return [tok.convert_tokens_to_ids(a) for a in AA_LIST]


def _input_device(model: Any) -> torch.device:
    """Determines execution device for input tensors."""
    dev = next((p.device for p in model.parameters() if p.device.type == "cuda"), None)
    if dev is not None:
        return dev
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _encode(tok: Any, seq: str) -> dict[str, torch.Tensor]:
    return tok(str(seq), return_tensors="pt", padding=True)


def _forward(
    model: Any,
    enc_ids: torch.Tensor,
    enc_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    kw = {"input_ids": enc_ids.to(device)}
    if enc_mask is not None:
        kw["attention_mask"] = enc_mask.to(device)
    with torch.inference_mode():
        out = model(**kw)
        return getattr(out, "logits", out)


def masked_logprobs_batched(
    model: Any,
    tok: Any,
    seq: str,
    positions: list[int] | set[int],
    batch_size: int = 32,
) -> dict[int, dict[str, float]]:
    """Calculates independent single-masked log probabilities for each position.

    Args:
        model: Loaded PLM.
        tok: Tokenizer.
        seq: Wild-type sequence string.
        positions: 1-indexed residue positions to mask.
        batch_size: Number of masked sequences per forward pass.

    Returns:
        Dict mapping position (1-indexed) -> {aa: log_prob}.
    """
    pos_list = sorted(set(positions))
    if not pos_list:
        return {}

    enc = _encode(tok, seq)
    base_ids = enc["input_ids"]
    base_mask = enc.get("attention_mask")
    m_id = mask_token_id(tok)
    device = _input_device(model)
    aa_ids = _aa_token_ids(tok)

    out: dict[int, dict[str, float]] = {}
    for start in range(0, len(pos_list), batch_size):
        chunk = pos_list[start : start + batch_size]
        ids = base_ids.repeat(len(chunk), 1)
        for r, p in enumerate(chunk):
            ids[r, p] = m_id  # 1-indexed matches +1 BOS token

        mask = base_mask.repeat(len(chunk), 1) if base_mask is not None else None
        logits = _forward(model, ids, mask, device)

        for r, p in enumerate(chunk):
            lp = F.log_softmax(logits[r, p].float(), dim=-1)
            out[p] = {a: float(lp[t].item()) for a, t in zip(AA_LIST, aa_ids)}

    return out


def score_variant_dataframe(
    model: Any,
    tok: Any,
    df: Any,
    target_seq: str,
    batch_size: int = 32,
) -> tuple[list[float], list[float]]:
    """Scores a DataFrame of variants with ['position', 'wt', 'mut'] columns.

    Returns:
        (zeroshot_scores, logp_wt_scores)
    """
    positions = df["position"].unique().tolist()
    lp_map = masked_logprobs_batched(model, tok, target_seq, positions, batch_size=batch_size)

    zs_scores = []
    logp_wt_scores = []
    for _, row in df.iterrows():
        p = int(row["position"])
        wt = str(row["wt"])
        mut = str(row["mut"])

        if p in lp_map:
            pos_dict = lp_map[p]
            lp_mut = pos_dict.get(mut, -99.0)
            lp_wt = pos_dict.get(wt, -99.0)
            zs = lp_mut - lp_wt
            zs_scores.append(zs)
            logp_wt_scores.append(lp_wt)
        else:
            zs_scores.append(float("nan"))
            logp_wt_scores.append(float("nan"))

    return zs_scores, logp_wt_scores
