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


def score_concatenated_interface_variants(
    model: Any,
    tok: Any,
    df_variants: Any,
    struct_dir: Any = None,
    ref_df: Any = None,
    batch_size: int = 16,
) -> list[float]:
    """Scores interface variants under concatenated co-sequence conditioning.

    Constructs multi-chain prompt: <cls> Target_Seq <eos> Partner_Seq <eos>
    and computes masked-marginal zero-shot scores for interface mutations.
    """
    from pathlib import Path
    from Bio.PDB import PDBParser, is_aa
    from Bio.SeqUtils import seq1
    from plmppi.data import PRIMARY_SYSTEMS, load_reference

    if struct_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        struct_dir = repo_root / "data" / "structures"
    else:
        struct_dir = Path(struct_dir)

    if ref_df is None:
        ref_df = load_reference()

    parser = PDBParser(QUIET=True)
    sys_lookup = {s.system_id: s for s in PRIMARY_SYSTEMS}

    m_id = mask_token_id(tok)
    device = _input_device(model)
    aa_ids = _aa_token_ids(tok)

    all_scores: list[float] = []

    for sys_id, group in df_variants.groupby("system", sort=False):
        if sys_id not in sys_lookup:
            all_scores.extend([float("nan")] * len(group))
            continue

        sys_obj = sys_lookup[sys_id]
        match = ref_df.query("DMS_id == @sys_obj.dms_abundance")
        if match.empty:
            all_scores.extend([float("nan")] * len(group))
            continue
        t_seq = match.iloc[0]["target_seq"]

        pdb_path = struct_dir / f"{sys_obj.pdb_id}.pdb"
        if not pdb_path.exists():
            all_scores.extend([float("nan")] * len(group))
            continue

        st = parser.get_structure(sys_id, str(pdb_path))
        partner_seqs = [
            "".join(seq1(r.get_resname()) for r in st[0][pc] if is_aa(r))
            for pc in sys_obj.partner_chains
            if pc in st[0]
        ]
        full_partner_seq = ":".join(partner_seqs) if partner_seqs else ""

        pos_list = sorted(group["position"].unique().tolist())
        if not pos_list:
            continue

        # Handle long target sequences (e.g. Spike 1273 aa) by windowing around interface positions
        if len(t_seq) + len(full_partner_seq) > 1020:
            min_p = min(pos_list)
            max_p = max(pos_list)
            pad_start = max(1, min_p - 50)
            pad_end = min(len(t_seq), max_p + 50)
            cropped_t_seq = t_seq[pad_start - 1 : pad_end]
            offset = pad_start - 1
            enc = tok(cropped_t_seq, full_partner_seq, return_tensors="pt")
        else:
            offset = 0
            enc = tok(t_seq, full_partner_seq, return_tensors="pt")

        base_ids = enc["input_ids"]
        base_mask = enc.get("attention_mask")

        lp_map: dict[int, dict[str, float]] = {}
        for start in range(0, len(pos_list), batch_size):
            chunk = pos_list[start : start + batch_size]
            ids = base_ids.repeat(len(chunk), 1)
            for r, p in enumerate(chunk):
                token_idx = p - offset
                if token_idx < ids.shape[1]:
                    ids[r, token_idx] = m_id

            mask = base_mask.repeat(len(chunk), 1) if base_mask is not None else None
            logits = _forward(model, ids, mask, device)

            for r, p in enumerate(chunk):
                token_idx = p - offset
                if token_idx < logits.shape[1]:
                    lp = F.log_softmax(logits[r, token_idx].float(), dim=-1)
                    lp_map[p] = {a: float(lp[t].item()) for a, t in zip(AA_LIST, aa_ids)}

        for _, row in group.iterrows():
            p = int(row["position"])
            wt = str(row["wt"])
            mut = str(row["mut"])
            if p in lp_map:
                lp_mut = lp_map[p].get(mut, -99.0)
                lp_wt = lp_map[p].get(wt, -99.0)
                all_scores.append(lp_mut - lp_wt)
            else:
                all_scores.append(float("nan"))

    return all_scores
