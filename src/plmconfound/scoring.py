"""Masked-marginal scoring mechanics for protein language models.

Everything downstream of this module is a log-odds `log P(mut | context) - log P(wt |
context)`; the only thing that varies is what "context" means. Three protocols are
implemented here because they disagree, and the disagreement is itself a measurement:

  additive     each position scored with only itself masked (the ESM-1v protocol used by
               ProteinGym and by the metal-coordination study this work continues).
  joint        both positions of a pair masked in the same forward pass.
  conditional  one position substituted, the other scored given that substitution --
               symmetrised over the two orderings.

`joint - additive` and `conditional - additive` are the model's implied pairwise
epistasis. The additive protocol cannot express it by construction, so any non-zero
value is signal the standard benchmark throws away.
"""

import torch
import torch.nn.functional as F

# Fixed order. Never sort or derive this from a tokenizer vocabulary: downstream
# arrays are positional and a reordering would silently permute results.
AA = list("ACDEFGHIKLMNPQRSTVWY")


def mask_token_id(tok):
    """ESM2's HF tokenizer exposes `mask_token_id`; EvolutionaryScale's `EsmcTokenizer`
    does not always, so fall back to resolving the literal `<mask>` token."""
    return getattr(tok, "mask_token_id", None) or tok.convert_tokens_to_ids("<mask>")


def _aa_token_ids(tok):
    return [tok.convert_tokens_to_ids(a) for a in AA]


def _input_device(model):
    """Where the input tensors must live.

    Under `accelerate.cpu_offload` every *parameter* stays in system RAM while
    submodules stream to the execution device, so "no CUDA parameter" does not mean
    "CPU model" -- the inputs still belong on the GPU.
    """
    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _encode(tok, seq, subs=None):
    s = list(seq)
    for p, a in (subs or {}).items():
        s[p - 1] = a                       # `subs` is 1-indexed like the DMS mutant strings
    return tok("".join(s), return_tensors="pt", padding=True)


def _forward(model, enc_ids, enc_mask, device):
    kw = {"input_ids": enc_ids.to(device)}
    if enc_mask is not None:
        kw["attention_mask"] = enc_mask.to(device)
    with torch.inference_mode():
        return model(**kw).logits


def masked_logprobs(model, tok, seq, positions, subs=None):
    """Mask all of `positions` (1-indexed) JOINTLY in one forward pass, after optionally
    applying `subs` {pos: aa}. -> {pos: {aa: logprob}}.

    Masking k positions in a single pass yields each position's conditional given the
    rest of the sequence with all k positions hidden -- that is the joint/double-mask
    quantity, not k independent marginals. Call this once per position if independent
    marginals are what you want.
    """
    positions = list(positions)
    enc = _encode(tok, seq, subs)
    ids = enc["input_ids"]
    mask_id = mask_token_id(tok)
    for p in positions:
        ids[0, p] = mask_id                # token index is `p`, not `p-1`: +1 for BOS/<cls>
    logits = _forward(model, ids, enc.get("attention_mask"), _input_device(model))
    # log_softmax in fp32: fp16 weights are fine, but a fp16 softmax over a ~33-token
    # vocabulary loses digits exactly where the small log-odds differences live.
    lp = F.log_softmax(logits[0].float(), dim=-1)
    aa_ids = _aa_token_ids(tok)
    return {p: {a: lp[p, t].item() for a, t in zip(AA, aa_ids)} for p in positions}


def masked_logprobs_batched(model, tok, seq, positions, batch_size=32, subs=None):
    """Independent masked marginals: one masked variant per row, `batch_size` rows per
    forward pass. -> {pos: {aa: logprob}}, same contract as `masked_logprobs` called
    once per position (each row hides exactly one position).

    This exists for CPU offload. There the dominant cost is not arithmetic but streaming
    the weights across PCIe once per forward pass; masked variants are independent, so B
    of them amortise a single weight stream. Measured on ESMC-6B: 1.77 s/position at
    batch 1 versus 0.23 s/position at batch 128 -- a 7.7x speedup at 2.93 GB peak VRAM.
    """
    positions = list(positions)
    enc = _encode(tok, seq, subs)
    base_ids, base_mask = enc["input_ids"], enc.get("attention_mask")
    mask_id = mask_token_id(tok)
    device = _input_device(model)
    aa_ids = _aa_token_ids(tok)

    out = {}
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        ids = base_ids.repeat(len(chunk), 1)
        for r, p in enumerate(chunk):
            ids[r, p] = mask_id            # +1 BOS/<cls> offset, as above
        mask = base_mask.repeat(len(chunk), 1) if base_mask is not None else None
        logits = _forward(model, ids, mask, device)
        for r, p in enumerate(chunk):
            lp = F.log_softmax(logits[r, p].float(), dim=-1)
            out[p] = {a: lp[t].item() for a, t in zip(AA, aa_ids)}
    return out


def pair_scores(model, tok, seq, i, j, mi, mj):
    """Three scoring schemes for the double mutant (i->mi, j->mj), 1-indexed.

    additive     = sum_k [ log P(mut_k | x_\\k) - log P(wt_k | x_\\k) ]          (ESM-1v)
    joint        = sum_k [ log P(mut_k | x_\\{i,j}) - log P(wt_k | x_\\{i,j}) ]  (both masked)
    conditional  = 0.5 * [ path(i->j) + path(j->i) ], where
                   path(i->j) = [log P(mi|x_\\i) - log P(wi|x_\\i)]
                              + [log P(mj|x with i:=mi, \\j) - log P(wj|x with i:=mi, \\j)]

    The symmetrisation is mandatory, not cosmetic: path(i->j) != path(j->i) because the
    second step of each path conditions on a different already-substituted residue.
    Reporting one ordering would make the statistic depend on which cysteine happens to
    be listed first.

    epsilon = (conditional or joint) - additive is the model's implied pairwise
    epistasis. The additive protocol is epsilon-blind BY CONSTRUCTION -- it never sees
    the two positions at once -- so the question is whether the model's own
    representation carries a non-zero epsilon at real chemistry, with the correct sign.

    Validation (ESM2-650M, see recon report §4.1): `eps_joint_minus_add` is
    **+4.41 nats** at the genuine TEM-1 C75/C121 disulfide (`1BTL`:
    `SSBOND CYS A 77 CYS A 123`), **+0.34** at GFP C48/C70 -- a free-thiol pair, since
    `1EMA` carries zero SSBOND records -- and **+0.003** at TEM-1 G76/S122, a non-Cys
    pair with identical sequence separation. The effect tracks the bond, not the
    distance.

    Costs five forward passes (two marginals, one double mask, two conditionals).
    """
    wi, wj = seq[i - 1], seq[j - 1]
    li = masked_logprobs(model, tok, seq, [i])[i]
    lj = masked_logprobs(model, tok, seq, [j])[j]
    additive = (li[mi] - li[wi]) + (lj[mj] - lj[wj])

    both = masked_logprobs(model, tok, seq, [i, j])
    joint = (both[i][mi] - both[i][wi]) + (both[j][mj] - both[j][wj])

    lj_gi = masked_logprobs(model, tok, seq, [j], subs={i: mi})[j]
    li_gj = masked_logprobs(model, tok, seq, [i], subs={j: mj})[i]
    path_ij = (li[mi] - li[wi]) + (lj_gi[mj] - lj_gi[wj])
    path_ji = (lj[mj] - lj[wj]) + (li_gj[mi] - li_gj[wi])
    cond = 0.5 * (path_ij + path_ji)

    return dict(additive=additive, joint_doublemask=joint, conditional_sym=cond,
                eps_cond_minus_add=cond - additive, eps_joint_minus_add=joint - additive)
