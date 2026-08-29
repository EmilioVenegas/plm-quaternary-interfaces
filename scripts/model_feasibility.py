"""Model feasibility harness for the PTM/disulfide confound study.

Answers, by measurement rather than assumption: which protein language models can run the
masked-marginal scoring protocol on an RTX 4060 Laptop (8,188 MiB VRAM) with 62 GB of
system RAM? Produces the table in `research/recon_ptm_disulfide_results.md` §7.1.

The key observation is that the workload is small. Masked-marginal scoring needs one
forward pass per masked position, so all of Track A is ~1,500 positions -- not millions.
Seconds-per-forward is therefore acceptable, which makes system RAM substitutable for
VRAM and dissolves the nominal 8GB ceiling.

Modes
  gpu      whole model resident on CUDA.
  offload  accelerate `cpu_offload`: weights live in system RAM, each submodule streams to
           the GPU on demand and is evicted. No quantisation -- fp16 rounding only.
           Bandwidth-bound, so batching independent masked variants amortises one weight
           stream over many positions.

ESMC note: `EsmcForMaskedLM` is NOT a transformers `PreTrainedModel` (it is
EvolutionaryScale's own `HubPreTrainedModel(nn.Module)`) and rejects `device_map=`.
`accelerate.cpu_offload` hooks arbitrary `nn.Module`s, so it works anyway.

Requires a separate venv: `pip install esm accelerate pandas` pulls its own torch and
would disturb the pinned environment that `recon_ptm_disulfide.py` is verified against.
    python -m venv .venv-esmc && .venv-esmc/bin/pip install esm accelerate pandas
    .venv-esmc/bin/python analysis/model_feasibility.py

Measured 2026-08-29 (see §7.1): ESM2-650M 2.68 GB; ESM2-3B 5.72-5.90 GB @ 0.20 s;
ESMC-600M 2.38 GB @ 0.022 s; ESMC-6B via offload 2.91 GB @ 0.21 s/position at batch 128
(8.6x faster per position than batch 1). ESM3 is excluded on data grounds, not hardware:
its open checkpoint removed ~4M viral sequences from training, which would confound
Track A's overwhelmingly viral glycosylation arm.
"""

import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
REF_CSV = REPO / "data" / "proteingym" / "DMS_substitutions_ref.csv"
OUT_JSON = REPO / "analysis" / "results" / "model_feasibility.json"
AA = list("ACDEFGHIKLMNPQRSTVWY")

# TEM-1 beta-lactamase disulfide, target_seq indices (PDB 1BTL SSBOND is Ambler 77/123).
# Used as the correctness probe: a working model must return C as argmax at both.
PROBE_ASSAY = "BLAT_ECOLX_Stiffler_2015"
PROBE_POS = (75, 121)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_esmc(repo, mode, dtype=torch.float16):
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
    if mode == "gpu":
        model = EsmcForMaskedLM.from_pretrained(repo, device="cuda", dtype=dtype).eval()
    elif mode == "offload":
        from accelerate import cpu_offload
        model = EsmcForMaskedLM.from_pretrained(repo, device="cpu", dtype=dtype).eval()
        model = cpu_offload(model, execution_device=torch.device("cuda:0"))
    else:
        raise ValueError(mode)
    return model, EsmcTokenizer()


def load_esm2(repo, dtype=torch.float16):
    """ESM2-3B ships only sharded `.bin` weights and transformers 5 refuses torch.load
    under torch<2.6, so the state dict is assembled by hand. Loading ONE shard silently
    yields 0.35B parameters instead of 2.84B -- always merge every shard."""
    import glob, os
    from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM
    tok = AutoTokenizer.from_pretrained(repo)
    snap = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/*"))
    shards = sorted(p for p in snap if p.endswith(".bin"))
    if not shards:
        model = EsmForMaskedLM.from_pretrained(repo, dtype=dtype)
    else:
        sd = {}
        for sh in shards:
            part = torch.load(sh, map_location="cpu", weights_only=True)
            sd.update({k: v.half() for k, v in part.items()})
            del part
            gc.collect()
        torch.set_default_dtype(torch.float16)
        model = EsmForMaskedLM(EsmConfig.from_pretrained(repo))
        torch.set_default_dtype(torch.float32)
        model.load_state_dict(sd, strict=False)
        del sd
        gc.collect()
    return model.to("cuda").eval(), tok


def mask_token_id(tok):
    return getattr(tok, "mask_token_id", None) or tok.convert_tokens_to_ids("<mask>")


def probe(model, tok, seq, positions=PROBE_POS):
    """Masked marginal at each position: one masked variant per forward."""
    out = {}
    enc = tok(seq, return_tensors="pt", padding=True)
    dev = next((p.device for p in model.parameters() if p.device.type == "cuda"), None) or "cuda"
    for pos in positions:
        ids = enc["input_ids"].clone()
        ids[0, pos] = mask_token_id(tok)          # +1 offset for BOS/cls
        with torch.inference_mode():
            logits = model(input_ids=ids.to(dev),
                           attention_mask=enc["attention_mask"].to(dev)).logits
        lp = F.log_softmax(logits[0, pos].float(), dim=-1)
        d = {a: lp[tok.convert_tokens_to_ids(a)].item() for a in AA}
        wt = seq[pos - 1]
        out[pos] = {"wt": wt, "argmax": max(d, key=d.get),
                    "wt_logprob": round(d[wt], 3),
                    "wt_to_A_logodds": round(d["A"] - d[wt], 3)}
    return out


def batch_scaling(model, tok, seq, batches=(1, 8, 32, 64, 128), n_positions=1500):
    """Masked variants are independent, so B of them share one weight stream. Under
    `cpu_offload` that streaming dominates, so cost per position falls with B."""
    base = tok(seq, return_tensors="pt", padding=True)
    rows = []
    for B in batches:
        ids = base["input_ids"].repeat(B, 1).clone()
        for r in range(B):
            ids[r, r + 1] = mask_token_id(tok)
        am = base["attention_mask"].repeat(B, 1)
        ids, am = ids.to("cuda"), am.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=am)          # warm
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=am)
            torch.cuda.synchronize()
            dt = time.time() - t0
            rows.append({"batch": B, "total_s": round(dt, 2),
                         "s_per_position": round(dt / B, 4),
                         "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                         "hours_for_%d_positions" % n_positions: round(n_positions * dt / B / 3600, 3)})
        except torch.OutOfMemoryError:
            rows.append({"batch": B, "oom": True})
            free()
            break
        free()
    return rows


def latency_by_length(model, tok, lengths=(286, 500, 800, 1300, 2000)):
    dev = next((p.device for p in model.parameters() if p.device.type == "cuda"), None) or "cuda"
    lat = {}
    for L in lengths:
        e = tok("A" * L, return_tensors="pt", padding=True)
        ids, am = e["input_ids"].to(dev), e["attention_mask"].to(dev)
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=am)          # warm
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=am)
            torch.cuda.synchronize()
            lat[L] = round(time.time() - t0, 3)
        except torch.OutOfMemoryError:
            lat[L] = "OOM"
            free()
            break
    return lat


CASES = [
    ("facebook/esm2_t33_650M_UR50D", "esm2", "gpu"),
    ("facebook/esm2_t36_3B_UR50D", "esm2", "gpu"),
    ("biohub/ESMC-600M", "esmc", "gpu"),
    ("biohub/ESMC-6B", "esmc", "gpu"),        # expected OOM: ~12.7GB fp16 vs 7.65GB usable
    ("biohub/ESMC-6B", "esmc", "offload"),
]


def main():
    seq = pd.read_csv(REF_CSV).query("DMS_id == @PROBE_ASSAY").iloc[0].target_seq
    results = {"vram_total_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
               "probe_assay": PROBE_ASSAY, "probe_positions": list(PROBE_POS), "runs": []}
    for repo, family, mode in CASES:
        free()
        rec = {"model": repo, "family": family, "mode": mode}
        t0 = time.time()
        try:
            model, tok = (load_esmc(repo, mode) if family == "esmc" else load_esm2(repo))
            rec["load_s"] = round(time.time() - t0, 1)
            rec["params_B"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)
            rec["masked_marginal"] = probe(model, tok, seq)
            rec["cys_recovered"] = all(v["argmax"] == "C" for v in rec["masked_marginal"].values())
            rec["forward_s_by_len"] = latency_by_length(model, tok)
            if mode == "offload":
                rec["batch_scaling"] = batch_scaling(model, tok, seq)
            rec["peak_vram_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            rec["ok"] = True
            del model
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        free()
        results["runs"].append(rec)
        print(json.dumps(rec, indent=1), flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
