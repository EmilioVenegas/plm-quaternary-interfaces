"""Model registry and loaders for the PTM/disulfide confound study.

The workload is small -- masked-marginal scoring needs one forward pass per masked
position, and the whole study is ~1,500 positions -- so seconds-per-forward is
acceptable. That makes system RAM substitutable for VRAM and dissolves the nominal 8 GB
ceiling of the target machine (RTX 4060 Laptop, 8,188 MiB, 62 GB system RAM).

Modes
  gpu      whole model resident on CUDA.
  offload  `accelerate.cpu_offload`: weights live in system RAM, each submodule streams
           to the GPU on demand and is evicted. No quantisation -- fp16 rounding only.
           Bandwidth-bound, so batching independent masked variants amortises one weight
           stream over many positions (see `scoring.masked_logprobs_batched`).

ESM2-650M is the primary checkpoint for continuity: the metal-coordination null this
work follows up was obtained with that exact model and protocol, so swapping checkpoints
would confound "different model" with "different chemistry".

**ESM3 is excluded on DATA grounds, not hardware.** ESM3-open's training set was filtered
for biosecurity: Hayes et al. (*Science* 2025; preprint 10.1101/2024.07.01.600583)
describe a Viral Denylist of ~4 million UniProt sequences annotated as viral, plus ~147k
non-viral and ~40k additional viral Select Agent sequences, removed from training. This
study's glycosylation arm is 110 of its 150 usable positions and is almost entirely viral
(HIV-1 Env, influenza HA, SARS-CoV-2 Spike, Zika E; only ACE2 is not). Scoring those
sequons with ESM3 would make the nuisance variable identical to the independent
variable: a poor score could not be attributed to a PTM-chemistry blind spot rather than
to deletion of the relevant sequences from pretraining.

The `esm` package (ESMC) is deliberately absent from the core environment -- it pulls its
own torch and would disturb the pinned environment the rest of the pipeline is verified
against. It is therefore imported lazily, inside the ESMC branch only:
    python -m venv .venv-esmc && .venv-esmc/bin/pip install esm accelerate
"""

import gc
import glob
import os

import torch

# Peak-VRAM and parameter figures below are measured, not nominal (recon report §7.1).
MODELS = {
    "esm2-650m": {
        "repo": "facebook/esm2_t33_650M_UR50D",
        "family": "esm2",
        "params_b": 0.65,
        "default_mode": "gpu",
        "notes": "Primary checkpoint (continuity with the metal-coordination null). "
                 "1.45 GB peak fp16, 0.031 s/forward at 286 residues.",
    },
    "esm2-3b": {
        "repo": "facebook/esm2_t36_3B_UR50D",
        "family": "esm2",
        "params_b": 2.84,
        "default_mode": "gpu",
        "notes": "5.93 GB peak fp16, 0.093 s/forward at 286 residues. Fits, but only "
                 "just; sharded-weight loading needs the manual path below.",
    },
    "esmc-600m": {
        "repo": "biohub/ESMC-600M",
        "family": "esmc",
        "params_b": 0.58,
        "default_mode": "gpu",
        "notes": "2.39 GB peak fp16, 0.022 s/forward -- the cheapest cross-family "
                 "replication of any ESM2 result. MIT, ungated.",
    },
    "esmc-6b": {
        "repo": "biohub/ESMC-6B",
        "family": "esmc",
        "params_b": 6.35,
        "default_mode": "offload",
        "notes": "Naive fp16 on GPU OOMs: 12.7 GB of weights against 7.65 GB usable "
                 "VRAM. Under CPU offload it runs at 2.93 GB peak and 0.23 s/position "
                 "at batch 128 (7.7x cheaper per position than batch 1).",
    },
}


def _resolve(spec):
    if spec in MODELS:
        return spec, MODELS[spec]
    for name, entry in MODELS.items():
        if entry["repo"] == spec:
            return name, entry
    raise ValueError(f"unknown model {spec!r}; known: {sorted(MODELS)}")


def load_model(spec, mode="auto"):
    """-> (model, tokenizer). `spec` is a registry short name or a full HF repo id;
    `mode="auto"` takes the registry's measured `default_mode` for that checkpoint."""
    _, entry = _resolve(spec)
    if mode == "auto":
        mode = entry["default_mode"]
    if entry["family"] == "esmc":
        return _load_esmc(entry["repo"], mode)
    return _load_esm2(entry["repo"], mode)


def _load_esm2(repo, mode, dtype=torch.float16):
    """ESM2-3B ships only sharded `.bin` weights and transformers 5 refuses `torch.load`
    under torch<2.6, so the state dict is assembled by hand from the HF cache.

    Loading ONE shard silently succeeds and yields a 0.35B-parameter model instead of
    2.84B -- `load_state_dict(strict=False)` swallows the missing keys and the randomly
    initialised remainder still returns plausible-looking logits. Always merge every
    shard, and check `sum(p.numel() for p in model.parameters())` against the registry.
    """
    if mode != "gpu":
        raise ValueError(f"esm2 loader supports mode='gpu' only, got {mode!r}")
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
            gc.collect()          # 3B fp32 shards; without this the peak RSS doubles
        torch.set_default_dtype(torch.float16)
        model = EsmForMaskedLM(EsmConfig.from_pretrained(repo))
        torch.set_default_dtype(torch.float32)
        model.load_state_dict(sd, strict=False)
        del sd
        gc.collect()
    return model.to("cuda").eval(), tok


def _load_esmc(repo, mode, dtype=torch.float16):
    """`EsmcForMaskedLM` is NOT a transformers `PreTrainedModel` -- it is
    EvolutionaryScale's own `HubPreTrainedModel(nn.Module)` -- so it rejects `device_map=`
    and the usual accelerate big-model entry points. `cpu_offload` hooks any `nn.Module`,
    so the offload path works anyway; that is the only reason ESMC-6B is reachable at all
    on 8 GB.
    """
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer   # lazy: separate venv

    if mode == "gpu":
        model = EsmcForMaskedLM.from_pretrained(repo, device="cuda", dtype=dtype).eval()
    elif mode == "offload":
        from accelerate import cpu_offload
        model = EsmcForMaskedLM.from_pretrained(repo, device="cpu", dtype=dtype).eval()
        model = cpu_offload(model, execution_device=torch.device("cuda:0"))
    else:
        raise ValueError(mode)
    return model, EsmcTokenizer()
