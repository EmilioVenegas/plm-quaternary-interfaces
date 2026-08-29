"""Model registry and checkpoint loaders for ESM2 and ESMC architectures."""

from __future__ import annotations

import gc
import glob
import os
from typing import Any

import torch

MODELS: dict[str, dict[str, Any]] = {
    "esm2-650m": {
        "repo": "facebook/esm2_t33_650M_UR50D",
        "family": "esm2",
        "params_b": 0.65,
        "default_mode": "gpu",
        "notes": "Primary ESM2 checkpoint: 650M parameters, fits in VRAM (< 1.5 GB fp16).",
    },
    "esm2-3b": {
        "repo": "facebook/esm2_t36_3B_UR50D",
        "family": "esm2",
        "params_b": 2.84,
        "default_mode": "gpu",
        "notes": "Large ESM2 checkpoint: 2.84B parameters, ~5.8 GB fp16 on GPU.",
    },
    "esmc-600m": {
        "repo": "biohub/ESMC-600M",
        "family": "esmc",
        "params_b": 0.58,
        "default_mode": "gpu",
        "notes": "EvolutionaryScale ESMC 600M parameters, resident in GPU VRAM.",
    },
    "esmc-6b": {
        "repo": "biohub/ESMC-6B",
        "family": "esmc",
        "params_b": 6.35,
        "default_mode": "offload",
        "notes": "EvolutionaryScale ESMC 6.35B parameters, runs via CPU offload with batching.",
    },
}


def resolve_model(spec: str) -> tuple[str, dict[str, Any]]:
    """Resolves a model short name or HuggingFace repo ID."""
    if spec in MODELS:
        return spec, MODELS[spec]
    for name, entry in MODELS.items():
        if entry["repo"] == spec:
            return name, entry
    raise ValueError(f"Unknown model {spec!r}; available: {sorted(MODELS)}")


def load_model(spec: str, mode: str = "auto") -> tuple[Any, Any]:
    """Loads model and tokenizer. mode='auto' takes registry default."""
    name, entry = resolve_model(spec)
    if mode == "auto":
        mode = entry["default_mode"]

    if entry["family"] == "esmc":
        return _load_esmc(entry["repo"], mode)
    return _load_esm2(entry["repo"], mode)


def _load_esm2(repo: str, mode: str, dtype: torch.dtype = torch.float16) -> tuple[Any, Any]:
    """Loads ESM2 checkpoint with sharded bin fallback if needed."""
    if mode != "gpu":
        raise ValueError(f"ESM2 loader supports mode='gpu' only, got {mode!r}")

    from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM

    tok = AutoTokenizer.from_pretrained(repo)
    snap = glob.glob(
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/*"
        )
    )
    shards = sorted(p for p in snap if p.endswith(".bin"))

    if not shards:
        model = EsmForMaskedLM.from_pretrained(repo, torch_dtype=dtype)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tok


def _load_esmc(repo: str, mode: str, dtype: torch.dtype = torch.float16) -> tuple[Any, Any]:
    """Loads EvolutionaryScale ESMC model."""
    try:
        from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
    except ImportError as err:
        raise ImportError(
            "ESMC models require the 'esm' package in .venv-esmc: "
            ".venv-esmc/bin/python scripts/02_score_models.py ..."
        ) from err

    tok = EsmcTokenizer()
    if mode == "gpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EsmcForMaskedLM.from_pretrained(repo, device=device, dtype=dtype)
        return model.eval(), tok

    if mode == "offload":
        from accelerate import cpu_offload

        model = EsmcForMaskedLM.from_pretrained(repo, device="cpu", dtype=dtype)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu_offload(model, execution_device=device)
        return model.eval(), tok

    raise ValueError(f"Unsupported mode {mode!r} for ESMC")
