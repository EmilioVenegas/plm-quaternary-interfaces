"""Do protein language models encode PTM chemistry, or only positional conservation?

Package for the confirmatory study specified in `PLAN.md` and frozen in
`PRE_REGISTRATION.md`. Module map:

    chemistry   preserving/abolishing families per locus; the sequon motif.
                The scientific core: `{S,T}` at a Ser/Thr phosphosite, `{S,T,C}` at
                sequon +2 (Bause & Legler 1981), and the empty family at the Asn
                acceptor that makes it a designed negative control.
    data        ProteinGym reference + assay loading, UniProt feature fetch, and the
                obsolete-entry repair that stops a failed lookup masquerading as a
                biological zero.
    mapping     alignment-based coordinate mapping. PDB, UniProt and ProteinGym
                `target_seq` indices are three different systems; assumed offsets are
                prohibited here.
    scoring     masked-marginal log-odds, batched scoring, and the additive vs
                joint vs symmetrized-conditional pair scores.
    models      the four measured-feasible model arms, including ESMC-6B via CPU offload.
    stats       within-assay residual, interaction statistic, stratified permutation test.

`scoring` and `models` are resolved lazily: they import `torch`, which costs seconds,
and a good deal of useful work (coverage counting, statistics, re-running a test from a
committed score table) needs neither. `import plmconfound` therefore stays cheap, and
`plmconfound.scoring` still works on first attribute access.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from . import chemistry, data, mapping, stats
from .chemistry import (
    ASN_ACCEPTOR,
    GLYC_PLUS2_ACCEPTOR,
    LOCI,
    PHOSPHO_ACCEPTOR,
    SEQUON,
    classify,
    find_sequons,
    preserving_family,
)
from .data import (
    BULK_DIR,
    PROTEINGYM_REF,
    fetch_uniprot_features,
    load_assay,
    load_reference,
    parse_features,
    parse_mutants,
    repair_sequenceless_entries,
    single_mutant_index,
)
from .mapping import build_map, pdb_ssbond
from .stats import (
    add_residual,
    interaction_stat,
    stratified_permutation_test,
    within_assay_z,
)

if TYPE_CHECKING:  # keeps type checkers aware of the lazily bound submodules
    from . import models, scoring

__version__ = "0.1.0"

_LAZY = {"scoring", "models"}

__all__ = [
    # submodules
    "chemistry", "data", "mapping", "stats", "scoring", "models",
    # chemistry
    "SEQUON", "PHOSPHO_ACCEPTOR", "GLYC_PLUS2_ACCEPTOR", "ASN_ACCEPTOR", "LOCI",
    "preserving_family", "classify", "find_sequons",
    # data
    "PROTEINGYM_REF", "BULK_DIR", "load_reference", "load_assay", "parse_mutants",
    "single_mutant_index", "fetch_uniprot_features", "repair_sequenceless_entries",
    "parse_features",
    # mapping
    "build_map", "pdb_ssbond",
    # stats
    "within_assay_z", "add_residual", "interaction_stat", "stratified_permutation_test",
    "__version__",
]


def __getattr__(name: str):
    """Bind `plmconfound.scoring` / `plmconfound.models` on first access (PEP 562)."""
    if name in _LAZY:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
