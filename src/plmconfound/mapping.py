"""Coordinate reconciliation between PDB, UniProt and ProteinGym numbering.

Three numbering systems are in play and they agree by luck roughly half the time: PDB
residue numbers (author/legacy numbering), UniProt sequence positions (full-length
canonical isoform), and ProteinGym `target_seq` indices (whatever construct was assayed).
Nothing here ever assumes an offset; every relation is established by real pairwise
alignment.

The aligner object is built at import. Constructing it touches no files and opens no
sockets -- it only materialises BLASTP's substitution matrix.
"""

import requests
from Bio.Align import PairwiseAligner

# BLASTP scoring with BLAST's own affine gap costs (-11 open, -1 extend): the comparisons
# are same-protein construct-vs-canonical, where a construct truncation must be paid for
# once as one long gap rather than repeatedly as scattered short ones.
ALIGNER = PairwiseAligner(scoring="blastp")
ALIGNER.mode = "global"
ALIGNER.open_gap_score = -11
ALIGNER.extend_gap_score = -1


def build_map(uniprot_seq: str, target_seq: str) -> tuple[dict[int, int], float, int]:
    """-> (dict uniprot_1idx -> target_1idx, identity over aligned pairs, n_aligned).

    Mandatory: PDB numbers, UniProt numbers and ProteinGym target_seq indices are three
    different coordinate systems. This is the only sanctioned way to relate them.

    Two real traps this caught, either of which would have silently corrupted the study
    had an offset been assumed:

    * TEM-1 beta-lactamase. Its disulfide is `SSBOND CYS 77 - CYS 123` in PDB 1BTL, which
      uses Ambler numbering; the same two cysteines are at `target_seq` indices 75 and 121.
      A direct read of the SSBOND record would have scored the wrong pair of residues.
    * `MET_HUMAN_Estevam_2023`. The assay is a kinase-domain-only construct, so only 4 of
      MET's 13 annotated phosphosites fall inside the assayed region at all. Mapping
      exposes the 9 unmeasurable sites instead of mis-assigning them to whatever residue
      happens to sit at that index in the construct.

    The returned identity is computed over aligned pairs only (gaps excluded), so it is a
    check that the two sequences really are the same protein, not a coverage measure.
    """
    aln = ALIGNER.align(uniprot_seq, target_seq)[0]
    mp, same, tot = {}, 0, 0
    for (us, ue), (ts, _te) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(ue - us):
            u, t = us + k, ts + k
            mp[u + 1] = t + 1
            tot += 1
            same += uniprot_seq[u] == target_seq[t]
    return mp, (same / tot if tot else 0.0), tot


def pdb_ssbond(pdb_id: str) -> list[str]:
    """Real SSBOND records straight from RCSB -- the structural ground truth. Cheaper and
    more auditable than a full-PDB bulk mirror for a few hundred specific entries: the
    per-entry fetch is exact and instant, against roughly 1.6 TB for the bulk archive.

    Returns the raw record lines; residue numbers in them are PDB author numbering and
    MUST be passed through `build_map` before being used as sequence indices.
    """
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60)
    r.raise_for_status()
    return [ln for ln in r.text.splitlines() if ln.startswith("SSBOND")]
