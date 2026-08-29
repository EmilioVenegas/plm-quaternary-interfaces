"""
PTM acceptor chemistry: the definition of the study's preserving/abolishing contrast.

The whole confound test rests on splitting the substitutions available at a PTM site
into two families that a conservation-only account of a protein language model cannot
tell apart. A PLM that has learned only "this column of the MSA is invariant" must
penalise every substitution at a modified position roughly equally. A PLM that has
learned the *chemistry of the modification* must penalise substitutions that destroy
the acceptor group far more than substitutions that swap one competent acceptor for
another. So the families below are not similarity classes and are not BLOSUM
neighbourhoods -- they are enzymological statements about which side chains the
modifying enzyme can actually use as a substrate, taken from primary sources.

Family membership is deliberately narrow, because the failure mode that killed two
assays in the preceding metal-coordination study was a one-member family: if the
acceptor family at a site has exactly one member, then no non-wild-type substitution
can be modification-preserving, the preserving cell is empty, and the site contributes
nothing to the interaction statistic while still inflating the apparent site count.
Two of the three loci here are exactly that degenerate case, and saying so up front is
what keeps the coverage accounting honest:

  - ``phospho``     -- {S,T} are interchangeable Ser/Thr-kinase acceptors. A Tyr
                      phosphosite has NO preserving partner: tyrosine kinases are a
                      separate enzyme class from Ser/Thr kinases, so S and T are not
                      substrates for the kinase that modified that site. The acceptor
                      family at a Tyr site therefore has exactly one member.
  - ``sequon_asn``  -- the Asn of the N-X-S/T sequon. Family = {N} alone. No other
                      residue accepts an N-glycan; in particular N->D and N->Q are
                      steric/charge mimics that still abolish glycosylation, so they
                      are abolishing, not preserving. A modification-preserving non-WT
                      substitution cannot exist at this locus at all.
  - ``sequon_plus2`` -- the +2 position of the sequon. Family = {S,T,C}. This is the
                      ONLY locus in the glycosylation track where a preserving
                      substitution exists, and it is the only locus that carries an
                      *ordinal* prediction: Bause & Legler measured three different
                      acceptor rates for the three competent residues (Thr > Ser > Cys),
                      so the chemistry account predicts a graded fitness cost
                      Thr->Ser < Thr->Cys among substitutions that are all equally
                      "non-conservative" from a conservation standpoint. A
                      conservation-only account has no mechanism that produces a
                      three-level ordering within a single invariant column, so this
                      ordering is a discriminating prediction rather than a restatement
                      of the binary contrast.
"""

import re

# Asn-X-Ser/Thr, X != Pro. Proline at +1 is excluded because it blocks the
# oligosaccharyltransferase-competent turn; a sequon with X=P is not glycosylated.
SEQUON = re.compile(r"N[^P][ST]")

# Ser/Thr-kinase acceptors. Tyr is intentionally absent: it is an acceptor for a
# different enzyme class, so it neither joins nor is substitutable by this family.
PHOSPHO_ACCEPTOR = frozenset("ST")

# Bause & Legler 1981, Biochem J 195:639-644 (PMID 7316978): the threonine-, serine-
# and cysteine-containing hexapeptide derivatives could all be glycosylated, "although
# at very different rates" (~4x Vmax drop Thr->Ser, a further 2-3x drop Ser->Cys),
# whereas the valine and O-methylthreonine analogues did not work as glycosyl
# acceptors. Cys is therefore preserving-but-attenuated, not abolishing.
GLYC_PLUS2_ACCEPTOR = frozenset("STC")

# The glycosyl-acceptor residue itself. Exported so callers can assert that a mapped
# sequon position really is an Asn in the target sequence, not as a preserving family:
# see ``preserving_family`` for why the preserving family at that locus is empty.
ASN_ACCEPTOR = frozenset("N")

LOCI = ("phospho", "sequon_asn", "sequon_plus2")


def preserving_family(locus: str, wt: str) -> frozenset[str]:
    """Residues that keep the modification intact at ``locus`` given wild-type ``wt``.

    Returns the *acceptor* family, which may be empty. An empty return is the
    scientifically meaningful one-member-family collapse, not a missing case: at a Tyr
    phosphosite and at every sequon Asn there is no other residue the enzyme can use,
    so the preserving cell for that site is empty by chemistry and the site must be
    excluded from the interaction statistic rather than counted with a zero.
    """
    if locus == "phospho":
        # A Ser site can go to Thr and vice versa; a Tyr site has nowhere to go.
        return PHOSPHO_ACCEPTOR if wt in PHOSPHO_ACCEPTOR else frozenset()
    if locus == "sequon_asn":
        # Structurally empty, not conditionally empty: {N} minus the wild-type Asn
        # leaves nothing, and D/Q mimicry does not restore glycan attachment.
        return frozenset()
    if locus == "sequon_plus2":
        return GLYC_PLUS2_ACCEPTOR
    raise ValueError(f"unknown locus {locus!r}; expected one of {LOCI}")


def classify(locus: str, wt: str, mut: str) -> str:
    """Label a substitution ``wt``->``mut`` at ``locus`` as preserving or abolishing.

    ``mut == wt`` is abolishing rather than preserving because a synonymous row is not
    a substitution at all; letting it fall in the preserving cell would pad that cell
    with wild-type-like DMS scores and manufacture the effect being tested.
    """
    return "preserving" if mut != wt and mut in preserving_family(locus, wt) else "abolishing"


def find_sequons(seq: str) -> list[tuple[int, int]]:
    """Locate N-X-S/T sequons, returning 1-indexed ``(asn_pos, plus2_pos)`` pairs.

    Coordinates are 1-indexed to match UniProt feature numbering and ProteinGym mutant
    strings, so callers never have to add an off-by-one at the point of comparison.
    Overlapping sequons (N-X-S/T where the S/T is itself the N of nothing, or tandem
    NNST motifs) are reported as ``re.finditer`` finds them: non-overlapping,
    left-to-right, which is what the reconnaissance counts were computed with.
    """
    return [(m.start() + 1, m.start() + 3) for m in SEQUON.finditer(seq)]
