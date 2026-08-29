"""Acceptor-chemistry behaviour: the preserving/abolishing contrast the study rests on.

These tests are written against the *biochemistry*, not the implementation. Each
assertion below is an enzymological claim with a source, so a failure here means either
the family definitions drifted or a claim in `PLAN.md` / `PRE_REGISTRATION.md` changed.
No network, no data files, no models: pure functions only.
"""

import pytest

from plmconfound.chemistry import (
    ASN_ACCEPTOR,
    GLYC_PLUS2_ACCEPTOR,
    LOCI,
    PHOSPHO_ACCEPTOR,
    classify,
    find_sequons,
    preserving_family,
)

# --------------------------------------------------------------------------------------
# phospho: {S,T} are interchangeable Ser/Thr-kinase acceptors
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("wt", "mut"), [("S", "T"), ("T", "S")])
def test_serthr_swap_is_preserving(wt, mut):
    """S<->T keeps a Ser/Thr-kinase site phosphorylatable, so it is the preserving cell."""
    got = classify("phospho", wt, mut)
    assert got == "preserving", f"phospho {wt}->{mut} should be preserving, got {got!r}"


@pytest.mark.parametrize("mut", ["A", "V", "D", "E", "G", "N"])
def test_phospho_loss_of_hydroxyl_is_abolishing(mut):
    """Any substitution that removes the phosphorylatable hydroxyl abolishes the site.

    D and E are included deliberately: they are the standard 'phosphomimetic' pair, but
    they cannot be *phosphorylated*, and this study's contrast is about the enzyme's
    substrate, not about electrostatic mimicry of the product.
    """
    got = classify("phospho", "S", mut)
    assert got == "abolishing", f"phospho S->{mut} should be abolishing, got {got!r}"


def test_tyr_phosphosite_has_empty_preserving_family():
    """A Tyr phosphosite has NO preserving partner: the one-member-family collapse.

    Tyrosine kinases are a different enzyme class from Ser/Thr kinases, so S and T are
    not substrates for the kinase that modified a phosphotyrosine. The acceptor family
    at a pTyr site is therefore {Y} alone, the preserving cell is empty by chemistry,
    and such a site must be excluded from the interaction statistic rather than counted
    with a zero-sized preserving effect.
    """
    fam = preserving_family("phospho", "Y")
    assert fam == frozenset(), f"pTyr preserving family must be empty, got {sorted(fam)}"
    for mut in ("S", "T", "F"):
        # F is the classic non-phosphorylatable Tyr control; S/T belong to the *other*
        # kinase class. All three destroy the modification at this locus.
        got = classify("phospho", "Y", mut)
        assert got == "abolishing", f"phospho Y->{mut} should be abolishing, got {got!r}"


def test_phospho_acceptor_constant_excludes_tyr():
    """The exported constant is the Ser/Thr-kinase family only; Y is intentionally absent."""
    assert PHOSPHO_ACCEPTOR == frozenset("ST"), f"unexpected family {sorted(PHOSPHO_ACCEPTOR)}"
    assert "Y" not in PHOSPHO_ACCEPTOR, "Tyr must not join the Ser/Thr-kinase acceptor family"


# --------------------------------------------------------------------------------------
# sequon_asn: the designed negative control
# --------------------------------------------------------------------------------------


def test_asn_acceptor_family_is_empty():
    """No residue but Asn accepts an N-glycan, so the preserving cell is empty a priori.

    This is exactly what makes `sequon_asn` a *designed negative control* rather than a
    third measurement arm: the chemistry account predicts no preserving/abolishing
    contrast can exist here at all, so a non-zero interaction at this locus would
    indicate a bug or a leak in the pipeline rather than a biological finding.
    """
    fam = preserving_family("sequon_asn", "N")
    assert fam == frozenset(), f"Asn acceptor preserving family must be empty, got {sorted(fam)}"
    assert ASN_ACCEPTOR == frozenset("N"), f"unexpected acceptor {sorted(ASN_ACCEPTOR)}"


@pytest.mark.parametrize("mut", ["D", "Q", "S", "T", "A"])
def test_asn_substitutions_are_all_abolishing(mut):
    """N->D and N->Q abolish glycosylation despite being steric/charge mimics.

    Asp is an isosteric charge mimic and Gln a one-methylene homologue; neither is an
    oligosaccharyltransferase substrate. Treating them as preserving would manufacture
    a preserving cell at the locus that exists precisely to have none.
    """
    got = classify("sequon_asn", "N", mut)
    assert got == "abolishing", f"sequon_asn N->{mut} should be abolishing, got {got!r}"


# --------------------------------------------------------------------------------------
# sequon_plus2: the only glycosylation locus with a real preserving cell
# --------------------------------------------------------------------------------------


def test_sequon_plus2_family_is_ser_thr_cys():
    """Family = {S,T,C}.

    Bause & Legler 1981, Biochem J 195:639-644 (PMID 7316978): the Thr-, Ser- and
    Cys-containing hexapeptide acceptors were all glycosylated, "although at very
    different rates" (Thr > Ser > Cys), whereas the Val and O-methyl-Thr analogues were
    not acceptors at all. Cys is therefore preserving-but-attenuated, not abolishing.
    """
    assert GLYC_PLUS2_ACCEPTOR == frozenset("STC"), (
        f"sequon +2 family must be {{S,T,C}}, got {sorted(GLYC_PLUS2_ACCEPTOR)}"
    )
    assert preserving_family("sequon_plus2", "T") == frozenset("STC")
    assert preserving_family("sequon_plus2", "S") == frozenset("STC")


@pytest.mark.parametrize(("wt", "mut"), [("T", "S"), ("T", "C"), ("S", "T"), ("S", "C")])
def test_sequon_plus2_competent_acceptors_are_preserving(wt, mut):
    """Swaps among the three competent +2 acceptors keep the sequon glycosylatable."""
    got = classify("sequon_plus2", wt, mut)
    assert got == "preserving", f"sequon_plus2 {wt}->{mut} should be preserving, got {got!r}"


@pytest.mark.parametrize("mut", ["A", "V", "G", "N", "Y"])
def test_sequon_plus2_non_acceptors_are_abolishing(mut):
    """V in particular is measured, not assumed: Bause & Legler's Val analogue was inert."""
    got = classify("sequon_plus2", "T", mut)
    assert got == "abolishing", f"sequon_plus2 T->{mut} should be abolishing, got {got!r}"


# --------------------------------------------------------------------------------------
# invariants that hold across every locus
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("locus", LOCI)
@pytest.mark.parametrize("wt", ["S", "T", "C", "N", "Y"])
def test_synonymous_row_is_never_preserving(locus, wt):
    """`mut == wt` must never land in the preserving cell.

    A wild-type-to-wild-type row is not a substitution; counting it as preserving would
    pad the preserving cell with wild-type-like DMS scores and manufacture the very
    effect the interaction statistic is supposed to detect.
    """
    got = classify(locus, wt, wt)
    assert got == "abolishing", f"{locus} {wt}->{wt} must not be preserving, got {got!r}"


@pytest.mark.parametrize("locus", ["disulfide", "sequon", "PHOSPHO", "", "sequon_plus1"])
def test_unknown_locus_raises(locus):
    """An unrecognised locus is a programming error and must fail loudly, not default."""
    with pytest.raises(ValueError):
        preserving_family(locus, "S")
    with pytest.raises(ValueError):
        classify(locus, "S", "T")


@pytest.mark.parametrize("locus", LOCI)
@pytest.mark.parametrize(("wt", "mut"), [("S", "T"), ("T", "C"), ("N", "D"), ("Y", "F")])
def test_classify_returns_only_two_labels(locus, wt, mut):
    """The statistic is a 2x2; `classify` must never emit a third label."""
    got = classify(locus, wt, mut)
    assert got in {"preserving", "abolishing"}, f"unexpected label {got!r}"


# --------------------------------------------------------------------------------------
# sequon motif search
# --------------------------------------------------------------------------------------


def test_find_sequons_single_hit_is_one_indexed():
    """`find_sequons` returns 1-indexed (asn, plus2) pairs, matching UniProt numbering."""
    hits = find_sequons("AANATB")
    assert hits == [(3, 5)], f"expected [(3, 5)] for 'AANATB', got {hits}"
    asn, plus2 = hits[0]
    assert plus2 - asn == 2, "plus2 must sit exactly two residues after the Asn"


def test_find_sequons_finds_two_sequons():
    """Both N-X-S and N-X-T forms are sequons and both must be reported, left to right."""
    seq = "MNASGGNQTKK"
    hits = find_sequons(seq)
    assert hits == [(2, 4), (7, 9)], f"expected [(2, 4), (7, 9)] for {seq!r}, got {hits}"
    for asn, plus2 in hits:
        assert seq[asn - 1] == "N", f"position {asn} is not the Asn in {seq!r}"
        assert seq[plus2 - 1] in "ST", f"position {plus2} is not S/T in {seq!r}"


def test_find_sequons_rejects_proline_at_x():
    """N-P-[ST] is not a sequon: Pro at +1 blocks the OST-competent turn.

    The second motif in this sequence is a genuine sequon, so the test distinguishes
    "Pro rejected" from "regex found nothing".
    """
    hits = find_sequons("NPTNAT")
    assert hits == [(4, 6)], f"N-P-T must be skipped and N-A-T kept; got {hits}"


def test_find_sequons_pure_proline_sequon_yields_nothing():
    """A sequence whose only N-X-[ST] candidate has X=Pro contributes no sequon at all."""
    assert find_sequons("MNPTKK") == [], "N-P-T must not be reported as a sequon"
    assert find_sequons("MNPSKK") == [], "N-P-S must not be reported as a sequon"


def test_find_sequons_requires_ser_or_thr_at_plus2():
    """N-X-Y is not a sequon; only Ser and Thr are glycosyl acceptors at +2.

    N-A-C is not a sequon either: Cys is competent at the +2 position of an already
    recognised N-X-S/T motif (Bause & Legler 1981), but it does not itself create one,
    so it must not enlarge the sequon inventory.
    """
    assert find_sequons("MNAYKK") == [], "N-A-Y must not be reported as a sequon"
    assert find_sequons("MNACKK") == [], "N-A-C must not be reported as a sequon"


def test_find_sequons_empty_and_short_sequences():
    """Sequences shorter than the motif must return empty, not raise."""
    assert find_sequons("") == []
    assert find_sequons("NA") == []
