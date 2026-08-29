#!/usr/bin/env python
"""Render real PDB cartoon assets for the Figure 1 schematic using headless PyMOL.

Figure 1 panel A illustrates the monomer-folding confound with three cases. Rather than
drawing stylised blobs, this script renders the *actual* crystallographic complex behind
one of the study systems (`6H46`: KRAS G-domain + DARPin K55, the synthetic-binder arm of
the paired benchmark) and emits transparent PNG layers that `plot_figures.py` composites.

Why `6H46`: of the five benchmark complexes it is by far the best balanced for a small
schematic panel (chain A 170 CA vs chain B 157 CA). `6M0J` would render a 598-residue ACE2
dwarfing a 195-residue RBD; `1OLG` chains are only 42 residues each.

Layer contract
--------------
Every layer is rendered at the SAME canvas size with the SAME stored camera matrix, so
compositing them into one rectangle reproduces the true relative geometry of the complex.
That is what lets `plot_figures.py` place `target` and `partner_bound` on top of each other
and get a real interface, while swapping in `partner_away` for the dissociated case.

Outputs (docs/figures/assets/):
    mol_target.png         chain A cartoon, folded target monomer
    mol_target_mut.png     chain A + real interface residue shown as red spheres
    mol_partner_bound.png  chain B in its crystallographic (bound) position
    mol_partner_away.png   chain B rigid-body displaced along the interface normal
    mol_coil_ensemble.png  statistical-coil MODEL of the denatured state (not experimental)
    coil_ensemble.pdb      the coil coordinates, kept as a provenance artifact

Run with the PyMOL environment interpreter, e.g.
    ~/.conda/envs/figures/bin/python scripts/render_structures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymol2

REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCT_DIR = REPO_ROOT / "data" / "structures"
ASSET_DIR = REPO_ROOT / "docs" / "figures" / "assets"

PDB_ID = "6H46"
TARGET_CHAIN = "A"   # KRAS G-domain
PARTNER_CHAIN = "B"  # DARPin K55

CANVAS = 1200        # px, square; downscaled at composite time
INTERFACE_CUTOFF = 4.5

# Palette kept in sync with plot_figures.py panel A.
C_TARGET_SURF = "0x7dd3fc"    # light molecular surface
C_TARGET_PATCH = "0x0ea5e9"   # darker: the real 4.5 A contact epitope
C_PARTNER = "0xf43f5e"
C_PARTNER_PALE = "0xfda4af"
C_MUT = "0xdc2626"
C_COIL = "0xdc2626"         # representative denatured conformer
C_COIL_FAINT = "0xfca5a5"   # remaining ensemble members


def _style_common(cmd) -> None:
    """Cartoon styling shared by every layer: flat illustration look, no depth cue clutter."""
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("surface_quality", 1)
    cmd.set("transparency", 0.0)
    # Flat single-colour ribbons. Left on, PyMOL paints the ribbon underside in a
    # contrasting colour, which leaks the target's blue onto the rose partner.
    cmd.set("cartoon_highlight_color", -1)
    cmd.set("specular", 0.15)
    cmd.set("ambient", 0.32)
    cmd.set("direct", 0.55)
    cmd.set("reflect", 0.28)
    cmd.set("ray_shadow", 0)
    cmd.set("depth_cue", 0)
    cmd.set("fog", 0)
    # Thin dark outline reads as a schematic illustration rather than a photo render,
    # which keeps it coherent with the flat vector styling of the rest of the figure.
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_color", "0x1e293b")
    cmd.set("ray_trace_gain", 0.12)


def _interface_residues(cmd, obj: str) -> list[tuple[str, str]]:
    """Target-chain residues with any heavy atom within INTERFACE_CUTOFF of the partner.

    Matches `plmppi.interfaces.INTERFACE_DISTANCE_THRESHOLD` (4.5 A heavy-atom contact)
    so the residue flagged in the figure is drawn from the same definition the analysis
    uses to label interface variants."""
    cmd.select(
        "iface_res",
        f"byres (({obj} and chain {TARGET_CHAIN} and polymer and not hydrogens) "
        f"within {INTERFACE_CUTOFF} of ({obj} and chain {PARTNER_CHAIN} and not hydrogens))",
    )
    seen: dict[int, str] = {}
    for atom in cmd.get_model("iface_res and name CA").atom:
        seen[int(atom.resi)] = atom.resn
    return [(str(k), seen[k]) for k in sorted(seen)]


def _chain_centroid(cmd, obj: str, chain: str) -> np.ndarray:
    coords = cmd.get_coords(f"{obj} and chain {chain} and polymer")
    return np.asarray(coords, dtype=float).mean(axis=0)


# ---------------------------------------------------------------------------
# Statistical-coil (denatured state) model.
#
# A denatured protein is an ENSEMBLE, not a structure, and no experimental
# structure of denatured KRAS exists. So Case 2 shows a flexible-meccano style
# statistical coil built from the real chain A sequence: backbone dihedrals are
# sampled from coil-library basins and grown with steric rejection. It is a
# model, is labelled as one in the figure, and is rendered as tubes rather than
# a surface so it cannot be mistaken for the crystallographic layers.
# ---------------------------------------------------------------------------
B_N_CA, B_CA_C, B_C_N = 1.458, 1.525, 1.329
A_N_CA_C, A_CA_C_N, A_C_N_CA = np.deg2rad([111.2, 116.2, 121.7])

# (phi, psi, sd, weight) -- beta / polyproline-II / alpha_R / alpha_L
COIL_BASINS = [
    (-120.0, 130.0, 28.0, 0.34),
    (-75.0, 145.0, 18.0, 0.34),
    (-65.0, -40.0, 18.0, 0.26),
    (60.0, 40.0, 20.0, 0.06),
]
_COIL_W = np.array([b[3] for b in COIL_BASINS], dtype=float)
_COIL_W /= _COIL_W.sum()

COIL_N_CONF = 3      # one representative + two faint ensemble members
COIL_SEED = 11       # fixed -> byte-identical output on every run


def _nerf(a, b, c, bond, angle, torsion):
    """Position atom d given a-b-c, bond c-d, angle b-c-d and torsion a-b-c-d."""
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-8 else np.array([0.0, 0.0, 1.0])
    m = np.cross(n, bc)
    d = np.array([-bond * np.cos(angle),
                  bond * np.sin(angle) * np.cos(torsion),
                  bond * np.sin(angle) * np.sin(torsion)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def _sample_phi_psi(rng, resn):
    """Draw backbone dihedrals from the coil library, with Pro/Gly special-cased."""
    if resn == "PRO":                      # pyrrolidine ring restrains phi
        return (np.deg2rad(-65.0 + rng.normal(0, 8)),
                np.deg2rad(float(rng.choice([145.0, -35.0])) + rng.normal(0, 12)))
    if resn == "GLY":                      # achiral: samples both mirrors broadly
        phi = (-80.0 if rng.integers(0, 2) == 0 else 80.0) + rng.normal(0, 30)
        return np.deg2rad(phi), np.deg2rad(rng.uniform(-180, 180))
    phi, psi, sd, _ = COIL_BASINS[rng.choice(len(COIL_BASINS), p=_COIL_W)]
    return np.deg2rad(phi + rng.normal(0, sd)), np.deg2rad(psi + rng.normal(0, sd))


def _build_coil(resnames, rng, clash=4.2, tries=40):
    """Grow one self-avoiding backbone; returns (n_res, 3, 3) of N/CA/C coordinates."""
    n = len(resnames)
    N = np.zeros((n, 3)); CA = np.zeros((n, 3)); C = np.zeros((n, 3))
    CA[0] = [B_N_CA, 0.0, 0.0]
    C[0] = CA[0] + B_CA_C * np.array([np.cos(np.pi - A_N_CA_C), np.sin(np.pi - A_N_CA_C), 0.0])
    for i in range(1, n):
        best, best_d = None, -1.0
        for _ in range(tries):
            phi, psi = _sample_phi_psi(rng, resnames[i])
            omega = np.pi + rng.normal(0, np.deg2rad(3))
            ni = _nerf(N[i-1], CA[i-1], C[i-1], B_C_N, A_CA_C_N, psi)
            cai = _nerf(CA[i-1], C[i-1], ni, B_N_CA, A_C_N_CA, omega)
            ci = _nerf(C[i-1], ni, cai, B_CA_C, A_N_CA_C, phi)
            d = np.linalg.norm(CA[:i-2] - cai, axis=1).min() if i >= 3 else 1e9
            if d >= clash:
                best = (ni, cai, ci)
                break
            if d > best_d:
                best_d, best = d, (ni, cai, ci)
        N[i], CA[i], C[i] = best
    return np.stack([N, CA, C], axis=1)


def _bbox_diagonal(pts):
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _write_coil_pdb(path, confs, resnames):
    """Multi-MODEL PDB, one model per conformer -- kept as a provenance artifact."""
    names = (" N  ", " CA ", " C  ")
    with open(path, "w") as fh:
        fh.write("REMARK   1 STATISTICAL-COIL MODEL OF THE DENATURED STATE\n")
        fh.write("REMARK   1 NOT AN EXPERIMENTAL STRUCTURE\n")
        for m, coords in enumerate(confs, start=1):
            fh.write(f"MODEL     {m:>4d}\n")
            serial = 1
            for ri, resn in enumerate(resnames, start=1):
                for ai in range(3):
                    x, y, z = coords[ri-1, ai]
                    fh.write(f"ATOM  {serial:>5d} {names[ai]} {resn:>3s} A{ri:>4d}    "
                             f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           "
                             f"{names[ai].strip()[0]}\n")
                    serial += 1
            fh.write("TER\nENDMDL\n")

def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    pdb_path = STRUCT_DIR / f"{PDB_ID}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"missing structure {pdb_path}")

    meta: dict[str, object] = {"pdb_id": PDB_ID, "canvas_px": CANVAS,
                               "target_chain": TARGET_CHAIN, "partner_chain": PARTNER_CHAIN}

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(str(pdb_path), "cx")
        cmd.remove("solvent or not polymer")
        cmd.remove("hydrogens")
        _style_common(cmd)

        iface = _interface_residues(cmd, "cx")
        meta["interface_residues"] = [f"{resn}{resi}" for resi, resn in iface]

        # Pick the interface residue closest to the partner centroid as the mutation site --
        # deterministic, and genuinely at the quaternary contact.
        partner_c = _chain_centroid(cmd, "cx", PARTNER_CHAIN)
        best, best_d = None, 1e9
        for resi, resn in iface:
            ca = cmd.get_coords(f"cx and chain {TARGET_CHAIN} and resi {resi} and name CA")
            if ca is None:
                continue
            d = float(np.linalg.norm(np.asarray(ca[0], dtype=float) - partner_c))
            if d < best_d:
                best, best_d = (resi, resn), d
        mut_resi, mut_resn = best
        meta["mutation_site"] = f"{mut_resn}{mut_resi}"

        # ---- Establish ONE camera for every layer -------------------------------------
        cmd.hide("everything")
        cmd.show("cartoon", "polymer")
        cmd.orient("cx")
        cmd.zoom("cx", buffer=14.0)  # headroom so the displaced partner never clips
        view = cmd.get_view()
        meta["view"] = list(view)

        def render(name: str) -> None:
            cmd.set_view(view)
            out = ASSET_DIR / f"{name}.png"
            cmd.png(str(out), width=CANVAS, height=CANVAS, dpi=300, ray=1)
            print(f"[rendered] {out}")

        # Target is drawn as a molecular SURFACE, partner as cartoon. Two cartoons side
        # by side read as "two proteins near each other"; a cartoon nestling into a
        # surface groove reads as binding. Polar-contact dashes were tried and are
        # invisible at the ~1.9 in final panel width, so shape complementarity plus a
        # darker interface patch carries the interaction instead.

        # ---- Layer: folded target ------------------------------------------------------
        cmd.hide("everything")
        cmd.show("surface", f"cx and chain {TARGET_CHAIN}")
        cmd.color(C_TARGET_SURF, f"cx and chain {TARGET_CHAIN}")
        cmd.color(C_TARGET_PATCH, "iface_res")   # real 4.5 A contact patch
        render("mol_target")

        # ---- Layer: target with the real interface mutation site flagged ---------------
        # On a surface the mutated residue reads as a coloured patch inside the epitope,
        # which shows the contact surface being disrupted far better than spheres did.
        cmd.color(C_MUT, f"cx and chain {TARGET_CHAIN} and resi {mut_resi}")
        render("mol_target_mut")

        # ---- Layer: partner in its crystallographic bound pose ------------------------
        cmd.hide("everything")
        cmd.show("cartoon", f"cx and chain {PARTNER_CHAIN}")
        cmd.color(C_PARTNER, f"cx and chain {PARTNER_CHAIN}")
        render("mol_partner_bound")

        # ---- Layer: partner rigid-body displaced along the interface normal ------------
        # A real dissociation: same coordinates, translated away from the target centroid.
        target_c = _chain_centroid(cmd, "cx", TARGET_CHAIN)
        axis = partner_c - target_c
        axis /= np.linalg.norm(axis)
        cmd.create("partner_away", f"cx and chain {PARTNER_CHAIN}")
        cmd.translate((axis * 13.0).tolist(), object="partner_away", camera=0)
        cmd.hide("everything")
        cmd.show("cartoon", "partner_away")
        cmd.color(C_PARTNER_PALE, "partner_away")
        cmd.set("cartoon_transparency", 0.35, "partner_away")
        render("mol_partner_away")

        # ---- Layer: denatured-state ensemble (statistical coil model) ------------------
        # Case 2's monomer is core-destabilised: it never reaches the surface. There is no
        # experimental structure of that state, so this is an explicit MODEL -- coils
        # grown from the real chain A sequence (see the block comment above). Rendered as
        # tubes, never a surface, so it reads as distinct from the crystallographic layers.
        cmd.delete("partner_away")
        cmd.hide("everything")
        cmd.set("cartoon_transparency", 0.0)

        ca_model = cmd.get_model(f"cx and chain {TARGET_CHAIN} and name CA")
        resnames = [a.resn for a in ca_model.atom]
        native_ca = np.array([a.coord for a in ca_model.atom], dtype=float)
        native_diag = _bbox_diagonal(native_ca)
        rg_native = float(np.sqrt(((native_ca - native_ca.mean(0)) ** 2).sum(1).mean()))

        rng = np.random.default_rng(COIL_SEED)
        raw_confs = [_build_coil(resnames, rng) for _ in range(COIL_N_CONF)]
        rg_coil = float(np.mean([
            np.sqrt(((c[:, 1] - c[:, 1].mean(0)) ** 2).sum(1).mean()) for c in raw_confs]))

        # Scale each conformer so its bounding box matches the native monomer's. A true
        # denatured chain is ~2x wider; at panel scale that is an illegible smear and it
        # would also inflate the shared union crop, shrinking every other molecule. The
        # real expansion factor is recorded in the metadata and stated in the caption.
        fitted = []
        for c in raw_confs:
            ca = c[:, 1]
            s = native_diag / _bbox_diagonal(ca)
            fitted.append((c - ca.mean(0)) * s + native_ca.mean(0))
        coil_pdb = ASSET_DIR / "coil_ensemble.pdb"
        _write_coil_pdb(coil_pdb, fitted, resnames)

        cmd.load(str(coil_pdb), "coil")
        cmd.split_states("coil")
        cmd.delete("coil")
        coil_objs = sorted(o for o in cmd.get_object_list() if o.startswith("coil_"))
        for i, obj in enumerate(coil_objs):
            cmd.show("cartoon", obj)
            cmd.cartoon("tube", obj)
            cmd.set("cartoon_trace_atoms", 1, obj)
            cmd.set("cartoon_sampling", 14, obj)
            if i == 0:                       # representative conformer
                cmd.set("cartoon_tube_radius", 0.58, obj)
                cmd.color(C_COIL, obj)
            else:                            # faint ensemble members behind it
                cmd.set("cartoon_tube_radius", 0.30, obj)
                cmd.color(C_COIL_FAINT, obj)
                cmd.set("cartoon_transparency", 0.66, obj)
        cmd.set("ray_trace_gain", 0.05)
        render("mol_coil_ensemble")
        meta["coil_model"] = {
            "n_conformers": COIL_N_CONF, "seed": COIL_SEED,
            "rg_native_A": round(rg_native, 1), "rg_coil_A": round(rg_coil, 1),
            "expansion_factor": round(rg_coil / rg_native, 2),
            "note": "statistical-coil model of the denatured state; drawn scaled to the "
                    "native bounding box for legibility, not at its true expanded size",
        }

    (ASSET_DIR / "structure_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[wrote] {ASSET_DIR / 'structure_meta.json'}")
    print(f"  pdb={PDB_ID}  mutation_site={meta['mutation_site']}  "
          f"n_interface={len(meta['interface_residues'])}")


if __name__ == "__main__":
    main()
