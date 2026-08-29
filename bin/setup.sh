#!/usr/bin/env bash
#
# Build the two virtualenvs this project needs.
#
# Why two? `pip install esm` hard-requires torch 2.11.0+cu130 and
# transformers 4.57.x. The verified pipeline -- every coverage count, ESM2
# masked log-prob and permutation p-value already committed under results/ --
# was measured against torch 2.5.1+cu121 and transformers 5.16.1. Installing
# esm into that environment would silently swap both out and invalidate the
# numbers. So:
#
#   .venv       core: ProteinGym + UniProt + ESM2 + stats, pinned, load-bearing
#   .venv-esmc  ESMC only, ~5.8 GB, disposable, skippable with --core-only
#
# The split is deliberate. Do not "simplify" it into one env.
#
# This script is idempotent: existing venvs are reused and their requirements
# re-applied, so re-running after a pin change is the supported upgrade path.
#
# It downloads NO datasets and NO model weights. ProteinGym / MaveDB /
# MegaScale / PDB acquisition lives in scripts/fetch_data.py; model weights are
# pulled lazily on first load by plmconfound.models.load_model.
#
# Usage:
#   bin/setup.sh                # both envs
#   bin/setup.sh --core-only    # skip the 5.8 GB ESMC env
#   bin/setup.sh --esmc-only    # ESMC env only
#
# Intended to be executable (chmod +x bin/setup.sh); `bash bin/setup.sh` works
# either way.

set -euo pipefail

# Resolve the repo root from this script's own location so the script works
# from any cwd (CI, editor task runners, a shell parked in scripts/).
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"

CORE_VENV="$REPO_ROOT/.venv"
ESMC_VENV="$REPO_ROOT/.venv-esmc"
PY_VERSION="3.12"

# torch 2.5.1+cu121 is not published to PyPI proper. Without this index the
# resolver either fails outright or quietly installs a CPU-only build, which
# turns every scoring run into a silent 100x slowdown.
CU121_INDEX="https://download.pytorch.org/whl/cu121"
# Kept in sync with requirements-core.txt. Installed on its own from the index above so
# the cu121 mirror never gets a chance to win an unrelated package.
TORCH_PIN="torch==2.5.1+cu121"

do_core=1
do_esmc=1

for arg in "$@"; do
    case "$arg" in
        --core-only) do_esmc=0 ;;
        --esmc-only) do_core=0 ;;
        -h|--help)
            sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "setup.sh: unknown flag '$arg' (expected --core-only or --esmc-only)" >&2
            exit 2
            ;;
    esac
done

# Prefer uv: it is on PATH here, and it is what created the current .venv.
# Consequence worth knowing -- a uv-created venv has no `pip` inside it, so
# `.venv/bin/pip freeze` will fail with "no such file"; inspect it with
#     uv pip freeze --python .venv/bin/python
# The stdlib-venv fallback below does install pip, so both shapes are handled
# by the helpers rather than by the caller.
if command -v uv >/dev/null 2>&1; then
    HAVE_UV=1
else
    HAVE_UV=0
fi

log() { printf '\n==> %s\n' "$*"; }

# Pick a concrete CPython for the non-uv path. uv resolves --python itself and
# will download an interpreter if the system lacks 3.12.
find_python() {
    local candidate
    for candidate in "python$PY_VERSION" python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 \
           && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
            command -v "$candidate"
            return 0
        fi
    done
    echo "setup.sh: no Python $PY_VERSION interpreter found and uv is unavailable" >&2
    return 1
}

# Idempotent: an existing, working venv is reused rather than rebuilt, so a
# re-run only re-resolves requirements instead of re-downloading torch.
make_venv() {
    local venv_dir="$1"
    if [[ -x "$venv_dir/bin/python" ]]; then
        log "reusing existing venv $venv_dir"
        return 0
    fi
    log "creating venv $venv_dir (Python $PY_VERSION)"
    if [[ "$HAVE_UV" -eq 1 ]]; then
        uv venv --python "$PY_VERSION" "$venv_dir"
    else
        "$(find_python)" -m venv "$venv_dir"
        "$venv_dir/bin/python" -m pip install --upgrade pip
    fi
}

venv_pip_install() {
    local venv_dir="$1"
    shift
    if [[ "$HAVE_UV" -eq 1 ]]; then
        uv pip install --python "$venv_dir/bin/python" "$@"
    else
        "$venv_dir/bin/python" -m pip install "$@"
    fi
}

# One line per env: interpreter version, torch version, and whether CUDA is
# actually visible. The last one is the check that catches a CPU-only wheel
# sneaking in, which is otherwise invisible until a scoring run takes hours.
report_env() {
    local label="$1" venv_dir="$2"
    if [[ ! -x "$venv_dir/bin/python" ]]; then
        printf '  %-10s (not installed)\n' "$label"
        return 0
    fi
    "$venv_dir/bin/python" - "$label" <<'PY'
import sys

label = sys.argv[1]
py = "%d.%d.%d" % sys.version_info[:3]
try:
    import torch

    detail = "torch %s, cuda.is_available()=%s" % (torch.__version__, torch.cuda.is_available())
except Exception as exc:  # torch missing or broken install
    detail = "torch unavailable (%s)" % type(exc).__name__
print("  %-10s python %s | %s" % (label, py, detail))
PY
}

if [[ "$do_core" -eq 1 ]]; then
    make_venv "$CORE_VENV"
    # torch first, and ONLY torch, from the cu121 index. uv resolves each package
    # against the first index that carries it, so exposing the torch index to the whole
    # requirements file makes it win packages it merely happens to mirror -- `requests`
    # is present there at a version we do not pin, which hard-fails resolution. Keeping
    # the extra index scoped to the one package that needs it avoids reaching for
    # --index-strategy unsafe-best-match, which would weaken every other resolution.
    log "installing torch (cu121 wheel) into $CORE_VENV"
    venv_pip_install "$CORE_VENV" --extra-index-url "$CU121_INDEX" "$TORCH_PIN"
    log "installing remaining core requirements into $CORE_VENV"
    venv_pip_install "$CORE_VENV" -r "$REPO_ROOT/requirements-core.txt"
    log "installing plmconfound (editable) into $CORE_VENV"
    # Editable so scripts/ and tests/ import the working tree, not a stale copy.
    venv_pip_install "$CORE_VENV" -e "$REPO_ROOT"
fi

if [[ "$do_esmc" -eq 1 ]]; then
    make_venv "$ESMC_VENV"
    log "installing ESMC requirements into $ESMC_VENV (~5.8 GB, pulls its own torch)"
    venv_pip_install "$ESMC_VENV" -r "$REPO_ROOT/requirements-esmc.txt"
fi

log "verification"
report_env ".venv" "$CORE_VENV"
report_env ".venv-esmc" "$ESMC_VENV"
printf '\nNo data or model weights were downloaded. Next: .venv/bin/python scripts/fetch_data.py\n'
