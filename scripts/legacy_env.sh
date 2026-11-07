#!/usr/bin/env bash
# Rebuild the legacy engine environment (vectorbt 0.28.5 / pandas 2.3.3 / numpy 2.2.6)
# from git tag legacy-engine-vbt0285 plus requirements-legacy.lock.
set -euo pipefail

TAG="legacy-engine-vbt0285"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

usage() {
  cat <<USAGE
Usage: scripts/legacy_env.sh [--help] [TARGET_DIR]

Recreate the legacy ggTrader engine in TARGET_DIR (default: ./legacy-engine):
  TARGET_DIR/src/       code extracted with 'git archive ${TAG}' (no git worktree needed)
  TARGET_DIR/venv/      virtualenv built from requirements-legacy.lock (exact pins)

Environment:
  PYTHON_BIN   interpreter used to create the venv (default: python3.12)
  REPO_DIR     repository holding the tag and the lock (default: this script's repo)

Afterwards run legacy code with:
  PYTHONPATH=TARGET_DIR/src/src TARGET_DIR/venv/bin/python -c 'import ggTrader.lab.simulate'
(ggTrader itself is not pip-installed; the extracted checkout is put on PYTHONPATH.)
The script never touches a database, the network beyond pip, or the production venv.
USAGE
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TARGET="${1:-./legacy-engine}"
LOCK="${REPO_DIR}/requirements-legacy.lock"

[ -f "$LOCK" ] || { echo "missing $LOCK" >&2; exit 1; }
git -C "$REPO_DIR" rev-parse -q --verify "refs/tags/${TAG}" >/dev/null \
  || { echo "tag ${TAG} not found in ${REPO_DIR}" >&2; exit 1; }
[ ! -e "$TARGET/venv" ] || { echo "$TARGET/venv already exists; remove it first" >&2; exit 1; }

mkdir -p "$TARGET/src"
git -C "$REPO_DIR" archive "$TAG" | tar -x -C "$TARGET/src"
"$PYTHON_BIN" -m venv "$TARGET/venv"
"$TARGET/venv/bin/python" -m pip install --quiet --upgrade pip
"$TARGET/venv/bin/python" -m pip install --quiet -r "$LOCK"

# Import check: the tagged simulator must load against the pinned stack.
PYTHONPATH="$TARGET/src/src" "$TARGET/venv/bin/python" -c \
  'import vectorbt, pandas, numpy, ggTrader.lab.simulate as s; print("legacy env ok:", vectorbt.__version__, pandas.__version__, numpy.__version__, s.__name__)'
