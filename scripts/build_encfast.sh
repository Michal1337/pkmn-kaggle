#!/bin/bash
# Build rl/_encfast*.so on a node that can import numpy (e.g. an srun compute node).
# Cython is installed to a THROWAWAY dir (pip --target) and used only at build time, so the
# training venv is left untouched; the resulting .so imports with no Cython at runtime.
#
#   VENV=/path/to/venv-with-numpy  bash scripts/build_encfast.sh  [repo_dir]
#
set -e
REPO="${1:-$PWD}"
: "${VENV:?set VENV=/path/to/venv (with numpy) -- e.g. the training venv}"
CYLIB="${CYLIB:-$HOME/.cy_build}"
. "$VENV/bin/activate"
pip install --target="$CYLIB" --quiet --disable-pip-version-check "Cython>=3" setuptools 2>&1 | tail -1
cd "$REPO"
PYTHONPATH="$CYLIB" python setup_encfast.py build_ext --inplace
python -c "from rl import _encfast; print('_encfast built + imports OK:', _encfast.__file__)"
