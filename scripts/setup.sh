#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pnpm install --frozen-lockfile
npm --prefix apps/website ci --no-fund
python_executable="${PYTHON:-python3.11}"
"$python_executable" -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check 'pip==25.1.1' 'setuptools==80.9.0' 'wheel==0.45.1'
if [ "$(uname -s)" = Linux ]; then
  .venv/bin/python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r packages/training/requirements.lock
.venv/bin/python -m pip install -e packages/training --no-deps --no-build-isolation
