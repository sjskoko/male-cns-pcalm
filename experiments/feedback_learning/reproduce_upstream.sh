#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
checkout=${1:?Pass a NEW directory for the pinned official checkout}
git clone https://github.com/SakanaAI/pc-alm.git "$checkout"
cd "$checkout"
git checkout --detach 660747f61a8a7e547c0ecd2c48c8883380a7d1f6
uv sync --frozen --extra test
uv run --no-sync pytest -q
mkdir -p data/FashionMNIST/raw
for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl --fail --location "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/${name}.gz" -o "data/FashionMNIST/raw/${name}.gz"
done
out="$repo_root/experiments/feedback_learning/upstream_reproduction_new"
if [ -e "$out" ]; then
  echo "Refusing to overwrite $out" >&2
  exit 1
fi
uv run --no-sync python scripts/run_headline_grid.py --config configs/headline_fashion.yaml --widths 32 --depths 32 --activations relu --seeds 0 --methods bp,pc,pcalm --budget-rule 2L --state-lr-table configs/eta_best_by_cell.csv --output-dir "$out" --data-dir data
