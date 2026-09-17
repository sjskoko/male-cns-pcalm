#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

raw_dir="data/raw/malecns-v1.0"
experiment_dir="experiments/malecns_v1_real"
processed_graph="data/processed/malecns-v1-visual-descending.pt"
base_url="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
mkdir -p "$raw_dir" "$experiment_dir"

download_and_verify() {
  local filename="$1"
  local checksum="$2"
  if [[ -f "$raw_dir/$filename" ]] && \
    printf '%s  %s\n' "$checksum" "$raw_dir/$filename" | sha256sum --check --status; then
    printf 'Verified existing %s\n' "$raw_dir/$filename"
    return
  fi
  curl --fail --location --retry 5 --continue-at - \
    "$base_url/$filename" --output "$raw_dir/$filename"
  printf '%s  %s\n' "$checksum" "$raw_dir/$filename" | sha256sum --check -
}

download_and_verify \
  "body-annotations-male-cns-v1.0-minconf-0.5.feather" \
  "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"
download_and_verify \
  "body-neurotransmitters-male-cns-v1.0.feather" \
  "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"
download_and_verify \
  "connectome-weights-male-cns-v1.0-minconf-0.5.feather" \
  "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"

uv sync --extra dev

uv run flypcalm select-malecns \
  --annotations "$raw_dir/body-annotations-male-cns-v1.0-minconf-0.5.feather" \
  --neurotransmitters "$raw_dir/body-neurotransmitters-male-cns-v1.0.feather" \
  --weights "$raw_dir/connectome-weights-male-cns-v1.0-minconf-0.5.feather" \
  --output "$experiment_dir/assignments.csv" \
  --report "$experiment_dir/selection-report.json" \
  --layer-sizes 32 48 40 24 \
  --min-weight 5 \
  --nt-confidence 0.5

uv run flypcalm prepare \
  --edges "$raw_dir/connectome-weights-male-cns-v1.0-minconf-0.5.feather" \
  --assignments "$experiment_dir/assignments.csv" \
  --min-weight 5 \
  --output "$processed_graph" \
  --report "$experiment_dir/projection-report.json"

uv run flypcalm benchmark \
  --config configs/benchmarks/malecns_visual_motor.yaml \
  --output-dir "$experiment_dir/results"
