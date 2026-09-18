#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
dest=data/raw/malecns-v1.0
mkdir -p "$dest"
base=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for name in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  curl --fail --location --retry 3 --continue-at - "$base/$name" --output "$dest/$name"
done
# study.py prepare verifies all three SHA-256 digests before any extraction.
