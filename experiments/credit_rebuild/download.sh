#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
dest=data/raw/malecns-v1.0
mkdir -p "$dest"
base=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for name in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  if python -c 'import sys; from pathlib import Path; sys.path.insert(0,"experiments/credit_rebuild"); from study import digest,HASHES; p=Path(sys.argv[1]); sys.exit(0 if p.exists() and digest(p)==HASHES[sys.argv[2]] else 1)' "$dest/$name" "$name"; then
    continue
  fi
  curl --fail --location --retry 3 --continue-at - "$base/$name" --output "$dest/$name"
done
# study.py prepare verifies all three SHA-256 digests before any extraction.
