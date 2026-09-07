#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 {check|s3} RELEASE_TRAIN_CONFIG [options]" >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
command=$1
release_config=$(realpath "$2")
shift 2

cd "$root/minicpm_ft"
python=${MCPMFT_PYTHON:-python}
exec "$python" -m mcpmft.data_cli "$command" \
  --config "$root/minicpm_ft/configs/train.yaml" \
  --config "$release_config" \
  "$@"
