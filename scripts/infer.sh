#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config="$root/minicpm_ft/configs/infer.yaml"
if (( $# > 0 )) && [[ $1 != --* ]]; then
  config=$(realpath "$1")
  shift
fi

cd "$root/minicpm_ft"
exec "${MCPMFT_PYTHON:-python}" -m mcpmft.infer.cli --config "$config" "$@"
