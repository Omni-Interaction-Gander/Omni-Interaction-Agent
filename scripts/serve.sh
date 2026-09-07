#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config="$root/gander_runtime/configs/serve.local.yaml"
if (( $# > 0 )) && [[ $1 != --* ]]; then
  config=$(realpath "$1")
  shift
elif [[ ! -f $config ]]; then
  printf '%s\n' \
    "Missing gander_runtime/configs/serve.local.yaml." \
    "Create it with:" \
    "  cp gander_runtime/configs/serve.example.yaml gander_runtime/configs/serve.local.yaml" >&2
  exit 2
fi

export PYTHONPATH="$root/minicpm_ft:$root/gander_runtime${PYTHONPATH:+:$PYTHONPATH}"
cd "$root/gander_runtime"
exec "${GANDER_PYTHON:-python}" -m gander_runtime.cli --config "$config" "$@"
