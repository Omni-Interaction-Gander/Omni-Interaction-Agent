#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
configs=("$root/minicpm_ft/configs/train.yaml")
while (( $# > 0 )) && [[ $1 != --* ]]; do
  configs+=("$(realpath "$1")")
  shift
done

cd "$root/minicpm_ft"
python=${MCPMFT_PYTHON:-python}
python_override=()
if [[ -n ${MCPMFT_PYTHON:-} ]]; then
  python_override=(--launch.python "$MCPMFT_PYTHON")
fi
config_args=()
for config in "${configs[@]}"; do
  config_args+=(--config "$config")
done
exec "$python" -m mcpmft.launch "${config_args[@]}" "${python_override[@]}" "$@"
