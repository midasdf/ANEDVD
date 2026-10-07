#!/bin/sh
# Runs tools/real_model_check.zig against a real HuggingFace model directory.
#
#   tools/real_model_check.sh [model_dir]
#
# Default model_dir: /tmp/hf_real/tiny-random-LlamaForCausalLM
#
# Zig file imports may not escape the root module's directory, so the harness and
# the two modules it exercises are copied into one temp dir and run from there.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-/tmp/hf_real/tiny-random-LlamaForCausalLM}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if [ ! -f "$dir/model.safetensors" ] && [ ! -f "$dir/model.safetensors.index.json" ]; then
  echo "no model.safetensors / model.safetensors.index.json in $dir" >&2
  exit 1
fi

cp "$root/src/safetensors.zig" "$root/src/hf.zig" "$work/"
sed -e 's|@import("hf")|@import("hf.zig")|' \
    -e 's|@import("safetensors")|@import("safetensors.zig")|' \
    -e "s|^const model_dir = .*|const model_dir = \"$dir\";|" \
    "$root/tools/real_model_check.zig" > "$work/main.zig"

zig run "$work/main.zig"
