#!/bin/bash
# Unified pipeline runner with profile support

PROFILE=${1:-eval}
MODEL=${2:-gemma3}
CONFIG_PATH="config/profiles/${PROFILE}/svd_${MODEL}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    echo "Usage: $0 [eval|profile] [gemma3|qwen3]"
    exit 1
fi

echo "Running with profile: $PROFILE, model: $MODEL"
echo "Config: $CONFIG_PATH"

python scripts/examples/cpu/svd_${MODEL}.py --config $CONFIG_PATH
