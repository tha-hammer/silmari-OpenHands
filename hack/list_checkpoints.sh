#!/usr/bin/env bash
# list_checkpoints.sh - Convenience wrapper for listing checkpoints
# Usage: ./hack/list_checkpoints.sh [--recent N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/checkpoint_manager.sh" list "$@"
