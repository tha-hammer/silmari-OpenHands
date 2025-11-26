#!/usr/bin/env bash
# create_checkpoint.sh - Convenience wrapper for creating checkpoints
# Usage: ./hack/create_checkpoint.sh <name> <description>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/checkpoint_manager.sh" create "$@"
