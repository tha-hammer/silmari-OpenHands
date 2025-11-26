#!/usr/bin/env bash
# restore_checkpoint.sh - Convenience wrapper for restoring checkpoints
# Usage: ./hack/restore_checkpoint.sh <checkpoint_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/checkpoint_manager.sh" restore "$@"
