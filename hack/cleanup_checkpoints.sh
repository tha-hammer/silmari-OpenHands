#!/usr/bin/env bash
# cleanup_checkpoints.sh - Convenience wrapper for cleaning up checkpoints
# Usage: ./hack/cleanup_checkpoints.sh [--keep-recent N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/checkpoint_manager.sh" cleanup "$@"
