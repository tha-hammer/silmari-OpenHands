#!/usr/bin/env bash
set -euo pipefail

# checkpoint_manager.sh - Comprehensive checkpoint management for implementation plans
#
# Usage:
#   ./hack/checkpoint_manager.sh create <name> <description>
#   ./hack/checkpoint_manager.sh list [--recent N]
#   ./hack/checkpoint_manager.sh restore <checkpoint_name>
#   ./hack/checkpoint_manager.sh cleanup [--keep-recent N]
#   ./hack/checkpoint_manager.sh status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CHECKPOINT_DIR="$HOME/.silmaricheckpoints"
REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")
CURRENT_BRANCH=$(git branch --show-current)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Ensure checkpoint directory exists
mkdir -p "$CHECKPOINT_DIR/$REPO_NAME"

# Function to get checkpoint path
get_checkpoint_path() {
    local checkpoint_name="$1"
    echo "$CHECKPOINT_DIR/$REPO_NAME/${checkpoint_name}_${TIMESTAMP}"
}

# Function to create a checkpoint
create_checkpoint() {
    local checkpoint_name="$1"
    local description="$2"

    if [ -z "$checkpoint_name" ] || [ -z "$description" ]; then
        echo -e "${RED}Error: Both checkpoint name and description are required${NC}"
        echo "Usage: $0 create <name> <description>"
        exit 1
    fi

    # Validate checkpoint name (alphanumeric, underscores, hyphens only)
    if [[ ! "$checkpoint_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo -e "${RED}Error: Checkpoint name must contain only alphanumeric characters, underscores, and hyphens${NC}"
        exit 1
    fi

    local checkpoint_path="$CHECKPOINT_DIR/$REPO_NAME/${checkpoint_name}_${TIMESTAMP}"

    echo -e "${BLUE}Creating checkpoint: $checkpoint_name${NC}"
    echo -e "${YELLOW}Description: $description${NC}"
    echo -e "${YELLOW}Location: $checkpoint_path${NC}"

    # Create checkpoint directory
    mkdir -p "$checkpoint_path"

    # Save git status and diff
    echo "Saving git status..."
    git status --porcelain > "$checkpoint_path/git_status.txt"
    git diff > "$checkpoint_path/git_diff.txt" 2>/dev/null || touch "$checkpoint_path/git_diff.txt"
    git diff --cached > "$checkpoint_path/git_diff_cached.txt" 2>/dev/null || touch "$checkpoint_path/git_diff_cached.txt"

    # Save current branch and commit
    echo "$CURRENT_BRANCH" > "$checkpoint_path/branch.txt"
    git rev-parse HEAD > "$checkpoint_path/commit.txt"

    # Save plan progress if plan file exists
    if [ -f "thoughts/shared/plans" ] && find thoughts/shared/plans -name "*.md" -type f | head -1 | grep -q .; then
        echo "Saving plan progress..."
        find thoughts/shared/plans -name "*.md" -type f -exec cp {} "$checkpoint_path/" \;
    fi

    # Save current working directory structure (key files only)
    echo "Saving key files..."
    mkdir -p "$checkpoint_path/files"

    # Save modified files
    git status --porcelain | awk '{print $2}' | while read -r file; do
        if [ -f "$file" ]; then
            local dir_path=$(dirname "$file")
            mkdir -p "$checkpoint_path/files/$dir_path"
            cp "$file" "$checkpoint_path/files/$file"
        fi
    done

    # Create checkpoint metadata
    cat > "$checkpoint_path/metadata.json" << EOF
{
    "name": "$checkpoint_name",
    "description": "$description",
    "timestamp": "$TIMESTAMP",
    "branch": "$CURRENT_BRANCH",
    "commit": "$(git rev-parse HEAD)",
    "repo": "$REPO_NAME",
    "created_by": "$(whoami)",
    "hostname": "$(hostname)"
}
EOF

    echo -e "${GREEN}✓ Checkpoint created successfully${NC}"
    echo -e "${BLUE}Checkpoint ID: ${checkpoint_name}_${TIMESTAMP}${NC}"
}

# Function to list checkpoints
list_checkpoints() {
    local keep_recent="${1:-10}"

    echo -e "${BLUE}Available checkpoints for $REPO_NAME:${NC}"
    echo ""

    if [ ! -d "$CHECKPOINT_DIR/$REPO_NAME" ]; then
        echo -e "${YELLOW}No checkpoints found${NC}"
        return 0
    fi

    # List checkpoints sorted by timestamp (newest first)
    find "$CHECKPOINT_DIR/$REPO_NAME" -name "metadata.json" -type f | \
        sort -r | \
        head -n "$keep_recent" | \
        while read -r metadata_file; do
            local checkpoint_dir=$(dirname "$metadata_file")
            local checkpoint_id=$(basename "$checkpoint_dir")

            # Extract info from metadata
            local name=$(jq -r '.name' "$metadata_file" 2>/dev/null || echo "unknown")
            local description=$(jq -r '.description' "$metadata_file" 2>/dev/null || echo "no description")
            local timestamp=$(jq -r '.timestamp' "$metadata_file" 2>/dev/null || echo "unknown")
            local branch=$(jq -r '.branch' "$metadata_file" 2>/dev/null || echo "unknown")

            # Format timestamp for display
            local display_time=$(echo "$timestamp" | sed 's/_/ /' | sed 's/\([0-9]\{8\}\) \([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1 \2:\3:\4/')

            echo -e "${GREEN}$checkpoint_id${NC}"
            echo -e "  ${YELLOW}Name:${NC} $name"
            echo -e "  ${YELLOW}Description:${NC} $description"
            echo -e "  ${YELLOW}Created:${NC} $display_time"
            echo -e "  ${YELLOW}Branch:${NC} $branch"
            echo ""
        done

    local total_count=$(find "$CHECKPOINT_DIR/$REPO_NAME" -name "metadata.json" -type f | wc -l)
    if [ "$total_count" -gt "$keep_recent" ]; then
        echo -e "${YELLOW}Showing $keep_recent most recent checkpoints (total: $total_count)${NC}"
        echo -e "${YELLOW}Use --recent N to see more${NC}"
    fi
}

# Function to restore a checkpoint
restore_checkpoint() {
    local checkpoint_name="$1"

    if [ -z "$checkpoint_name" ]; then
        echo -e "${RED}Error: Checkpoint name is required${NC}"
        echo "Usage: $0 restore <checkpoint_name>"
        exit 1
    fi

    # Find the checkpoint directory
    local checkpoint_dir=$(find "$CHECKPOINT_DIR/$REPO_NAME" -name "${checkpoint_name}_*" -type d | sort -r | head -1)

    if [ -z "$checkpoint_dir" ]; then
        echo -e "${RED}Error: Checkpoint '$checkpoint_name' not found${NC}"
        echo ""
        echo "Available checkpoints:"
        list_checkpoints 5
        exit 1
    fi

    echo -e "${BLUE}Restoring checkpoint: $checkpoint_name${NC}"
    echo -e "${YELLOW}Location: $checkpoint_dir${NC}"

    # Read metadata
    local metadata_file="$checkpoint_dir/metadata.json"
    if [ -f "$metadata_file" ]; then
        local description=$(jq -r '.description' "$metadata_file")
        local branch=$(jq -r '.branch' "$metadata_file")
        local commit=$(jq -r '.commit' "$metadata_file")

        echo -e "${YELLOW}Description: $description${NC}"
        echo -e "${YELLOW}Branch: $branch${NC}"
        echo -e "${YELLOW}Commit: $commit${NC}"
    fi

    # Confirm restoration
    echo ""
    read -p "This will restore your working directory to this checkpoint. Continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Restoration cancelled"
        exit 0
    fi

    # Check if we're on the right branch
    if [ "$CURRENT_BRANCH" != "$branch" ]; then
        echo -e "${YELLOW}Switching to branch: $branch${NC}"
        git checkout "$branch" || {
            echo -e "${RED}Error: Could not switch to branch $branch${NC}"
            exit 1
        }
    fi

    # Reset to the checkpoint commit
    echo "Resetting to checkpoint commit..."
    git reset --hard "$commit" || {
        echo -e "${RED}Error: Could not reset to commit $commit${NC}"
        exit 1
    }

    # Restore files if they exist
    if [ -d "$checkpoint_dir/files" ]; then
        echo "Restoring modified files..."
        cp -r "$checkpoint_dir/files/"* . 2>/dev/null || true
    fi

    # Restore plan files if they exist
    if [ -d "$checkpoint_dir" ] && find "$checkpoint_dir" -name "*.md" -type f | head -1 | grep -q .; then
        echo "Restoring plan files..."
        find "$checkpoint_dir" -name "*.md" -type f -exec cp {} thoughts/shared/plans/ \; 2>/dev/null || true
    fi

    echo -e "${GREEN}✓ Checkpoint restored successfully${NC}"
}

# Function to cleanup old checkpoints
cleanup_checkpoints() {
    local keep_recent="${1:-5}"

    echo -e "${BLUE}Cleaning up old checkpoints (keeping $keep_recent most recent)${NC}"

    if [ ! -d "$CHECKPOINT_DIR/$REPO_NAME" ]; then
        echo -e "${YELLOW}No checkpoints to clean up${NC}"
        return 0
    fi

    # Find all checkpoint directories, sort by timestamp (newest first)
    local checkpoints_to_delete=$(find "$CHECKPOINT_DIR/$REPO_NAME" -name "*_*" -type d | sort -r | tail -n +$((keep_recent + 1)))

    if [ -z "$checkpoints_to_delete" ]; then
        echo -e "${YELLOW}No old checkpoints to clean up${NC}"
        return 0
    fi

    echo "Checkpoints to be deleted:"
    echo "$checkpoints_to_delete" | while read -r checkpoint_dir; do
        local checkpoint_id=$(basename "$checkpoint_dir")
        echo -e "  ${RED}$checkpoint_id${NC}"
    done

    echo ""
    read -p "Delete these checkpoints? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$checkpoints_to_delete" | while read -r checkpoint_dir; do
            echo "Deleting $checkpoint_dir..."
            rm -rf "$checkpoint_dir"
        done
        echo -e "${GREEN}✓ Cleanup complete${NC}"
    else
        echo "Cleanup cancelled"
    fi
}

# Function to show checkpoint status
show_status() {
    echo -e "${BLUE}Checkpoint Status for $REPO_NAME${NC}"
    echo ""

    # Current git status
    echo -e "${YELLOW}Current Git Status:${NC}"
    git status --short
    echo ""

    # Recent checkpoints
    echo -e "${YELLOW}Recent Checkpoints:${NC}"
    list_checkpoints 3
}

# Main command handling
case "${1:-}" in
    "create")
        create_checkpoint "${2:-}" "${3:-}"
        ;;
    "list")
        list_checkpoints "${2:-10}"
        ;;
    "restore")
        restore_checkpoint "${2:-}"
        ;;
    "cleanup")
        cleanup_checkpoints "${2:-5}"
        ;;
    "status")
        show_status
        ;;
    *)
        echo -e "${BLUE}Checkpoint Manager for $REPO_NAME${NC}"
        echo ""
        echo "Usage:"
        echo "  $0 create <name> <description>     Create a new checkpoint"
        echo "  $0 list [--recent N]               List recent checkpoints (default: 10)"
        echo "  $0 restore <checkpoint_name>       Restore to a checkpoint"
        echo "  $0 cleanup [--keep-recent N]       Clean up old checkpoints (default: keep 5)"
        echo "  $0 status                          Show current status and recent checkpoints"
        echo ""
        echo "Examples:"
        echo "  $0 create phase_1_start \"Starting Phase 1: Initial setup\""
        echo "  $0 list --recent 5"
        echo "  $0 restore phase_1_start_20240101_120000"
        echo "  $0 cleanup --keep-recent 3"
        ;;
esac
