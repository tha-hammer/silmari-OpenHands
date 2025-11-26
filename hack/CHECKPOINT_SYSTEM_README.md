# Checkpoint Management System

This directory contains a comprehensive checkpoint management system designed to work with implementation plans and git workflows. The system provides recovery points, progress tracking, and seamless integration with your existing worktree management.

## Overview

The checkpoint system allows you to:
- Create recovery points during implementation work
- Track progress through complex multi-phase plans
- Restore to any previous state quickly
- Integrate with git commits and worktree management
- Maintain clean recovery points for debugging and iteration

## Files

### Core Scripts
- `checkpoint_manager.sh` - Main checkpoint management script
- `create_checkpoint.sh` - Convenience wrapper for creating checkpoints
- `list_checkpoints.sh` - Convenience wrapper for listing checkpoints
- `restore_checkpoint.sh` - Convenience wrapper for restoring checkpoints
- `cleanup_checkpoints.sh` - Convenience wrapper for cleaning up checkpoints

### Integration Files
- `../.claude/commands/implement_plan_with_checkpoints.md` - Enhanced implementation plan with checkpoint integration
- `../.claude/commands/commit_with_checkpoints.md` - Enhanced commit process with checkpoint integration

## Usage

### Basic Checkpoint Operations

```bash
# Create a checkpoint
./hack/create_checkpoint.sh "phase_1_start" "Starting Phase 1: Initial setup"

# List recent checkpoints
./hack/list_checkpoints.sh --recent 5

# Restore to a checkpoint
./hack/restore_checkpoint.sh "phase_1_start_20240101_120000"

# Clean up old checkpoints
./hack/cleanup_checkpoints.sh --keep-recent 3
```

### Integration with Implementation Plans

The checkpoint system integrates seamlessly with the enhanced implementation plan workflow:

1. **Pre-phase checkpoint**: Create before starting each major phase
2. **Post-phase checkpoint**: Create after completing verification
3. **Recovery checkpoint**: Create when encountering issues
4. **Commit checkpoint**: Create before committing changes

### Integration with Git Commits

The checkpoint system provides safety nets for git operations:

1. **Pre-commit checkpoint**: Create before any git changes
2. **Post-commit checkpoint**: Create after successful commits
3. **Recovery checkpoint**: Create before rollbacks or modifications

## Checkpoint Storage

Checkpoints are stored in `$HOME/.silmari/checkpoints/$REPO_NAME/` with the following structure:

```
checkpoints/
└── repo-name/
    ├── checkpoint_name_timestamp/
    │   ├── metadata.json          # Checkpoint metadata
    │   ├── git_status.txt         # Git status at checkpoint
    │   ├── git_diff.txt           # Unstaged changes
    │   ├── git_diff_cached.txt    # Staged changes
    │   ├── branch.txt             # Current branch
    │   ├── commit.txt             # Current commit hash
    │   ├── files/                 # Modified files
    │   └── *.md                   # Plan files (if any)
    └── ...
```

## Naming Conventions

### Checkpoint Names
- `phase_N_description` - Phase boundaries
- `recovery_issue_description` - Recovery points
- `commit_feature_name` - Commit boundaries
- `discovery_finding_name` - Significant discoveries

### Examples
- `phase_1_initial_setup`
- `recovery_database_connection_issue`
- `commit_user_authentication`
- `discovery_api_rate_limits`

## Workflow Integration

### Implementation Plan Workflow
```bash
# Start of phase
./hack/create_checkpoint.sh "phase_1_start" "Starting Phase 1"

# During implementation (if issues arise)
./hack/create_checkpoint.sh "recovery_issue_found" "Found issue with API integration"

# End of phase
./hack/create_checkpoint.sh "phase_1_complete" "Completed Phase 1"

# Before commit
./hack/create_checkpoint.sh "commit_phase_1" "Ready to commit Phase 1 changes"
```

### Git Commit Workflow
```bash
# Pre-commit safety
./hack/create_checkpoint.sh "commit_ready" "Ready to commit changes"

# Execute commits
git add <files>
git commit -m "Add feature"

# Post-commit confirmation
./hack/create_checkpoint.sh "commit_complete" "Successfully committed changes"
```

## Recovery Scenarios

### Implementation Recovery
If you encounter issues during implementation:
1. Create a recovery checkpoint
2. Analyze the issue
3. Either fix the issue or restore to a previous checkpoint
4. Continue from the restored state

### Commit Recovery
If commits need to be modified or rolled back:
1. Create a recovery checkpoint
2. Execute git operations (reset, rebase, etc.)
3. Create a post-recovery checkpoint
4. Continue with corrected commits

### Plan Recovery
If you need to resume work from a previous state:
1. List available checkpoints
2. Restore to the appropriate checkpoint
3. Verify the restored state
4. Continue implementation from that point

## Best Practices

### Checkpoint Frequency
- Create checkpoints at natural stopping points
- Don't create too many checkpoints (use cleanup regularly)
- Create checkpoints before risky operations
- Create checkpoints after successful completions

### Naming
- Use descriptive names that indicate the purpose
- Include phase numbers for multi-phase work
- Use consistent naming patterns
- Avoid overly long names

### Cleanup
- Regularly clean up old checkpoints
- Keep only the most recent checkpoints needed
- Use `--keep-recent N` to maintain a reasonable number
- Clean up after successful completion of work

## Integration with Existing Tools

### Worktree Integration
The checkpoint system works alongside your existing worktree management:
- Checkpoints are repository-specific
- Worktrees can have their own checkpoint sets
- Use checkpoints to manage worktree state transitions

### Plan Integration
Checkpoints integrate with your plan management:
- Plan files are automatically saved in checkpoints
- Plan progress is preserved across checkpoint restores
- Use checkpoints to track plan completion status

## Troubleshooting

### Common Issues
1. **Checkpoint not found**: Use `list_checkpoints.sh` to see available checkpoints
2. **Restore failed**: Ensure you're on the correct branch before restoring
3. **Permission issues**: Check that checkpoint directories are writable
4. **Disk space**: Use `cleanup_checkpoints.sh` to remove old checkpoints

### Recovery Commands
```bash
# List all checkpoints
./hack/list_checkpoints.sh

# Show current status
./hack/checkpoint_manager.sh status

# Force cleanup if needed
./hack/cleanup_checkpoints.sh --keep-recent 1
```

## Examples

### Complete Implementation Workflow
```bash
# Start implementation
./hack/create_checkpoint.sh "implementation_start" "Starting implementation work"

# Phase 1
./hack/create_checkpoint.sh "phase_1_start" "Starting Phase 1: Setup"
# ... implement phase 1 ...
./hack/create_checkpoint.sh "phase_1_complete" "Completed Phase 1"

# Phase 2
./hack/create_checkpoint.sh "phase_2_start" "Starting Phase 2: Core logic"
# ... implement phase 2 ...
./hack/create_checkpoint.sh "phase_2_complete" "Completed Phase 2"

# Commit
./hack/create_checkpoint.sh "commit_ready" "Ready to commit all changes"
git add <files>
git commit -m "Implement feature with phases 1 and 2"
./hack/create_checkpoint.sh "commit_complete" "Successfully committed changes"

# Cleanup
./hack/cleanup_checkpoints.sh --keep-recent 3
```

This checkpoint system provides a robust foundation for managing complex implementation work with proper recovery and progress tracking capabilities.
