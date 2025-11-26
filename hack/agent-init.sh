#!/usr/bin/env bash
set -euo pipefail

# agent-init.sh - Initialize a directory with .claude structure and project directories
# Usage: agent-init <directory-name>
# 
# This script creates a new directory with:
# - .claude/agents/ and .claude/commands/ directories with all files from the template
# - hack/ directory with all scripts from the template
# - frontend/, backend/, and src/ directories
# - Preserves capitalization while using kebab-case for directory names

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to convert to kebab-case while preserving capitalization
to_kebab_case() {
    local input="$1"
    # Convert to lowercase and replace spaces/underscores with hyphens
    echo "$input" | tr '[:upper:]' '[:lower:]' | sed 's/[[:space:]_]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g'
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}agent-init - Initialize a directory with .claude structure${NC}"
    echo ""
    echo "Usage:"
    echo "  agent-init <directory-name>"
    echo ""
    echo "Examples:"
    echo "  agent-init my-new-project"
    echo "  agent-init \"My New Project\""
    echo "  agent-init my_awesome_project"
    echo ""
    echo "This will create:"
    echo "  - A directory named <directory-name> (in kebab-case)"
    echo "  - .claude/agents/ and .claude/commands/ with all template files"
    echo "  - hack/ directory with all template scripts"
    echo "  - frontend/, backend/, and src/ directories"
}

# Function to get the template directory
get_template_dir() {
    # First, try to find the template directory relative to this script
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Look for .claude directory in parent directories
    local current_dir="$script_dir"
    while [ "$current_dir" != "/" ]; do
        if [ -d "$current_dir/.claude" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done
    
    # If not found relative to script, try common locations
    local common_locations=(
        "$HOME/.claude"
        "$HOME/Dev"
        "$HOME/NS-hubspot-sales-tools"
        "/opt/templates/NS-hubspot-sales-tools"
    )
    
    for location in "${common_locations[@]}"; do
        if [ -d "$location/.claude" ]; then
            echo "$location"
            return 0
        fi
    done
    
    # If still not found, prompt user
    echo -e "${RED}Error: Could not find template .claude directory${NC}"
    echo ""
    echo "Please specify the path to your template directory (the one containing .claude/):"
    echo "Example: /home/username/Dev/NS-hubspot-sales-tools"
    echo ""
    read -p "Template directory path: " template_path
    
    if [ -d "$template_path/.claude" ]; then
        echo "$template_path"
        return 0
    else
        echo -e "${RED}Error: Invalid template directory path${NC}"
        exit 1
    fi
}

# Function to initialize the directory
init_directory() {
    local dir_name="$1"
    local kebab_name=$(to_kebab_case "$dir_name")
    local template_dir=$(get_template_dir)
    
    echo -e "${BLUE}Initializing directory: $kebab_name${NC}"
    echo -e "${YELLOW}Template directory: $template_dir${NC}"
    
    # Create the main directory
    if [ -d "$kebab_name" ]; then
        echo -e "${RED}Error: Directory '$kebab_name' already exists${NC}"
        exit 1
    fi
    
    mkdir -p "$kebab_name"
    cd "$kebab_name"
    
    echo -e "${GREEN}✓ Created directory: $kebab_name${NC}"
    
    # Create .claude structure
    echo "Creating .claude structure..."
    mkdir -p .claude/agents
    mkdir -p .claude/commands
    
    # Copy all files from template .claude/agents/
    if [ -d "$template_dir/.claude/agents" ]; then
        cp -r "$template_dir/.claude/agents/"* .claude/agents/ 2>/dev/null || true
        echo -e "${GREEN}✓ Copied .claude/agents/ files${NC}"
    else
        echo -e "${YELLOW}Warning: No template .claude/agents/ directory found${NC}"
    fi
    
    # Copy all files from template .claude/commands/
    if [ -d "$template_dir/.claude/commands" ]; then
        cp -r "$template_dir/.claude/commands/"* .claude/commands/ 2>/dev/null || true
        echo -e "${GREEN}✓ Copied .claude/commands/ files${NC}"
    else
        echo -e "${YELLOW}Warning: No template .claude/commands/ directory found${NC}"
    fi
    
    # Copy .claude/settings.local.json if it exists
    if [ -f "$template_dir/.claude/settings.local.json" ]; then
        cp "$template_dir/.claude/settings.local.json" .claude/
        echo -e "${GREEN}✓ Copied .claude/settings.local.json${NC}"
    fi
    
    # Create hack directory
    echo "Creating hack directory..."
    mkdir -p hack
    
    # Copy all files from template hack/ directory
    if [ -d "$template_dir/hack" ]; then
        cp -r "$template_dir/hack/"* hack/ 2>/dev/null || true
        echo -e "${GREEN}✓ Copied hack/ directory${NC}"
    else
        echo -e "${YELLOW}Warning: No template hack/ directory found${NC}"
    fi
    
    # Create standard project directories
    echo "Creating standard project directories..."
    mkdir -p frontend
    mkdir -p backend
    mkdir -p src
    
    echo -e "${GREEN}✓ Created frontend/, backend/, and src/ directories${NC}"
    
    # Create a basic README.md
    cat > README.md << EOF
# $dir_name

This project was initialized with agent-init.

## Directory Structure

- \`.claude/\` - Claude AI configuration and commands
- \`hack/\` - Development and utility scripts
- \`frontend/\` - Frontend application code
- \`backend/\` - Backend application code
- \`src/\` - Source code

## Getting Started

1. Navigate to the project directory
2. Set up your development environment
3. Use the scripts in \`hack/\` for common development tasks

## Available Scripts

Check the \`hack/\` directory for available utility scripts.
EOF
    
    echo -e "${GREEN}✓ Created README.md${NC}"
    
    # Create a basic .gitignore
    cat > .gitignore << EOF
# Dependencies
node_modules/
venv/
env/
.env

# Build outputs
dist/
build/
*.pyc
__pycache__/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/

# nyc test coverage
.nyc_output

# Dependency directories
jspm_packages/

# Optional npm cache directory
.npm

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variables file
.env

# next.js build output
.next

# Nuxt.js build output
.nuxt

# vuepress build output
.vuepress/dist

# Serverless directories
.serverless

# FuseBox cache
.fusebox/

# DynamoDB Local files
.dynamodb/

# TernJS port file
.tern-port

# Stores VSCode versions used for testing VSCode extensions
.vscode-test
EOF
    
    echo -e "${GREEN}✓ Created .gitignore${NC}"
    
    # Show summary
    echo ""
    echo -e "${GREEN}🎉 Directory initialization complete!${NC}"
    echo ""
    echo -e "${BLUE}Created:${NC}"
    echo -e "  📁 $kebab_name/"
    echo -e "  📁 $kebab_name/.claude/agents/ ($(ls -1 .claude/agents/ | wc -l) files)"
    echo -e "  📁 $kebab_name/.claude/commands/ ($(ls -1 .claude/commands/ | wc -l) files)"
    echo -e "  📁 $kebab_name/hack/ ($(ls -1 hack/ | wc -l) files)"
    echo -e "  📁 $kebab_name/frontend/"
    echo -e "  📁 $kebab_name/backend/"
    echo -e "  📁 $kebab_name/src/"
    echo -e "  📄 $kebab_name/README.md"
    echo -e "  📄 $kebab_name/.gitignore"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  cd $kebab_name"
    echo -e "  git init"
    echo -e "  # Start developing!"
    echo ""
}

# Main logic
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
    exit 0
fi

# Initialize the directory
init_directory "$1"
