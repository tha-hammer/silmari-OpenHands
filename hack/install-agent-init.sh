#!/usr/bin/env bash
# install-agent-init.sh - Install agent-init script globally

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Installing agent-init script globally...${NC}"

# Get the current directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_INIT_SCRIPT="$SCRIPT_DIR/agent-init.sh"

# Check if agent-init.sh exists
if [ ! -f "$AGENT_INIT_SCRIPT" ]; then
    echo -e "${RED}Error: agent-init.sh not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Create ~/bin directory if it doesn't exist
mkdir -p "$HOME/bin"

# Copy the script to ~/bin
cp "$AGENT_INIT_SCRIPT" "$HOME/bin/agent-init"
chmod +x "$HOME/bin/agent-init"

echo -e "${GREEN}✓ Copied agent-init.sh to $HOME/bin/agent-init${NC}"

# Check if ~/bin is in PATH
if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
    echo -e "${YELLOW}Warning: $HOME/bin is not in your PATH${NC}"
    echo ""
    echo "Add this line to your ~/.bashrc:"
    echo -e "${BLUE}export PATH=\"\$HOME/bin:\$PATH\"${NC}"
    echo ""
    echo "Then run:"
    echo -e "${BLUE}source ~/.bashrc${NC}"
    echo ""
    echo "Or add it manually:"
    echo -e "${BLUE}echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.bashrc${NC}"
else
    echo -e "${GREEN}✓ $HOME/bin is already in your PATH${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Installation complete!${NC}"
echo ""
echo -e "${BLUE}Usage:${NC}"
echo "  agent-init <directory-name>"
echo ""
echo -e "${BLUE}Examples:${NC}"
echo "  agent-init my-new-project"
echo "  agent-init \"My New Project\""
echo "  agent-init my_awesome_project"
echo ""
echo -e "${YELLOW}Note: The script will automatically find your template directory${NC}"
echo -e "${YELLOW}at $SCRIPT_DIR (where .claude/ is located)${NC}"
