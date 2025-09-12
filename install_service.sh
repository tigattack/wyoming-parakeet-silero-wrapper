#!/bin/bash
#
# Install Wyoming VAD ASR as a systemd service
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/wyoming-vad-asr.service"
SERVICE_NAME="wyoming-vad-asr"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Installing Wyoming VAD ASR Service"
echo "===================================="

# Stop current manual service if running
if pgrep -f "wyoming_vad_asr_server.py" > /dev/null; then
    echo -e "${YELLOW}Stopping current manual service...${NC}"
    "$SCRIPT_DIR/start_vad_asr.sh" stop || true
fi

# Copy service file to systemd
echo -e "${YELLOW}Installing systemd service...${NC}"
sudo cp "$SERVICE_FILE" "/etc/systemd/system/$SERVICE_NAME.service"

# Reload systemd and enable service
echo -e "${YELLOW}Enabling service for auto-start...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

# Start the service
echo -e "${YELLOW}Starting service...${NC}"
sudo systemctl start "$SERVICE_NAME"

# Wait a moment and check status
sleep 3
if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${GREEN}✅ Service installed and started successfully!${NC}"
    echo
    echo "Service Management Commands:"
    echo "  sudo systemctl start $SERVICE_NAME     # Start service"
    echo "  sudo systemctl stop $SERVICE_NAME      # Stop service"
    echo "  sudo systemctl restart $SERVICE_NAME   # Restart service"
    echo "  sudo systemctl status $SERVICE_NAME    # Check status"
    echo "  journalctl -u $SERVICE_NAME -f         # View logs"
    echo
    echo "The service will now:"
    echo "  - Start automatically on boot"
    echo "  - Restart automatically if it crashes"
    echo "  - Log to systemd journal"
else
    echo -e "${RED}❌ Service failed to start!${NC}"
    echo "Check logs with: journalctl -u $SERVICE_NAME -f"
    exit 1
fi

echo -e "${GREEN}🎉 Installation complete!${NC}"