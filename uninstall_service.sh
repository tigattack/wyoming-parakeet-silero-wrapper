#!/bin/bash
#
# Uninstall Wyoming VAD ASR systemd service
#

SERVICE_NAME="wyoming-vad-asr"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🗑️  Uninstalling Wyoming VAD ASR Service"
echo "======================================="

# Stop and disable service
echo -e "${YELLOW}Stopping and disabling service...${NC}"
sudo systemctl stop "$SERVICE_NAME" || true
sudo systemctl disable "$SERVICE_NAME" || true

# Remove service file
echo -e "${YELLOW}Removing service file...${NC}"
sudo rm -f "/etc/systemd/system/$SERVICE_NAME.service"

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
sudo systemctl daemon-reload

echo -e "${GREEN}✅ Service uninstalled successfully!${NC}"
echo
echo "You can now manage the service manually with:"
echo "  ./start_vad_asr.sh start|stop|restart|status"