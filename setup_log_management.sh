#!/bin/bash
#
# Setup log management for voice services
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🧹 Setting up log management for voice services"
echo "=============================================="

# 1. Configure systemd journal limits
echo -e "${BLUE}Configuring systemd journal limits...${NC}"

# Create journal configuration if it doesn't exist
if [ ! -f /etc/systemd/journald.conf.d/voice-services.conf ]; then
    sudo mkdir -p /etc/systemd/journald.conf.d
    sudo tee /etc/systemd/journald.conf.d/voice-services.conf > /dev/null << 'EOF'
[Journal]
# Limit journal size to 500MB total
SystemMaxUse=500M
# Keep only 1 week of logs
MaxRetentionSec=1week
# Limit individual log files to 50MB
SystemMaxFileSize=50M
# Forward to rsyslog (optional)
ForwardToSyslog=no
# Compress logs
Compress=yes
EOF

    echo -e "${GREEN}✅ Created systemd journal limits${NC}"
else
    echo -e "${YELLOW}Journal limits already configured${NC}"
fi

# 2. Setup logrotate for voice services logs
echo -e "${BLUE}Configuring logrotate for voice services...${NC}"

sudo tee /etc/logrotate.d/voice-services > /dev/null << 'EOF'
/home/attila/voice-services/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 attila attila
    postrotate
        # Send HUP signal to restart logging if service is running
        systemctl is-active wyoming-vad-asr >/dev/null && systemctl reload-or-restart wyoming-vad-asr || true
    endscript
}
EOF

    echo -e "${GREEN}✅ Created logrotate configuration${NC}"

# 3. Setup cleanup cron job
echo -e "${BLUE}Setting up automated cleanup...${NC}"

# Create cleanup script
sudo tee /usr/local/bin/voice-services-cleanup > /dev/null << 'EOF'
#!/bin/bash
# Voice services log cleanup script

LOG_DIR="/home/attila/voice-services/logs"
MAX_LOG_AGE_DAYS=14

# Clean old log files
find "$LOG_DIR" -name "*.log*" -type f -mtime +$MAX_LOG_AGE_DAYS -delete 2>/dev/null || true

# Clean systemd journal (keep only 1 week)
journalctl --vacuum-time=1week >/dev/null 2>&1 || true

# Clean Docker logs if they exist
docker system prune -f --filter "until=168h" >/dev/null 2>&1 || true

# Log the cleanup
echo "$(date): Voice services log cleanup completed" >> /var/log/voice-services-cleanup.log
EOF

sudo chmod +x /usr/local/bin/voice-services-cleanup

# Add to crontab (runs daily at 2 AM)
if ! sudo crontab -l 2>/dev/null | grep -q "voice-services-cleanup"; then
    (sudo crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/voice-services-cleanup") | sudo crontab -
    echo -e "${GREEN}✅ Added daily cleanup cron job${NC}"
else
    echo -e "${YELLOW}Cleanup cron job already exists${NC}"
fi

# 4. Configure Docker log limits
echo -e "${BLUE}Configuring Docker log limits...${NC}"

# Create or update Docker daemon configuration
DOCKER_CONFIG="/etc/docker/daemon.json"
if [ -f "$DOCKER_CONFIG" ]; then
    # Backup existing config
    sudo cp "$DOCKER_CONFIG" "${DOCKER_CONFIG}.backup"
fi

sudo tee "$DOCKER_CONFIG" > /dev/null << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    }
}
EOF

echo -e "${GREEN}✅ Configured Docker logging limits${NC}"

# 5. Restart services to apply changes
echo -e "${BLUE}Applying configuration changes...${NC}"

sudo systemctl restart systemd-journald
sudo systemctl restart docker || echo "Docker restart failed (may not be critical)"

echo -e "${GREEN}🎉 Log management setup complete!${NC}"
echo
echo "Configuration Summary:"
echo "  📊 Systemd journal: Max 500MB, 1 week retention"
echo "  🔄 Voice service logs: Daily rotation, 7 days retention"
echo "  🐳 Docker logs: Max 10MB per file, 3 files per container"
echo "  🧹 Automatic cleanup: Daily at 2 AM"
echo
echo "Manual cleanup commands:"
echo "  journalctl --vacuum-time=1d       # Clean journal to 1 day"
echo "  sudo /usr/local/bin/voice-services-cleanup  # Run cleanup now"
echo "  docker system prune -f            # Clean Docker"