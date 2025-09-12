#!/bin/bash
#
# Wyoming VAD ASR Service Startup Script
# Manages the Wyoming VAD ASR service as a drop-in replacement for Docker Whisper
#

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/wyoming_vad_asr_server.py"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
PID_FILE="$SCRIPT_DIR/.parakeet_wyoming_server.pid"
LOG_FILE="$SCRIPT_DIR/logs/parakeet_wyoming.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        return 1
    fi
    
    # Check server script
    if [ ! -f "$SERVER_SCRIPT" ]; then
        print_error "Server script not found: $SERVER_SCRIPT"
        return 1
    fi
    
    # Check config file
    if [ ! -f "$CONFIG_FILE" ]; then
        print_warning "Config file not found: $CONFIG_FILE"
        print_status "Server will use default configuration"
    fi
    
    print_success "Dependencies checked"
    return 0
}

stop_docker_whisper() {
    print_status "Checking Docker Whisper service..."
    
    if docker ps -q -f name=wyoming-whisper | grep -q .; then
        print_status "Stopping Docker Whisper service..."
        docker stop wyoming-whisper
        print_success "Docker Whisper service stopped"
    else
        print_status "Docker Whisper service not running"
    fi
}

start_docker_whisper() {
    print_status "Starting Docker Whisper service..."
    
    if [ -f "$SCRIPT_DIR/compose/docker-compose.yml" ]; then
        cd "$SCRIPT_DIR"
        docker-compose -f compose/docker-compose.yml up -d wyoming-whisper
        print_success "Docker Whisper service started"
    else
        print_error "Docker compose file not found"
        return 1
    fi
}

is_server_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

start_server() {
    if is_server_running; then
        print_warning "VAD ASR server is already running (PID: $(cat $PID_FILE))"
        return 0
    fi
    
    print_status "Starting Wyoming VAD ASR server..."
    
    # Stop Docker Whisper first
    stop_docker_whisper
    
    # Check dependencies
    if ! check_dependencies; then
        return 1
    fi
    
    # Start server
    local cmd="python3 $SERVER_SCRIPT"
    if [ -f "$CONFIG_FILE" ]; then
        cmd="$cmd --config $CONFIG_FILE"
    fi
    
    print_status "Running: $cmd"
    
    # Start in background and capture PID
    nohup $cmd > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    # Wait a moment and check if it's still running
    sleep 2
    if ps -p "$pid" > /dev/null 2>&1; then
        print_success "VAD ASR server started successfully (PID: $pid)"
        print_status "Server is listening on port 10300"
        print_status "Logs: $LOG_FILE"
        return 0
    else
        print_error "Failed to start VAD ASR server"
        rm -f "$PID_FILE"
        print_status "Check logs for details: $LOG_FILE"
        return 1
    fi
}

stop_server() {
    if ! is_server_running; then
        print_warning "VAD ASR server is not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    print_status "Stopping VAD ASR server (PID: $pid)..."
    
    kill "$pid"
    
    # Wait for graceful shutdown
    local count=0
    while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if ps -p "$pid" > /dev/null 2>&1; then
        print_warning "Force killing server..."
        kill -9 "$pid"
    fi
    
    rm -f "$PID_FILE"
    print_success "VAD ASR server stopped"
}

restart_server() {
    print_status "Restarting VAD ASR server..."
    stop_server
    sleep 1
    start_server
}

show_status() {
    echo
    echo "=== Wyoming VAD ASR Service Status ==="
    
    # Server status
    if is_server_running; then
        local pid=$(cat "$PID_FILE")
        print_success "VAD ASR Server: RUNNING (PID: $pid)"
        
        # Show process details
        local ps_info=$(ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers 2>/dev/null)
        if [ -n "$ps_info" ]; then
            echo "  Process Info: $ps_info"
        fi
        
        # Test connection
        print_status "Testing connection..."
        if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 10300)); s.close(); print('✅ Port 10300 accessible')" 2>/dev/null; then
            print_success "Service is accessible on port 10300"
        else
            print_warning "Service may not be responding on port 10300"
        fi
    else
        print_error "VAD ASR Server: STOPPED"
    fi
    
    # Docker Whisper status
    if docker ps -q -f name=wyoming-whisper | grep -q .; then
        print_warning "Docker Whisper: RUNNING (conflicts with VAD ASR)"
    else
        print_status "Docker Whisper: STOPPED"
    fi
    
    # Show recent logs
    if [ -f "$LOG_FILE" ]; then
        echo
        echo "Recent logs:"
        tail -n 5 "$LOG_FILE" | sed 's/^/  /'
    fi
    
    echo "=================================="
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo "=== VAD ASR Server Logs ==="
        tail -f "$LOG_FILE"
    else
        print_error "Log file not found: $LOG_FILE"
        return 1
    fi
}

test_service() {
    print_status "Testing Wyoming service discovery..."
    
    if [ -f "$SCRIPT_DIR/wyoming_test_client.py" ]; then
        python3 "$SCRIPT_DIR/wyoming_test_client.py" --uri tcp://localhost:10300
    else
        print_error "Test client not found: $SCRIPT_DIR/wyoming_test_client.py"
        return 1
    fi
}

switch_to_docker() {
    print_status "Switching back to Docker Whisper..."
    stop_server
    start_docker_whisper
    print_success "Switched to Docker Whisper service"
}

install_service() {
    print_status "Installing systemd service..."
    "$SCRIPT_DIR/install_service.sh"
}

uninstall_service() {
    print_status "Uninstalling systemd service..."
    "$SCRIPT_DIR/uninstall_service.sh"
}

setup_log_management() {
    print_status "Setting up log management..."
    "$SCRIPT_DIR/setup_log_management.sh"
}

cleanup_logs() {
    print_status "Cleaning up old logs..."
    
    # Clean voice service logs
    find "$SCRIPT_DIR/logs" -name "*.log*" -type f -mtime +7 -delete 2>/dev/null || true
    
    # Clean systemd journal (keep 3 days)
    sudo journalctl --vacuum-time=3d
    
    # Clean Docker logs
    docker system prune -f --filter "until=72h" 2>/dev/null || true
    
    print_success "Log cleanup completed"
}

show_help() {
    echo "Parakeet Wyoming Wrapper Management Script"
    echo
    echo "Usage: $0 {start|stop|restart|status|logs|test|switch-to-docker|install-service|setup-logs|cleanup-logs|help}"
    echo
    echo "Manual Management Commands:"
    echo "  start           Start the VAD ASR server (stops Docker Whisper)"
    echo "  stop            Stop the VAD ASR server"
    echo "  restart         Restart the VAD ASR server"
    echo "  status          Show service status and recent logs"
    echo "  logs            Show live server logs"
    echo "  test            Test Wyoming protocol compliance"
    echo "  switch-to-docker Switch back to Docker Whisper service"
    echo
    echo "Service Installation:"
    echo "  install-service Install as systemd service (auto-start, auto-restart)"
    echo "  uninstall-service Remove systemd service"
    echo
    echo "Log Management:"
    echo "  setup-logs      Setup automatic log rotation and cleanup"
    echo "  cleanup-logs    Manually clean old logs now"
    echo
    echo "  help            Show this help message"
    echo
    echo "Files:"
    echo "  Server:  $SERVER_SCRIPT"
    echo "  Config:  $CONFIG_FILE"
    echo "  PID:     $PID_FILE"
    echo "  Logs:    $LOG_FILE"
    echo
    echo "For production use, consider 'install-service' for automatic management!"
}

# Main script logic
case "${1:-help}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    test)
        test_service
        ;;
    switch-to-docker)
        switch_to_docker
        ;;
    install-service)
        install_service
        ;;
    uninstall-service)
        uninstall_service
        ;;
    setup-logs)
        setup_log_management
        ;;
    cleanup-logs)
        cleanup_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac