#!/bin/bash

# V7P3R Production VM Diagnostic Script
# Created: 2026-03-01
# Purpose: Diagnose time forfeit issue and identify running version

echo "======================================"
echo "V7P3R PRODUCTION VM DIAGNOSTIC"
echo "======================================"
echo "Timestamp: $(date)"
echo ""

# VM Resource Information
echo "--- VM RESOURCE INFORMATION ---"
echo "CPU Cores: $(nproc)"
echo "Total Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Available Memory: $(free -h | grep Mem | awk '{print $7}')"
echo "Memory Usage: $(free -h | grep Mem | awk '{print $3}')"
echo ""

# Docker Container Status
echo "--- DOCKER CONTAINER STATUS ---"
sudo docker ps -a --filter "name=v7p3r" --format "table {{.Names}}\t{{.Status}}\t{{.State}}\t{{.CreatedAt}}"
echo ""

# Container Resource Usage
echo "--- CONTAINER RESOURCE USAGE ---"
sudo docker stats v7p3r-production --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""

# Container Uptime & Restart Count
echo "--- CONTAINER UPTIME & RESTARTS ---"
CONTAINER_ID=$(sudo docker ps -aq --filter "name=v7p3r-production" | head -1)
if [ -n "$CONTAINER_ID" ]; then
    echo "Container ID: $CONTAINER_ID"
    sudo docker inspect $CONTAINER_ID --format "Created: {{.Created}}"
    sudo docker inspect $CONTAINER_ID --format "Started: {{.State.StartedAt}}"
    sudo docker inspect $CONTAINER_ID --format "Restart Count: {{.RestartCount}}"
    sudo docker inspect $CONTAINER_ID --format "Status: {{.State.Status}}"
    sudo docker inspect $CONTAINER_ID --format "OOMKilled: {{.State.OOMKilled}}"
else
    echo "No container found with name 'v7p3r-production'"
fi
echo ""

# Engine Version Check
echo "--- ENGINE VERSION CHECK ---"
echo "Attempting UCI version query..."
if sudo docker exec v7p3r-production bash -c 'echo uci | timeout 5 python /lichess-bot/engines/v7p3r/v7p3r_uci.py 2>/dev/null | grep "id name"'; then
    echo "UCI query successful"
else
    echo "UCI query failed or timed out"
fi

echo ""
echo "Checking v7p3r.py header..."
if sudo docker exec v7p3r-production head -10 /lichess-bot/engines/v7p3r/v7p3r.py 2>/dev/null | grep -i "version\|v[0-9]"; then
    echo "Found version in v7p3r.py"
else
    echo "Could not read v7p3r.py header"
fi
echo ""

# Check if v7p3r is a symlink or directory
echo "--- ENGINE DIRECTORY STRUCTURE ---"
sudo docker exec v7p3r-production ls -la /lichess-bot/engines/ 2>/dev/null | grep v7p3r || echo "Could not list engine directory"
echo ""

# Lichess-Bot Process Check
echo "--- LICHESS-BOT PROCESS STATUS ---"
if sudo docker exec v7p3r-production ps aux 2>/dev/null | grep -E "python|lichess" | grep -v grep; then
    echo "Found Python/lichess processes"
else
    echo "No lichess-bot processes found"
fi
echo ""

# Check for OOM kills in system logs
echo "--- OOM KILL CHECK (Last 50 lines) ---"
sudo dmesg 2>/dev/null | grep -i "oom\|killed\|out of memory" | tail -50 || echo "No dmesg access or no OOM events"
echo ""

# Recent Docker Container Logs (Last 100 lines)
echo "--- RECENT CONTAINER LOGS (Last 100 lines) ---"
sudo docker logs v7p3r-production --tail 100 2>&1 | tail -100
echo ""

# Check disk space
echo "--- DISK SPACE ---"
df -h / || echo "Cannot check disk space"
echo ""

# Network connectivity test
echo "--- NETWORK CONNECTIVITY ---"
sudo docker exec v7p3r-production ping -c 3 lichess.org 2>/dev/null && echo "Network: OK" || echo "Network: FAILED"
echo ""

# Config file check
echo "--- CONFIG FILE CHECK ---"
if sudo docker exec v7p3r-production test -f /lichess-bot/config.yml; then
    echo "Config file exists at /lichess-bot/config.yml"
    echo ""
    echo "Concurrency setting:"
    sudo docker exec v7p3r-production grep -A1 "concurrency:" /lichess-bot/config.yml 2>/dev/null || echo "Could not read concurrency"
    echo ""
    echo "Move overhead setting:"
    sudo docker exec v7p3r-production grep "move_overhead:" /lichess-bot/config.yml 2>/dev/null || echo "Could not read move_overhead"
else
    echo "Config file NOT found at /lichess-bot/config.yml"
fi
echo ""

# Game records count
echo "--- GAME RECORDS ---"
GAME_COUNT=$(sudo docker exec v7p3r-production find /lichess-bot/game_records -name "*.pgn" 2>/dev/null | wc -l)
echo "Total PGN files: $GAME_COUNT"
echo ""

echo "======================================"
echo "DIAGNOSTIC COMPLETE"
echo "======================================"
echo ""
echo "NEXT STEPS:"
echo "1. Review memory usage - should be under 800MB on e2-micro"
echo "2. Check for OOM kills or high restart count"
echo "3. Verify engine version matches expected deployment"
echo "4. Review container logs for crashes or errors"
echo "5. If memory > 800MB or OOMKilled = true, reduce concurrency to 1"
echo ""
