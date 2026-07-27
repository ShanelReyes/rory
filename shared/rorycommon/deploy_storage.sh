#!/bin/bash

echo "Creating [mictlanx] network..."
readonly ENV_FILE=${1:-".env.mictlanx.dev"}
readonly MICTLANX_ROUTER_PORT=${2:-63666}

# Check if jq is installed if not say to the user to install it
if ! command -v jq &> /dev/null; then
    echo "jq is required but not installed. Please install jq and try again."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    exit 1
fi

docker network create --driver=bridge mictlanx || true

echo "Removing existing routers"
docker compose -p mictlanx --env-file $ENV_FILE -f storage.yml down
sleep 3
echo "Starting a new MictlanX Cluster with router on port $MICTLANX_ROUTER_PORT..."
docker compose -p mictlanx --env-file $ENV_FILE -f storage.yml up -d
# -------------------------------
# Healthcheck: wait for peers
# -------------------------------
API="http://localhost:${MICTLANX_ROUTER_PORT}/api/v4/peers/stats"
TIMEOUT=180
DEADLINE=$((SECONDS + TIMEOUT))   # timeout after 180s; adjust as needed

echo "Waiting for peers to appear at $API ..."
sleep 5  # initial delay before first check; adjust as needed

while true; do
  # fetch JSON (fail on non-2xx; quiet errors to stderr)
  if json="$(curl -fsS "$API")"; then
    count="$(jq -r 'length' <<<"$json" 2>/dev/null || echo 0)"
    if [[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -gt 0 ]; then
      echo "✅ MictlanX cluster is healthy (peers: $count)"
      break
    else
      echo "⏳ No peers yet (length=$count)"
    fi
  else
    echo "⏳ API not ready yet (curl failed)"
  fi

  # optional timeout
  if [ $SECONDS -ge $DEADLINE ]; then
    echo "❌ Timed out waiting for peers at $API"
    exit 1
  fi
  sleep 1
  
done
