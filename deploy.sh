#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$SCRIPT_DIR}"

usage() {
    echo "Usage: $0 {up|restart|down} [--build] [--mictlanx]"
    echo ""
    echo "Commands:"
    echo "  up       Deploy Rory services (starts MictlanX first)"
    echo "  restart  Restart Rory services"
    echo "  down     Stop Rory services"
    echo ""
    echo "Options:"
    echo "  --build     Build images before deploying/restarting"
    echo "  --mictlanx  Also stop MictlanX (only with 'down')"
    echo "  -h, --help  Show this help"
    exit 0
}

COMMAND=""
BUILD=false
DOWN_MICTLANX=false

for arg in "$@"; do
    case "$arg" in
        --build)      BUILD=true ;;
        --mictlanx)   DOWN_MICTLANX=true ;;
        -h|--help)    usage ;;
        -*)
            echo "Unknown option: $arg"
            usage
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$arg"
            else
                echo "Unknown argument: $arg"
                usage
            fi
            ;;
    esac
done

MICTLANX_ENV_FILE=".env.mictlanx.dev"

case "${COMMAND:-}" in
    up)
        if $BUILD; then
            echo "==> Building images..."
            docker compose build
        fi

        echo "==> Starting MictlanX..."
        (cd "$BASE_PATH/mictlanx" && bash run.sh "$MICTLANX_ENV_FILE")

        echo "==> Starting Rory services..."
        docker compose up -d

        echo "==> Done. Services:"
        docker compose ps
        ;;
    restart)
        if $BUILD; then
            echo "==> Building images..."
            docker compose build
            echo "==> Recreating containers with new images..."
            docker compose up -d --force-recreate
        else
            echo "==> Restarting Rory services..."
            docker compose restart
        fi

        echo "==> Done. Services:"
        docker compose ps
        ;;
    down)
        echo "==> Stopping Rory services..."
        docker compose down

        if $DOWN_MICTLANX; then
            echo "==> Stopping MictlanX..."
            (cd "$BASE_PATH/mictlanx" && docker compose -p mictlanx -f storage.yml down 2>/dev/null || true)
        fi

        echo "==> Done."
        ;;
    "")
        usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
