readonly COMPOSE_FILE=${4:-docker-compose.yml}
readonly ENV_FILE_PATH=${3:-.env.dev}
readonly BUILD_MODE=${1:-0}
readonly DETACHED_MODE=${2:-0}
readonly detached_flag=$([ "$DETACHED_MODE" -eq 1 ] && echo "-d" || echo "")


if [ "$BUILD_MODE" -eq 1 ]; then
    echo "Building Docker images..."
    docker compose --env-file ${ENV_FILE_PATH} -f ./${COMPOSE_FILE} ${detached_flag} up --build 
else
    docker compose --env-file ${ENV_FILE_PATH} -f ./${COMPOSE_FILE} ${detached_flag} up
fi
