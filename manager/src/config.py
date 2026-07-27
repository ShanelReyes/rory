import os
import logging


class Settings:
    def __init__(self):
        env_file_path = os.environ.get("RORY_MANAGER_ENV_FILE_PATH")
        if not env_file_path:
            env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        if os.path.exists(env_file_path):
            from dotenv import load_dotenv
            load_dotenv(env_file_path)

        self.debug = bool(int(os.environ.get("RORY_DEBUG", 0)))

        self.node_id = os.environ.get("NODE_ID", "rory-manager-0")
        self.node_ip_addr = os.environ.get("NODE_IP_ADDR", self.node_id)
        self.node_port = int(os.environ.get("NODE_PORT", 6000))
        self.server_ip_addr = os.environ.get("SERVER_IP_ADDR", "0.0.0.0")
        self.node_prefix = os.environ.get("NODE_PREFIX", "rory-worker-")
        self.folder_keys = os.environ.get("FOLDER_KEYS", "keys128")

        self.init_workers = int(os.environ.get("INIT_WORKERS", "0"))
        self.init_port = int(os.environ.get("WORKER_INIT_PORT", 9000))
        self.docker_image_name = os.environ.get("DOCKER_IMAGE_NAME", "shanelreyes/rory")
        self.docker_image_tag = os.environ.get("DOCKER_IMAGE_TAG", "worker")
        self.docker_image = f"{self.docker_image_name}:{self.docker_image_tag}"
        self.docker_network_id = os.environ.get("DOCKER_NETWORK_ID", "mictlanx")
        self.max_retries = int(os.environ.get("MAX_RETRIES", 100))
        self.load_balancing = int(os.environ.get("LOAD_BALANCING", "0"))
        self.worker_max_threads = int(os.environ.get("WORKER_MAX_THREADS", 2))
        self.worker_memory = os.environ.get("WORKER_MEMORY", "1000000000")
        self.worker_cpu = os.environ.get("WORKER_CPU", 2)
        self.worker_timeout = int(os.environ.get("WORKER_TIMEOUT", 300))
        self.swarm_nodes = os.environ.get("SWARM_NODES", "2,3,4,8").split(",")
        self.liu_round = int(os.environ.get("LIU_ROUND", "2"))

        self.distance = os.environ.get("DISTANCE", "MANHATHAN")
        self.min_error = float(os.environ.get("MIN_ERROR", 0.015))
        self.ckks_round = int(os.environ.get("CKKS_ROUND", 0))
        self.ckks_decimals = int(os.environ.get("CKKS_DECIMALS", 2))
        self.ctx_filename = os.environ.get("CTX_FILENAME", "ctx")
        self.pubkey_filename = os.environ.get("PUBKEY_FILENAME", "pubkey")
        self.secret_key_filename = os.environ.get("SECRET_KEY_FILENAME", "secretkey")
        self.relinkey_filename = os.environ.get("RELINKEY_FILENAME", "relinkey")

        self.source_path = os.environ.get("SOURCE_PATH", "/rory/source")
        self.sink_path = os.environ.get("SINK_PATH", "/rory/sink")
        self.log_path = os.environ.get("LOG_PATH", "/rory/log")
        for path in [self.source_path, self.sink_path, self.log_path]:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print("MAKE_FOLDER_ERROR", e)

        self.mictlanx_summoner_ip_addr = os.environ.get("MICTLANX_SUMMONER_IP_ADDR", "localhost")
        self.mictlanx_summoner_port = int(os.environ.get("MICTLANX_SUMMONER_PORT", 15000))
        self.mictlanx_summoner_mode = os.environ.get("MICTLANX_SUMMONER_MODE", "docker")
        self.mictlanx_api_version = int(os.environ.get("MICTLANX_API_VERSION", 3))
        self.mictlanx_client_id = os.environ.get("MICTLANX_CLIENT_ID", "CLIENT_ID")
        self.mictlanx_uri = os.environ.get(
            "MICTLANX_URI",
            "mictlanx://mictlanx-router-0@localhost:63666?api_version=4&protocol=http",
        )
        self.mictlanx_timeout = int(os.environ.get("MICTLANX_TIMEOUT", 120))
        self.mictlanx_max_workers = int(os.environ.get("MICTLANX_MAX_WORKERS", 12))
        self.mictlanx_debug = bool(int(os.environ.get("MICTLANX_DEBUG", 0)))
        self.mictlanx_log_path = os.environ.get("MICTLANX_LOG_PATH", "/rory/mictlanx")
        self.mictlanx_log_interval = os.environ.get("MICTLANX_LOG_INTERVAL", "24")
        self.mictlanx_log_when = os.environ.get("MICTLANX_LOG_WHEN", "h")
        self.mictlanx_protocol = os.environ.get("MICTLANX_PROTOCOL", "https")
        self.mictlanx_bucket_id = os.environ.get("MICTLANX_BUCKET_ID", "rory")
        self.mictlanx_delay = int(os.environ.get("MICTLANX_DELAY", "2"))
        self.mictlanx_backoff_factor = float(os.environ.get("MICTLANX_BACKOFF_FACTOR", "0.5"))
        self.mictlanx_max_retries = int(os.environ.get("MICTLANX_MAX_RETRIES", "10"))
        self.mictlanx_chunk_size = os.environ.get("MICTLANX_CHUNK_SIZE", "256kb")
        self.mictlanx_max_parallel_gets = int(os.environ.get("MICTLANX_MAX_PARALELL_GETS", "2"))

        self.rory_manager_log_interval = int(os.environ.get("RORY_MANAGER_LOG_INTERVAL", 24))
        self.rory_manager_log_when = os.environ.get("RORY_MANAGER_LOG_WHEN", "h")
        json_indent = os.environ.get("RORY_MANAGER_LOG_JSON_FORMATTER_INDENT", 4)
        self.rory_manager_log_json_formatter_indent = json_indent
        self.rory_manager_log_disabled = bool(int(os.environ.get("RORY_MANAGER_LOG_DISABLED", "1")))
        self.rory_manager_log_error_to_file = bool(int(os.environ.get("RORY_MANAGER_LOG_ERROR_TO_FILE", "1")))
        self.rory_manager_log_file_handler_level = int(
            os.environ.get("RORY_MANAGER_LOG_FILE_HANDLER_LEVEL", logging.INFO)
        )
        self.rory_manager_log_level = int(os.environ.get("RORY_MANAGER_LOG_LEVEL", logging.DEBUG))
        self.rory_manager_log_to_file = bool(int(os.environ.get("RORY_MANAGER_LOG_TO_FILE", "1")))
        self.rory_manager_log_use_rich = bool(int(os.environ.get("RORY_MANAGER_LOG_USE_RICH", "0")))
