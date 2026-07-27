import os
import logging


class Settings:
    def __init__(self):
        env_file_path = os.environ.get("RORY_WORKER_ENV_FILE_PATH")
        if not env_file_path:
            env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        if os.path.exists(env_file_path):
            from dotenv import load_dotenv
            load_dotenv(env_file_path)

        self.debug = bool(int(os.environ.get("RORY_DEBUG", 1)))

        self.node_id = os.environ.get("NODE_ID", "rory-worker-0")
        self.node_port = int(os.environ.get("NODE_PORT", 9000))
        self.node_index = int(os.environ.get("NODE_INDEX", 0))
        self.host_port = os.environ.get("HOST_PORT", self.node_port + self.node_index)
        self.max_retries = int(os.environ.get("MAX_RETRIES", 100))
        self.rory_manager_port = int(os.environ.get("RORY_MANAGER_PORT", 6000))
        self.reload_flag = bool(int(os.environ.get("RELOAD", 0)))
        self.rory_manager_ip_addr = os.environ.get("RORY_MANAGER_IP_ADDR", "localhost")
        self.node_ip_addr = os.environ.get("NODE_IP_ADDR", self.node_id)
        self.server_ip_addr = os.environ.get("SERVER_IP_ADDR", "0.0.0.0")

        self.distance = os.environ.get("DISTANCE", "MANHATHAN")
        self.min_error = float(os.environ.get("MIN_ERROR", 0.015))

        self.ckks_round = bool(int(os.environ.get("CKKS_ROUND", 0)))
        self.ckks_decimals = int(os.environ.get("CKKS_DECIMALS", 2))
        self.ctx_filename = os.environ.get("CTX_FILENAME", "ctx")
        self.pubkey_filename = os.environ.get("PUBKEY_FILENAME", "pubkey")
        self.secret_key_filename = os.environ.get("SECRET_KEY_FILENAME", "secretkey")
        self.relinkey_filename = os.environ.get("RELINKEY_FILENAME", "relinkey")
        self.rotatekey_filename = os.environ.get("ROTATEKEY_FILENAME", "rotatekey")

        self.source_path = os.environ.get("SOURCE_PATH", "/rory/source")
        self.sink_path = os.environ.get("SINK_PATH", "/rory/sink")
        self.log_path = os.environ.get("LOG_PATH", "/rory/log")
        self.keys_path = os.environ.get("KEYS_PATH", "/rory/keys")
        for path in [self.source_path, self.sink_path, self.log_path]:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print("MAKE_FOLDER_ERROR", e)

        self.mictlanx_timeout = int(os.environ.get("MICTLANX_TIMEOUT", 120))
        self.mictlanx_client_id = os.environ.get("MICTLANX_CLIENT_ID", f"{self.node_id}_mictlanx")
        self.mictlanx_api_version = int(os.environ.get("MICTLANX_API_VERSION", "3"))
        self.mictlanx_routers = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666")
        self.mictlanx_debug = bool(int(os.environ.get("MICTLANX_DEBUG", 0)))
        self.mictlanx_max_workers = int(os.environ.get("MICTLANX_MAX_WORKERS", "4"))
        self.mictlanx_log_path = os.environ.get("MICTLANX_LOG_PATH", "/rory/mictlanx")
        self.mictlanx_log_interval = int(os.environ.get("MICTLANX_LOG_INTERVAL", "24"))
        self.mictlanx_log_when = os.environ.get("MICTLANX_LOG_WHEN", "h")
        self.mictlanx_bucket_id = os.environ.get("MICTLANX_BUCKET_ID", "rory")
        self.mictlanx_delay = int(os.environ.get("MICTLANX_DELAY", "2"))
        self.mictlanx_backoff_factor = float(os.environ.get("MICTLANX_BACKOFF_FACTOR", "0.5"))
        self.mictlanx_max_retries = int(os.environ.get("MICTLANX_MAX_RETRIES", "10"))
        self.mictlanx_chunk_size = os.environ.get("MICTLANX_CHUNK_SIZE", "256kb")
        self.mictlanx_max_parallel_gets = int(os.environ.get("MICTLANX_MAX_PARALELL_GETS", "2"))
        self.mictlanx_protocol = os.environ.get("MICTLANX_PROTOCOL", "http")
        mictlanx_api_version = int(os.environ.get("MICTLANX_API_VERSION", "4"))
        self.mictlanx_log_disable = bool(int(os.environ.get("MICTLANX_LOG_DISABLE", "1")))
        self.mictlanx_uri = os.environ.get("MICTLANX_URI", f"mictlanx://mictlanx-router-0@localhost:63666?api_version={mictlanx_api_version}&protocol={self.mictlanx_protocol}")

        self.num_chunks = int(os.environ.get("NUM_CHUNKS", "4"))

        self.rory_worker_log_interval = int(os.environ.get("RORY_WORKER_LOG_INTERVAL", 24))
        self.rory_worker_log_when = os.environ.get("RORY_WORKER_LOG_WHEN", "h")
        json_indent = os.environ.get("RORY_WORKER_LOG_JSON_FORMATTER_INDENT", 4)
        self.rory_worker_log_json_formatter_indent = json_indent
        self.rory_worker_log_disabled = bool(int(os.environ.get("RORY_WORKER_LOG_DISABLED", "1")))
        self.rory_worker_log_error_to_file = bool(int(os.environ.get("RORY_WORKER_LOG_ERROR_TO_FILE", "1")))
        self.rory_worker_log_file_handler_level = int(
            os.environ.get("RORY_WORKER_LOG_FILE_HANDLER_LEVEL", logging.INFO)
        )
        self.rory_worker_log_level = int(os.environ.get("RORY_WORKER_LOG_LEVEL", logging.DEBUG))
        self.rory_worker_log_to_file = bool(int(os.environ.get("RORY_WORKER_LOG_TO_FILE", "1")))
        self.rory_worker_log_use_rich = bool(int(os.environ.get("RORY_WORKER_LOG_USE_RICH", "0")))
