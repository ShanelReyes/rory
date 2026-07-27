import os
import logging


class Settings:
    def __init__(self):
        env_file_path = os.environ.get("ENV_FILE_PATH")
        if not env_file_path:
            env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        if os.path.exists(env_file_path):
            from dotenv import load_dotenv
            load_dotenv(env_file_path)

        self.debug = bool(int(os.environ.get("RORY_DEBUG", 0)))

        self.node_id = os.environ.get("NODE_ID", "rory-dataowner-0")
        self.node_ip_addr = os.environ.get("NODE_IP_ADDR", self.node_id)
        self.node_port = int(os.environ.get("NODE_PORT", 3000))
        self.server_ip_addr = os.environ.get("SERVER_IP_ADDR", "0.0.0.0")
        self.rory_manager_ip_addr = os.environ.get("RORY_MANAGER_IP_ADDR", "localhost")
        self.rory_manager_port = int(os.environ.get("RORY_MANAGER_PORT", 6000))

        self.max_workers = int(os.environ.get("MAX_WORKERS", 2))
        self.num_chunks = int(os.environ.get("NUM_CHUNKS", 4))
        self.worker_timeout = int(os.environ.get("WORKER_TIMEOUT", 300))
        self.max_iterations = int(os.environ.get("MAX_ITERATIONS", 10))

        self.liu_security_level = int(os.environ.get("LIU_SECURITY_LEVEL", "128"))
        self.liu_secure_random = bool(int(os.environ.get("LIU_SECURE_RANDOM", "0")))
        liu_seed_str = os.environ.get("LIU_SEED", "None")
        self.liu_seed = int(liu_seed_str) if liu_seed_str.isdigit() else None
        self.liu_use_np_random = bool(int(os.environ.get("LIU_USE_NP_RANDOM", "1")))
        self.liu_round = bool(int(os.environ.get("LIU_ROUND", "0")))
        self.liu_decimals = int(os.environ.get("LIU_DECIMALS", 6))

        self.ckks_round = bool(int(os.environ.get("CKKS_ROUND", 0)))
        self.ckks_decimals = int(os.environ.get("CKKS_DECIMALS", 2))
        self.ctx_filename = os.environ.get("CTX_FILENAME", "ctx")
        self.pubkey_filename = os.environ.get("PUBKEY_FILENAME", "pubkey")
        self.secret_key_filename = os.environ.get("SECRET_KEY_FILENAME", "secretkey")
        self.relinkey_filename = os.environ.get("RELINKEY_FILENAME", "relinkey")
        self.rotatekey_filename = os.environ.get("ROTATEKEY_FILENAME", "rotatekey")

        self.reload = bool(int(os.environ.get("RELOAD", 0)))
        self.np_random = bool(int(os.environ.get("NP_RANDOM", "1")))
        self.testing_env = os.environ.get("TESTING", "1")

        self.logger_name = os.environ.get("LOGGER_NAME", "rory-dataowner-0")
        self.source_path = os.environ.get("SOURCE_PATH", "/rory/source")
        self.sink_path = os.environ.get("SINK_PATH", "/rory/sink")
        self.log_path = os.environ.get("LOG_PATH", "/rory/log")
        self.keys_path = os.environ.get("KEYS_PATH", "/rory/keys")
        self.testing = bool(int(self.testing_env))

        for path in [self.source_path, self.sink_path, self.log_path]:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print("MAKE_FOLDER_ERROR", e)

        self.mictlanx_client_id = os.environ.get("MICTLANX_CLIENT_ID", f"{self.node_id}_mictlanx")
        self.mictlanx_timeout = int(os.environ.get("MICTLANX_TIMEOUT", 120))
        self.mictlanx_max_workers = int(os.environ.get("MICTLANX_MAX_WORKERS", "12"))
        self.mictlanx_bucket_id = os.environ.get("MICTLANX_BUCKET_ID", "rory")
        self.mictlanx_log_path = os.environ.get("MICTLANX_LOG_PATH", "/rory/mictlanx")
        self.mictlanx_log_interval = int(os.environ.get("MICTLANX_LOG_INTERVAL", "24"))
        self.mictlanx_log_when = os.environ.get("MICTLANX_LOG_WHEN", "h")
        self.mictlanx_delay = int(os.environ.get("MICTLANX_DELAY", "2"))
        self.mictlanx_backoff_factor = float(os.environ.get("MICTLANX_BACKOFF_FACTOR", "0.5"))
        self.mictlanx_max_retries = int(os.environ.get("MICTLANX_MAX_RETRIES", "10"))
        mictlanx_api_version = int(os.environ.get("MICTLANX_API_VERSION", "4"))
        mictlanx_protocol = os.environ.get("MICTLANX_PROTOCOL", "http")
        self.mictlanx_log_disable = bool(int(os.environ.get("MICTLANX_LOG_DISABLE", "1")))
        self.mictlanx_uri = os.environ.get(
            "MICTLANX_URI",
            f"mictlanx://mictlanx-router-0@localhost:63666?api_version={mictlanx_api_version}&protocol={mictlanx_protocol}",
        )

        self.rory_dataowner_log_interval = int(os.environ.get("RORY_DATAOWNER_LOG_INTERVAL", 24))
        self.rory_dataowner_log_when = os.environ.get("RORY_DATAOWNER_LOG_WHEN", "h")
        json_indent = os.environ.get("RORY_DATAOWNER_LOG_JSON_FORMATTER_INDENT", None)
        self.rory_dataowner_log_json_formatter_indent = int(json_indent) if json_indent is not None else None
        self.rory_dataowner_log_disabled = bool(int(os.environ.get("RORY_DATAOWNER_LOG_DISABLED", "0")))
        self.rory_dataowner_log_error_to_file = bool(int(os.environ.get("RORY_DATAOWNER_LOG_ERROR_TO_FILE", "1")))
        self.rory_dataowner_log_file_handler_level = int(
            os.environ.get("RORY_DATAOWNER_LOG_FILE_HANDLER_LEVEL", logging.INFO)
        )
        self.rory_dataowner_log_level = int(os.environ.get("RORY_DATAOWNER_LOG_LEVEL", logging.DEBUG))
        self.rory_dataowner_log_to_file = bool(int(os.environ.get("RORY_DATAOWNER_LOG_TO_FILE", "1")))
        self.rory_dataowner_log_use_rich = bool(int(os.environ.get("RORY_DATAOWNER_LOG_USE_RICH", "0")))
