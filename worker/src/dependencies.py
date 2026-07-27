from functools import lru_cache

from config import Settings
from mictlanx import AsyncClient
from mictlanx.logger.log import Log, JsonFormatter
from rory.core.security.cryptosystem.pqc.ckks import Ckks


@lru_cache()
def get_settings() -> Settings:
    return Settings()


_settings = get_settings()

LOGGER = Log(
    name=_settings.node_id,
    path=_settings.log_path,
    interval=_settings.rory_worker_log_interval,
    when=_settings.rory_worker_log_when,
    console_formatter=JsonFormatter(indent=int(_settings.rory_worker_log_json_formatter_indent) if _settings.rory_worker_log_json_formatter_indent is not None else None),
    console_handler_level=_settings.rory_worker_log_level,
    disabled=_settings.rory_worker_log_disabled,
    error_log=_settings.rory_worker_log_error_to_file,
    file_handler_level=_settings.rory_worker_log_file_handler_level,
    log_level=_settings.rory_worker_log_level,
    to_file=_settings.rory_worker_log_to_file,
    use_rich=_settings.rory_worker_log_use_rich,
)

ASYNC_STORAGE_CLIENT = AsyncClient(
    client_id=_settings.mictlanx_client_id,
    uri=_settings.mictlanx_uri,
    capacity_storage="200mb",
    debug=False,
    eviction_policy="LRU",
    max_workers=_settings.mictlanx_max_workers,
    verify=False,
    log_output_path=_settings.mictlanx_log_path,
    log_interval=_settings.mictlanx_log_interval,
    log_when=_settings.mictlanx_log_when,
    enable_logging=not _settings.mictlanx_log_disable,
)

CKKS = Ckks.from_pyfhel_server(
    _round=_settings.ckks_round,
    decimals=_settings.ckks_decimals,
    path=_settings.keys_path,
    ctx_filename=_settings.ctx_filename,
    pubkey_filename=_settings.pubkey_filename,
    relinkey_filename=_settings.relinkey_filename,
    rotatekey_filename=_settings.rotatekey_filename,
)


def get_logger():
    return LOGGER


def get_storage_client():
    return ASYNC_STORAGE_CLIENT


def get_ckks():
    return CKKS
