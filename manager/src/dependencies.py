from functools import lru_cache
from threading import Semaphore

from config import Settings
from mictlanx.services.summoner.summoner import Summoner
from mictlanx.logger.log import Log, JsonFormatter
from option import Some
from load_balancing.round_robin import RoundRobin
from load_balancing.two_choices import TwoChoices
from load_balancing.random import Random


@lru_cache()
def get_settings() -> Settings:
    return Settings()


_settings = get_settings()

LOGGER = Log(
    name=_settings.node_id,
    path=_settings.log_path,
    interval=_settings.rory_manager_log_interval,
    when=_settings.rory_manager_log_when,
    console_formatter=JsonFormatter(indent=int(_settings.rory_manager_log_json_formatter_indent) if _settings.rory_manager_log_json_formatter_indent is not None else None),
    console_handler_level=_settings.rory_manager_log_level,
    disabled=_settings.rory_manager_log_disabled,
    error_log=_settings.rory_manager_log_error_to_file,
    file_handler_level=_settings.rory_manager_log_file_handler_level,
    log_level=_settings.rory_manager_log_level,
    to_file=_settings.rory_manager_log_to_file,
    use_rich=_settings.rory_manager_log_use_rich,
)

REPLICATOR = Summoner(
    ip_addr=_settings.mictlanx_summoner_ip_addr,
    port=_settings.mictlanx_summoner_port,
    api_version=Some(_settings.mictlanx_api_version),
)

BALANCERS = [
    RoundRobin(n=_settings.init_workers, prefix=_settings.node_prefix),
    TwoChoices(n=_settings.init_workers, prefix=_settings.node_prefix),
    Random(n=_settings.init_workers, prefix=_settings.node_prefix),
]

LB = BALANCERS[_settings.load_balancing]

WORKERS: dict = {}

DEPLOY_START_TIMES: dict = {}


def get_logger():
    return LOGGER


def get_replicator():
    return REPLICATOR


def get_lb():
    return LB


def get_workers():
    return WORKERS
