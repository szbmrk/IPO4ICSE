import logging
from config_loader import config
import copy

LOG_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
LOGGER_COLOR = "\033[35m"
RESET_COLOR = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record_copy = copy.copy(record)
        level_color = LOG_COLORS.get(record_copy.levelname, "")
        record_copy.levelname = f"{level_color}{record_copy.levelname}{RESET_COLOR}"
        record_copy.name = f"{LOGGER_COLOR}{record_copy.name}{RESET_COLOR}"
        return super().format(record_copy)


console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
console_handler.setFormatter(
    ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt=config.LOG_DATEFMT,
    )
)

file_handler = None
if config.LOG_FILE:
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt=config.LOG_DATEFMT,
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
root_logger.addHandler(console_handler)
if file_handler:
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
