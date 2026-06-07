# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import cfg


def get_logger(name: str) -> logging.Logger:
    # If name=None, auto-detect caller's module name
    if name is None:
        import inspect

        current_frame = inspect.currentframe()

        if current_frame and current_frame.f_back:
            name = current_frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "unknown"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # handlers already added

    logs_dir: Path = cfg.paths.logs_dir
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )

    handler = RotatingFileHandler(
        logs_dir / "app.log", maxBytes=10_000_000, backupCount=5
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    error_handler = RotatingFileHandler(
        logs_dir / "errors.log", maxBytes=10_000_000, backupCount=5
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    # console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger
