# logging_config.py
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import cfg

def get_logger(name: str = None) -> logging.Logger:
    # If name=None, auto-detect caller's module name
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # handlers already added

    logs_dir: Path = cfg.paths.logs_dir
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=10_000_000, backupCount=5)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    error_handler = RotatingFileHandler(logs_dir / "errors.log", maxBytes=10_000_000, backupCount=5)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    return logger