import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional

# Define logger for this module itself (optional, mainly for setup logging)
# logger = logging.getLogger(__name__) # Avoid using logger before setup

# Flag to ensure setup runs only once
_logging_configured: bool = False


def setup_logging(
    log_level_str: Optional[str] = None,
    log_to_file: bool = True,
    log_dir_name: str = "logs",  # Relative directory name
    log_filename: str = "app.log",
) -> None:
    """
    Configures application-wide logging.

    Args:
        log_level_str (Optional[str]): Logging level (e.g., "INFO", "DEBUG").
                                        Defaults to "INFO" or env var LOG_LEVEL.
        log_to_file (bool): Whether to log to a file in addition to the console.
        log_dir_name (str): The name of the directory to store log files.
        log_filename (str): The name of the log file.
    """
    global _logging_configured
    if _logging_configured:
        # logging.getLogger(__name__).debug("Logging already configured.")
        return

    # Determine log level
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    effective_log_level_str = log_level_str.upper() if log_level_str else log_level_env
    log_level = getattr(logging, effective_log_level_str, logging.INFO)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    }

    loggers = {
        # Root logger configuration
        "": {
            "handlers": ["console"],
            "level": log_level,
            "propagate": False,  # Prevent root logger messages going potentially to default handlers
        },
        # Configuration for specific libraries (optional, can silence noisy libs)
        "httpx": {  # Example: httpx used by openai/arxiv can be verbose
            "handlers": ["console"],
            "level": logging.WARNING,  # Set higher level to reduce noise
            "propagate": False,
        },
        "chromadb": {  # ChromaDB telemetry logging
            "handlers": ["console"],
            "level": logging.WARNING,  # Change to INFO to see telemetry messages
            "propagate": False,
        },
        # Add other library-specific configurations here if needed
    }

    # --- Configure file handler using pathlib ---
    if log_to_file:
        try:
            # Assume logs directory is relative to project root (where app.py or scripts run from)
            # For robustness, determine project root relative to *this* file
            project_root = Path(__file__).resolve().parent.parent
            log_dir: Path = project_root / log_dir_name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path: Path = log_dir / log_filename

            handlers["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "standard",
                "filename": str(log_file_path),  # Handler expects string path
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            }
            # Add file handler to root logger AND specific loggers if desired
            loggers[""]["handlers"].append("file")
            # Example: Add file handler also to httpx logger if needed
            # loggers["httpx"]["handlers"].append("file")

        except Exception as e:
            # Use basic print as logging might not be functional yet
            print(
                f"Error configuring file logging: {e}. Logging to console only.",
                file=sys.stderr,
            )
            log_to_file = False  # Fallback to console only
    # --- End file handler configuration ---

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,  # Keep existing loggers (e.g., from libraries)
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": date_format,
            },
        },
        "handlers": handlers,
        "loggers": loggers,
    }

    try:
        logging.config.dictConfig(logging_config)
        _logging_configured = True
        # Use logging now that it's configured
        logging.getLogger(__name__).info(
            f"Logging configured successfully with level {effective_log_level_str}. File logging: {log_to_file}"
        )
    except Exception as e:
        # Fallback basic config if dictConfig fails
        logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)
        logging.getLogger(__name__).error(
            f"Failed to apply dictionary logging config: {e}. Using basicConfig."
        )
        _logging_configured = True  # Mark as configured even if basic


if __name__ == "__main__":
    print("Testing logging setup...")
    setup_logging(log_level_str="DEBUG", log_to_file=True)
    # Test loggers
    root_logger = logging.getLogger()
    module_logger = logging.getLogger("my_module")
    httpx_logger = logging.getLogger("httpx")

    root_logger.debug(
        "This is a root debug message."
    )  # Should appear if level is DEBUG
    root_logger.info("This is a root info message.")
    module_logger.info("This is a message from my_module.")
    module_logger.warning("This is a warning from my_module.")
    # httpx logs should only show WARNING level or higher by default from setup
    httpx_logger.info("This is an INFO message from httpx (should be hidden).")
    httpx_logger.warning("This is a WARNING message from httpx (should be visible).")
    print(f"Check the '{Path('logs') / 'app.log'}' file.")
