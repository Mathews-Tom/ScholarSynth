"""
Logging configuration for ScholarSynth.

This module provides functions for setting up logging with appropriate
handlers and formatters.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    console_level: str = "INFO",
) -> logging.Logger:
    """
    Set up logging with file and console handlers.

    Args:
        log_level (str): Log level for the file handler.
        log_file (str): Path to the log file.
        console_level (str): Log level for the console handler.

    Returns:
        logging.Logger: Configured root logger.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(log_level))
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(levelname)s - %(message)s"
    )
    
    # Create file handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.getLevelName(log_level))
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}", file=sys.stderr)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.getLevelName(console_level))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. File: {log_file} (Level: {log_level}), Console level: {console_level}")
    
    return root_logger
