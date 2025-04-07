import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from dotenv import load_dotenv

# Define logger for this module
logger = logging.getLogger(__name__)

# Cache the loaded configuration
_config_cache: Optional[SimpleNamespace] = None


def load_config() -> SimpleNamespace:
    """
    Loads configuration from environment variables and a .env file.

    Prioritizes environment variables over .env file variables.
    Validates required variables. Caches the result.

    Returns:
        SimpleNamespace: An object containing the configuration values.

    Raises:
        ValueError: If required configuration variables are missing or invalid.
        RuntimeError: If an unexpected critical error occurs during loading.
    """
    global _config_cache
    if _config_cache is not None:
        # logger.debug("Returning cached configuration.")
        return _config_cache

    try:  # Broad try block for the whole loading process
        # --- Load .env file using pathlib ---
        project_root = Path(__file__).resolve().parent.parent
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            logger.info(
                f".env file found at {dotenv_path}. Loading environment variables."
            )
            load_dotenv(dotenv_path=dotenv_path)
        else:
            logger.warning(
                f".env file not found at {dotenv_path}. Relying solely on environment variables."
            )

        # --- Define keys ---
        required_keys = ["OPENAI_API_KEY"]
        optional_keys = {
            "PDF_SOURCE_DIR": "data/local_pdfs",  # Note: Original default was raw_pdfs
            "CHROMA_PERSIST_PATH": "data/vector_store_db",
            "PARENT_DOC_STORE_PATH": "data/parent_doc_store",
            "LOG_LEVEL": "INFO",
            "EMBEDDING_MODEL_NAME": "text-embedding-3-small",
            "LLM_MODEL_NAME": "gpt-3.5-turbo",
            "ARXIV_QUERY": None,
            "ARXIV_MAX_RESULTS": "10",
            "VECTOR_STORE_BATCH_SIZE": "128",
        }

        config_dict = {}
        missing_keys = []

        # Load required keys
        for key in required_keys:
            value = os.getenv(key)
            # Also check for empty strings which might cause issues later
            if value is None or value.strip() == "":
                missing_keys.append(key)
            else:
                config_dict[key] = value

        # Load optional keys
        for key, default in optional_keys.items():
            value = os.getenv(key, default)
            config_dict[key] = value

        # --- Validate required keys ---
        if missing_keys:
            error_msg = f"Missing required configuration variable(s): {', '.join(missing_keys)}. Please check your .env file or environment variables."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # --- Type conversion and specific validation ---
        try:
            # Convert ARXIV_MAX_RESULTS
            if config_dict["ARXIV_MAX_RESULTS"] is not None:
                config_dict["ARXIV_MAX_RESULTS"] = int(config_dict["ARXIV_MAX_RESULTS"])
            else:
                config_dict["ARXIV_MAX_RESULTS"] = 10  # Default if None

            # Convert VECTOR_STORE_BATCH_SIZE
            if config_dict["VECTOR_STORE_BATCH_SIZE"] is not None:
                batch_size_val = int(config_dict["VECTOR_STORE_BATCH_SIZE"])
                if batch_size_val <= 0:
                    logger.warning(
                        "VECTOR_STORE_BATCH_SIZE must be positive. Using default 128."
                    )
                    config_dict["VECTOR_STORE_BATCH_SIZE"] = 128
                else:
                    config_dict["VECTOR_STORE_BATCH_SIZE"] = batch_size_val
            else:
                config_dict["VECTOR_STORE_BATCH_SIZE"] = 128  # Default if None

        except ValueError as e:
            error_key = (
                "ARXIV_MAX_RESULTS or VECTOR_STORE_BATCH_SIZE"  # Approximate key
            )
            logger.error(
                f"Invalid integer value configured for {error_key}. Check .env file. Error: {e}"
            )
            raise ValueError(
                f"Invalid integer configuration value for {error_key}."
            ) from e

        # --- Create SimpleNamespace, Cache, and Return ---
        config = SimpleNamespace(**config_dict)

        logger.info("Configuration loaded and validated successfully.")
        _config_cache = config  # Update the cache
        return config

    except ValueError as ve:  # Catch specific ValueErrors raised above
        # Logged already, just re-raise to signal failure
        raise ve
    except Exception as e:  # Catch any OTHER unexpected error during loading
        logger.critical(
            f"Unexpected critical error during configuration loading: {e}",
            exc_info=True,
        )
        # Raise a runtime error to prevent returning None implicitly
        raise RuntimeError(
            f"Failed to load configuration due to an unexpected error: {e}"
        ) from e


if __name__ == "__main__":
    # Setup basic logging FORMATTER for the test block output
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    # Use the logger defined in the module
    # logger = logging.getLogger(__name__) # Already defined at module level

    logger.info("--- Testing utils/config_loader.py ---")
    app_config_test: Optional[SimpleNamespace] = None  # Initialize to None

    try:
        # Call the function and store the result
        app_config_test = load_config()

        # Check if the result is None BEFORE accessing attributes
        if app_config_test is None:
            logger.error("load_config() returned None unexpectedly.")
        else:
            # Now it's safer to access attributes
            logger.info("load_config() executed successfully. Checking values...")
            logger.info(
                f"OpenAI Key Loaded: {hasattr(app_config_test, 'OPENAI_API_KEY') and bool(app_config_test.OPENAI_API_KEY)}"
            )
            logger.info(
                f"PDF Source Dir: {getattr(app_config_test, 'PDF_SOURCE_DIR', 'N/A')}"
            )
            logger.info(f"Log Level: {getattr(app_config_test, 'LOG_LEVEL', 'N/A')}")
            logger.info(
                f"Chroma Path: {getattr(app_config_test, 'CHROMA_PERSIST_PATH', 'N/A')}"
            )
            logger.info(
                f"Batch Size: {getattr(app_config_test, 'VECTOR_STORE_BATCH_SIZE', 'N/A')}"
            )
            logger.info(
                f"ArXiv Max Results: {getattr(app_config_test, 'ARXIV_MAX_RESULTS', 'N/A')}"
            )

            # Test caching (call again)
            # logger.info("Testing config caching...")
            # app_config_2 = load_config()
            # logger.info(f"Config retrieved from cache: {app_config_test is app_config_2}")

    except ValueError as e:
        # Specifically catch config validation errors
        logger.error(f"Configuration Error caught in test block: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the test
        logger.error(
            f"An unexpected error occurred during test: {e}", exc_info=True
        )  # Log traceback

    logger.info(
        "--- Finished testing utils/config_loader.py ---"
    )  # Corrected final message
