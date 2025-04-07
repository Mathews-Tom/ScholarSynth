import logging
import os
from types import SimpleNamespace
from typing import Optional  # Import Optional for type hinting cache

from langchain_openai import OpenAIEmbeddings

from utils.config_loader import load_config  # Load centralized config

# Define logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
# Variables to hold loaded config values, with type hints
OPENAI_API_KEY: Optional[str] = None
EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"  # Provide a default type hint

try:
    config: SimpleNamespace = load_config()
    OPENAI_API_KEY = config.OPENAI_API_KEY
    EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME
    # Validate API key immediately after loading
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY configuration is missing.")
except (ValueError, AttributeError, Exception) as e:
    logger.exception(f"Configuration error during embedding setup: {e}", exc_info=True)
    # API key is essential for this module to function
    raise RuntimeError("Failed to load necessary configuration for embeddings.") from e

# --- Cache ---
# Cache the embedding model instance, type hint indicates it can be None or the specific type
_embedding_model_cache: Optional[OpenAIEmbeddings] = None


# --- Embedding Model Function ---
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initializes and returns the configured OpenAI embedding model using LangChain.

    Uses a cached instance to avoid repeated initialization.

    Returns:
        OpenAIEmbeddings: An instance of the LangChain embedding model.

    Raises:
        RuntimeError: If the embedding model fails to initialize.
    """
    global _embedding_model_cache

    # Use the already loaded and validated API key
    # No need to check for OPENAI_API_KEY here again as it's checked during module load

    if _embedding_model_cache is not None:
        # logger.debug("Returning cached embedding model instance.")
        return _embedding_model_cache

    try:
        config: SimpleNamespace = load_config()  # Load config HERE
        api_key = config.OPENAI_API_KEY
        model_name = config.EMBEDDING_MODEL_NAME

        # --- Explicitly check and pass the API key ---
        if not api_key:
            error_msg = "OPENAI_API_KEY not found in loaded configuration. Check .env file and config_loader.py."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Initializing OpenAIEmbeddings model: {model_name}")
        # Log only the last few chars for security confirmation
        logger.debug(f"Using API key ending in '...{api_key[-4:]}'")

        embedding_function = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key,  # Explicitly pass the key
            # Można dodać inne opcje, np. chunk_size
        )
        # --- End explicit passing ---

        logger.info("OpenAIEmbeddings model initialized successfully.")
        _embedding_model_cache = embedding_function
        return embedding_function

    except (ValueError, AttributeError, Exception) as e:
        logger.exception("Failed to initialize OpenAI embedding model.", exc_info=True)
        # Re-raise as RuntimeError to signal critical failure
        raise RuntimeError("Embedding model initialization failed.") from e


if __name__ == "__main__":
    # Need to configure logging before using the logger implicitly created above
    from utils.logging_config import setup_logging

    setup_logging()  # Setup logging first
    logger.info("--- Testing embedding_config.py ---")  # Use the module logger

    try:
        model = get_embedding_model()
        logger.info(f"Successfully got embedding model: {type(model)}")

        # Example embedding call (costs $$)
        # vector = model.embed_query("This is a test sentence.")
        # logger.info(f"Embedding vector length: {len(vector)}") # Requires 'vector' defined
        # logger.info("Successfully tested embedding a query.")

        # Test getting it again to check caching
        model_2 = get_embedding_model()
        logger.info(
            f"Got embedding model again: {type(model_2)}. Cache worked: {model is model_2}"
        )

    except RuntimeError as e:
        logger.error(f"Runtime Error during embedding model test: {e}", exc_info=True)
    except Exception as e:
        logger.error(
            f"Unexpected Error during embedding model test: {e}", exc_info=True
        )
    logger.info("--- Finished testing embedding_config.py ---")
