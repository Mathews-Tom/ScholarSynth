"""
Configuration loading utilities for ScholarSynth.

This module provides functions for loading configuration from environment variables
and .env files.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from dotenv import load_dotenv

# Define logger for this module
logger = logging.getLogger(__name__)


def load_config() -> SimpleNamespace:
    """
    Load configuration from environment variables and .env file.

    Returns:
        SimpleNamespace: Configuration object with attributes for each config value.

    Raises:
        ValueError: If required configuration values are missing.
    """
    # Determine the project root directory
    project_root = Path(__file__).resolve().parent.parent
    
    # Load .env file if it exists
    env_path = project_root / ".env"
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logger.warning(f".env file not found at {env_path}. Using environment variables only.")
    
    # Create a SimpleNamespace to hold configuration
    config = SimpleNamespace()
    
    # --- Required Configuration ---
    # OpenAI API Key
    config.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not config.OPENAI_API_KEY:
        error_msg = "OPENAI_API_KEY is required but not set in environment variables or .env file."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # --- Optional Configuration with Defaults ---
    # Embedding Model
    config.EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    logger.info(f"Using embedding model: {config.EMBEDDING_MODEL_NAME}")
    
    # LLM Model
    config.LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-3.5-turbo")
    logger.info(f"Using LLM model: {config.LLM_MODEL_NAME}")
    
    # Vector Store Path
    config.CHROMA_PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "data/vector_store_db")
    logger.info(f"Using vector store path: {config.CHROMA_PERSIST_PATH}")
    
    # Parent Document Store Path
    config.PARENT_DOC_STORE_PATH = os.environ.get("PARENT_DOC_STORE_PATH", "data/parent_doc_store")
    logger.info(f"Using parent document store path: {config.PARENT_DOC_STORE_PATH}")
    
    # Vector Store Batch Size
    try:
        config.VECTOR_STORE_BATCH_SIZE = int(os.environ.get("VECTOR_STORE_BATCH_SIZE", "128"))
    except ValueError:
        logger.warning("Invalid VECTOR_STORE_BATCH_SIZE value. Using default: 128")
        config.VECTOR_STORE_BATCH_SIZE = 128
    
    # Parent Document Retriever Configuration
    try:
        config.PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE", "2000"))
    except ValueError:
        logger.warning("Invalid PARENT_CHUNK_SIZE value. Using default: 2000")
        config.PARENT_CHUNK_SIZE = 2000
    
    try:
        config.PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP", "400"))
    except ValueError:
        logger.warning("Invalid PARENT_CHUNK_OVERLAP value. Using default: 400")
        config.PARENT_CHUNK_OVERLAP = 400
    
    try:
        config.CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE", "500"))
    except ValueError:
        logger.warning("Invalid CHILD_CHUNK_SIZE value. Using default: 500")
        config.CHILD_CHUNK_SIZE = 500
    
    try:
        config.CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP", "100"))
    except ValueError:
        logger.warning("Invalid CHILD_CHUNK_OVERLAP value. Using default: 100")
        config.CHILD_CHUNK_OVERLAP = 100
    
    # PDF Source Directory
    config.PDF_SOURCE_DIR = os.environ.get("PDF_SOURCE_DIR", "data/local_pdfs")
    logger.info(f"Using PDF source directory: {config.PDF_SOURCE_DIR}")
    
    # ArXiv Configuration
    config.ARXIV_QUERY = os.environ.get("ARXIV_QUERY", None)
    if config.ARXIV_QUERY:
        logger.info(f"Using ArXiv query: {config.ARXIV_QUERY}")
    else:
        logger.info("No ArXiv query specified. ArXiv document loading will be skipped.")
    
    try:
        config.ARXIV_MAX_RESULTS = int(os.environ.get("ARXIV_MAX_RESULTS", "10"))
    except ValueError:
        logger.warning("Invalid ARXIV_MAX_RESULTS value. Using default: 10")
        config.ARXIV_MAX_RESULTS = 10
    
    return config
