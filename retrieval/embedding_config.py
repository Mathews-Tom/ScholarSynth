"""
Embedding model configuration for ScholarSynth.

This module provides functions for initializing and configuring embedding models
used for document vectorization and retrieval.
"""

import logging
import os
import random
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from utils.config_loader import load_config

# Define logger for this module
logger = logging.getLogger(__name__)


class MockEmbeddings(Embeddings):
    """Mock embedding model for testing purposes."""

    def __init__(self, size: int = 1536):
        """Initialize the mock embedding model.

        Args:
            size (int): Dimension of the embedding vectors.
        """
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for a list of documents.

        Args:
            texts (List[str]): List of document texts to embed.

        Returns:
            List[List[float]]: List of mock embedding vectors.
        """
        return [[random.uniform(-1, 1) for _ in range(self.size)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate a mock embedding for a query string.

        Args:
            text (str): Query text to embed.

        Returns:
            List[float]: Mock embedding vector.
        """
        return [random.uniform(-1, 1) for _ in range(self.size)]


def get_embedding_model(model_name: Optional[str] = None) -> Embeddings:
    """
    Initialize and return an embedding model.

    Args:
        model_name (Optional[str]): Name of the embedding model to use.
            If None, the model name is loaded from the configuration.

    Returns:
        Embeddings: Initialized embedding model.

    Raises:
        ValueError: If the model name is invalid or the API key is missing.
    """
    # Load configuration if model_name not provided
    if model_name is None:
        try:
            config = load_config()
            model_name = getattr(
                config, "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
            )
            logger.info(f"Using embedding model from config: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load config for embedding model: {e}")
            model_name = "text-embedding-3-small"
            logger.info(f"Falling back to default embedding model: {model_name}")

    # Check if we're in test mode (no API key)
    if (
        not os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY") == "YOUR_API_KEY_HERE"
    ):
        logger.warning(
            "No valid OpenAI API key found. Using mock embeddings for testing."
        )
        return MockEmbeddings()

    # Initialize the OpenAI embedding model
    try:
        logger.info(f"Initializing OpenAI embedding model: {model_name}")
        embedding_model = OpenAIEmbeddings(model=model_name)
        logger.info("Embedding model initialized successfully.")
        return embedding_model
    except Exception as e:
        error_msg = f"Failed to initialize embedding model '{model_name}': {e}"
        logger.error(error_msg, exc_info=True)
        logger.warning("Falling back to mock embeddings for testing.")
        return MockEmbeddings()
