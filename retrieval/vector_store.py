"""
Vector store utilities for ScholarSynth.

This module provides functions for initializing and managing vector stores
used for document retrieval.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from retrieval.embedding_config import get_embedding_model

# Define logger for this module
logger = logging.getLogger(__name__)


def get_vector_store(
    persist_directory_str: str,
    embedding_function: Optional[Embeddings] = None,
    collection_name: str = "documents",
) -> Chroma:
    """
    Initialize and return a Chroma vector store.

    Args:
        persist_directory_str (str): Directory path where the vector store is persisted.
        embedding_function (Optional[Embeddings]): Embedding function to use.
            If None, a default embedding model is initialized.
        collection_name (str): Name of the collection in the vector store.

    Returns:
        Chroma: Initialized Chroma vector store.

    Raises:
        ValueError: If the vector store cannot be initialized.
    """
    # Get embedding function if not provided
    if embedding_function is None:
        logger.info("No embedding function provided, initializing default model.")
        embedding_function = get_embedding_model()

    # Convert string path to Path object and resolve
    persist_directory = Path(persist_directory_str).resolve()

    # Create directory if it doesn't exist
    persist_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing Chroma vector store at: {persist_directory}")

    try:
        # Initialize Chroma vector store
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
            collection_name=collection_name,
        )

        # Log the number of items in the vector store
        count = vector_store._collection.count()
        logger.info(f"Vector store initialized with {count} items.")

        return vector_store
    except Exception as e:
        error_msg = f"Failed to initialize vector store at '{persist_directory}': {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e


def add_documents_to_vector_store(
    vector_store: Chroma,
    documents: List[Document],
    batch_size: int = 128,
) -> None:
    """
    Add documents to a vector store in batches.

    Args:
        vector_store (Chroma): The vector store to add documents to.
        documents (List[Document]): List of documents to add.
        batch_size (int): Number of documents to add in each batch.

    Raises:
        ValueError: If the documents cannot be added to the vector store.
    """
    if not documents:
        logger.warning("No documents provided to add to vector store.")
        return

    logger.info(
        f"Adding {len(documents)} documents to vector store in batches of {batch_size}."
    )

    try:
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(batch)
            logger.info(
                f"Added batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size} ({len(batch)} documents)."
            )

        # Log the final count
        count = vector_store._collection.count()
        logger.info(
            f"Successfully added all documents. Vector store now contains {count} items."
        )
    except Exception as e:
        error_msg = f"Failed to add documents to vector store: {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e
