#!/usr/bin/env python3
"""
Data ingestion script for ScholarSynth.

This script:
1. Loads documents from local PDFs and ArXiv
2. Processes them using either standard chunking or the Parent Document Retriever approach
3. Stores documents in a vector store for retrieval

Usage:
    python scripts/ingest.py [--clear] [--use-parent-retriever]

Options:
    --clear                 Delete existing vector store before ingestion
    --use-parent-retriever  Use the Parent Document Retriever approach (retrieve smaller chunks
                           but provide larger parent context)
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

# --- Add project root using pathlib ---
# This file is in scripts/, so root is one level up
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:  # Compare string paths
    sys.path.insert(0, str(project_root))  # Add string representation
# --- End path modification ---

# Project-specific modules
from retrieval.document_loader import load_arxiv_papers, load_pdfs
from retrieval.embedding_config import get_embedding_model
from retrieval.text_splitter import (
    get_parent_child_splitters,
    get_text_splitter,
    split_documents,
)
from retrieval.vector_store import add_documents_to_vector_store, get_vector_store
from utils.config_loader import load_config
from utils.logging_config import setup_logging

# Conditionally import parent document retriever
try:
    from retrieval.parent_document_retriever import get_parent_document_retriever

    PARENT_RETRIEVER_AVAILABLE = True
except ImportError:
    PARENT_RETRIEVER_AVAILABLE = False

# Setup logging
logger = setup_logging()


def ingest_with_standard_approach(
    config: SimpleNamespace, documents: List, clear_existing: bool = False
) -> None:
    """
    Process documents using the standard chunking approach.

    Args:
        config: Configuration object
        documents: List of documents to process
        clear_existing: Whether to clear the existing vector store
    """
    logger.info("Using standard document chunking approach")

    # Get vector store path
    vector_store_path = config.CHROMA_PERSIST_PATH
    logger.info(f"Vector store path from config: {vector_store_path}")

    # Get batch size for vector store operations
    batch_size = getattr(config, "VECTOR_STORE_BATCH_SIZE", 128)
    logger.info(f"Using vector store batch size: {batch_size}")

    # Get chunk size and overlap
    chunk_size = getattr(config, "CHUNK_SIZE", 1000)
    chunk_overlap = getattr(config, "CHUNK_OVERLAP", 200)
    logger.info(f"Using chunk size: {chunk_size}, overlap: {chunk_overlap}")

    # Clear existing vector store if requested
    if clear_existing:
        vector_store_dir = Path(project_root) / vector_store_path
        if vector_store_dir.exists():
            logger.warning(
                f"--clear flag specified. Attempting to remove vector store directory: {vector_store_dir}"
            )
            try:
                shutil.rmtree(vector_store_dir)
                logger.info(
                    f"Successfully removed existing vector store directory: {vector_store_dir}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to remove vector store directory: {e}", exc_info=True
                )
                sys.exit(1)

    try:
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        embedding_model = get_embedding_model()
        logger.info(f"Using embedding model: {type(embedding_model).__name__}")

        # Initialize text splitter
        logger.info("Initializing text splitter...")
        text_splitter = get_text_splitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split documents
        logger.info(f"Splitting {len(documents)} documents...")
        split_docs = split_documents(documents, text_splitter)
        logger.info(f"Created {len(split_docs)} document chunks")

        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = get_vector_store(
            persist_directory_str=vector_store_path, embedding_function=embedding_model
        )

        # Add documents to vector store
        logger.info(f"Adding {len(split_docs)} document chunks to vector store...")
        add_documents_to_vector_store(
            vector_store=vector_store, documents=split_docs, batch_size=batch_size
        )

        # Get count from vector store
        count = vector_store._collection.count()
        logger.info(f"Vector store contains {count} document chunks")

    except Exception as e:
        logger.error(f"Error in standard ingestion process: {e}", exc_info=True)
        sys.exit(1)


def ingest_with_parent_retriever(
    config: SimpleNamespace, documents: List, clear_existing: bool = False
) -> None:
    """
    Process documents using the Parent Document Retriever approach.

    Args:
        config: Configuration object
        documents: List of documents to process
        clear_existing: Whether to clear the existing stores
    """
    if not PARENT_RETRIEVER_AVAILABLE:
        logger.error(
            "Parent Document Retriever is not available. Make sure retrieval/parent_document_retriever.py exists."
        )
        sys.exit(1)

    logger.info("Using Parent Document Retriever approach")

    # Get vector store path
    vector_store_path = config.CHROMA_PERSIST_PATH
    logger.info(f"Vector store path from config: {vector_store_path}")

    # Get parent document store path
    parent_doc_store_path = getattr(
        config, "PARENT_DOC_STORE_PATH", "data/parent_doc_store"
    )
    logger.info(f"Parent document store path: {parent_doc_store_path}")

    # Get chunk sizes
    parent_chunk_size = getattr(config, "PARENT_CHUNK_SIZE", 2000)
    parent_chunk_overlap = getattr(config, "PARENT_CHUNK_OVERLAP", 400)
    child_chunk_size = getattr(config, "CHILD_CHUNK_SIZE", 500)
    child_chunk_overlap = getattr(config, "CHILD_CHUNK_OVERLAP", 100)
    logger.info(
        f"Using parent chunks: size={parent_chunk_size}, overlap={parent_chunk_overlap}"
    )
    logger.info(
        f"Using child chunks: size={child_chunk_size}, overlap={child_chunk_overlap}"
    )

    # Clear existing stores if requested
    if clear_existing:
        # Clear vector store
        vector_store_dir = Path(project_root) / vector_store_path
        if vector_store_dir.exists():
            logger.warning(
                f"--clear flag specified. Attempting to remove vector store directory: {vector_store_dir}"
            )
            try:
                shutil.rmtree(vector_store_dir)
                logger.info(
                    f"Successfully removed existing vector store directory: {vector_store_dir}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to remove vector store directory: {e}", exc_info=True
                )
                sys.exit(1)

        # Clear parent document store
        parent_doc_dir = Path(project_root) / parent_doc_store_path
        if parent_doc_dir.exists():
            logger.warning(
                f"--clear flag specified. Attempting to remove parent document store directory: {parent_doc_dir}"
            )
            try:
                shutil.rmtree(parent_doc_dir)
                logger.info(
                    f"Successfully removed existing parent document store directory: {parent_doc_dir}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to remove parent document store directory: {e}",
                    exc_info=True,
                )
                sys.exit(1)

    try:
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        embedding_model = get_embedding_model()
        logger.info(f"Using embedding model: {type(embedding_model).__name__}")

        # Create parent document store directory if it doesn't exist
        parent_doc_dir = Path(project_root) / parent_doc_store_path
        parent_doc_dir.mkdir(parents=True, exist_ok=True)

        # Initialize parent document retriever
        logger.info("Initializing parent document retriever...")
        retriever = get_parent_document_retriever(
            vector_store_path=vector_store_path,
            parent_doc_store_path=parent_doc_store_path,
            embedding_model=embedding_model,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
        )
        logger.info("Parent document retriever initialized successfully.")

        # Process documents
        logger.info(
            f"Processing {len(documents)} documents with parent document retriever..."
        )
        doc_ids = retriever.add_documents(documents)
        logger.info(f"Successfully processed {len(doc_ids)} parent documents.")

        # Get counts from stores
        child_count = retriever.vectorstore._collection.count()
        parent_count = retriever.get_parent_document_count()

        logger.info(f"Vector store contains {child_count} child chunks.")
        logger.info(f"Document store contains {parent_count} parent documents.")

    except Exception as e:
        logger.error(
            f"Error in parent document retriever ingestion process: {e}", exc_info=True
        )
        sys.exit(1)


def main() -> None:
    """
    Main function to run the data ingestion pipeline.
    Handles command-line arguments, loads data, and processes it using
    either standard chunking or the Parent Document Retriever approach.
    """
    logger.info("--- Starting ScholarSynth Ingestion Pipeline ---")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Ingest documents (PDFs, ArXiv) into ScholarSynth."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the existing vector store before starting ingestion.",
    )
    parser.add_argument(
        "--use-parent-retriever",
        action="store_true",
        help="Use the Parent Document Retriever approach (retrieve smaller chunks but provide larger parent context).",
    )
    args = parser.parse_args()
    logger.info(
        f"Command line arguments parsed. Clear store: {args.clear}, Use parent retriever: {args.use_parent_retriever}"
    )

    # --- Load Configuration ---
    try:
        config: SimpleNamespace = load_config()
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Documents ---
    # 1. Load PDFs
    logger.info("Loading documents from PDF sources...")
    pdf_dir = getattr(config, "PDF_SOURCE_DIR", "data/local_pdfs")
    logger.info(f"Reading PDFs from: {pdf_dir}")
    pdf_docs = load_pdfs(pdf_dir)
    logger.info(f"Loaded {len(pdf_docs)} pages from local PDFs.")

    # 2. Load ArXiv papers
    logger.info("Loading documents from ArXiv...")
    arxiv_query = getattr(config, "ARXIV_QUERY", None)
    arxiv_max_results = getattr(config, "ARXIV_MAX_RESULTS", 10)
    arxiv_docs = load_arxiv_papers(
        query=arxiv_query,
        max_results=arxiv_max_results,
    )
    logger.info(f"Loaded {len(arxiv_docs)} pages from ArXiv papers.")

    # Combine all documents
    all_docs = pdf_docs + arxiv_docs
    logger.info(f"Total source document pages loaded: {len(all_docs)}")

    # --- Process Documents ---
    if args.use_parent_retriever:
        ingest_with_parent_retriever(config, all_docs, args.clear)
    else:
        ingest_with_standard_approach(config, all_docs, args.clear)

    logger.info("--- ScholarSynth Ingestion Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
