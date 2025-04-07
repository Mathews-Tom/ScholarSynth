import argparse  # For command-line arguments
import logging
import shutil  # For removing directories (rmtree)
import sys
from pathlib import Path  # For path operations
from types import SimpleNamespace  # For accessing config like attributes
from typing import List, Optional

from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import Chroma

# LangChain types
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# --- Add project root using pathlib ---
# This file is in scripts/, so root is one level up
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:  # Compare string paths
    sys.path.insert(0, str(project_root))  # Add string representation
# --- End path modification ---

# Project-specific modules
from retrieval.document_loader import load_arxiv_papers, load_pdfs
from retrieval.embedding_config import get_embedding_model
from retrieval.text_splitter import get_text_splitter
from retrieval.vector_store import (
    add_documents_to_store,
    generate_chunk_id,
    get_vector_store,
)
from utils.config_loader import load_config
from utils.logging_config import setup_logging

# Setup logging BEFORE anything else that might log
setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to run the data ingestion pipeline.
    Handles command-line arguments for clearing the store, loads data,
    splits it, generates IDs, initializes the vector store, and
    adds/updates documents idempotently in batches.
    """
    logger.info("--- Starting Data Ingestion Pipeline ---")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Ingest documents (PDFs, ArXiv) into the Chroma vector store."
    )
    parser.add_argument(
        "--clear",
        action="store_true",  # Makes it a boolean flag, stores True if present
        help="Delete the existing vector store directory specified in config before starting ingestion.",
    )
    args = parser.parse_args()
    logger.info(f"Command line arguments parsed. Clear store requested: {args.clear}")
    # --- End Argument Parsing ---

    # 1. Load Configuration
    # Load config *after* parsing args, as config is needed for the clear path
    try:
        config: SimpleNamespace = load_config()
        logger.info("Configuration loaded successfully.")
        logger.info(f"Vector store path from config: {config.CHROMA_PERSIST_PATH}")
        logger.info(f"Using vector store batch size: {config.VECTOR_STORE_BATCH_SIZE}")
        logger.debug(f"PDF source directory setting: {config.PDF_SOURCE_DIR}")
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)  # Exit if config fails

    # --- Handle --clear Action (if requested) ---
    if args.clear:
        persist_path_str = config.CHROMA_PERSIST_PATH
        if persist_path_str:
            # Construct the absolute path using pathlib relative to project root
            persist_path = (project_root / persist_path_str).resolve()
            logger.warning(
                f"--clear flag specified. Attempting to remove vector store directory: {persist_path}"
            )

            if persist_path.is_dir():  # Check if it's actually a directory
                try:
                    # Use shutil.rmtree for recursive deletion
                    shutil.rmtree(str(persist_path))  # rmtree requires a string path
                    logger.info(
                        f"Successfully removed existing vector store directory: {persist_path}"
                    )
                except OSError as e:
                    logger.error(
                        f"Error removing directory {persist_path}: {e}", exc_info=True
                    )
                    # Exit if clearing failed, as continuing might lead to unexpected state
                    sys.exit("Failed to clear vector store directory. Exiting.")
                except Exception as e:
                    logger.error(
                        f"Unexpected error during directory removal: {e}", exc_info=True
                    )
                    sys.exit("Unexpected error during vector store clearing. Exiting.")
            elif persist_path.exists():
                # Path exists but is not a directory
                logger.error(
                    f"Path specified for vector store exists but is not a directory: {persist_path}. Cannot clear."
                )
                sys.exit("Vector store path is not a directory. Cannot clear. Exiting.")
            else:
                # Directory does not exist, nothing to clear
                logger.info(
                    f"Vector store directory not found at {persist_path}. No need to clear."
                )
        else:
            # Should be caught by config loader ideally, but double check
            logger.error(
                "CHROMA_PERSIST_PATH not configured. Cannot perform --clear operation."
            )
            sys.exit("CHROMA_PERSIST_PATH not configured. Exiting.")
    # --- End Handle --clear Action ---

    # 2. Load Documents from sources
    all_docs: List[Document] = []
    try:
        logger.info("Loading documents from PDF sources...")
        pdf_source_path: Optional[Path] = None
        if config.PDF_SOURCE_DIR:
            pdf_source_path = (project_root / config.PDF_SOURCE_DIR).resolve()
            if not pdf_source_path.is_dir():
                logger.warning(
                    f"PDF source directory '{pdf_source_path}' not found or invalid. Skipping PDF load."
                )
                pdf_source_path = None
            else:
                logger.info(f"Reading PDFs from: {pdf_source_path}")
        else:
            logger.info("No PDF_SOURCE_DIR configured. Skipping local PDF loading.")

        pdf_docs: List[Document] = load_pdfs(pdf_source_path)
        logger.info(f"Loaded {len(pdf_docs)} pages from local PDFs.")
        all_docs.extend(pdf_docs)

        logger.info("Loading documents from ArXiv...")
        if config.ARXIV_QUERY:
            arxiv_docs: List[Document] = load_arxiv_papers(
                config.ARXIV_QUERY,
                config.ARXIV_MAX_RESULTS,
                download_dir_name="data/arxiv_pdfs",
            )
            logger.info(f"Loaded {len(arxiv_docs)} pages from ArXiv papers.")
            all_docs.extend(arxiv_docs)
        else:
            logger.info("ARXIV_QUERY not configured. Skipping ArXiv loading.")

        if not all_docs:
            logger.warning(
                "No documents were loaded from any source. Ingestion process stopping."
            )
            sys.exit(0)

        logger.info(f"Total source document pages loaded: {len(all_docs)}")

    except Exception as e:
        logger.error(f"Error during document loading: {e}", exc_info=True)
        sys.exit(1)

    # 3. Split Documents into Chunks and Generate IDs
    chunks: List[Document] = []
    chunk_ids: List[str] = []
    try:
        logger.info("Initializing text splitter...")
        text_splitter: TextSplitter = get_text_splitter()
        logger.info("Splitting loaded documents into chunks...")
        chunks = text_splitter.split_documents(all_docs)

        if not chunks:
            logger.error("Splitting resulted in zero chunks. Cannot proceed.")
            sys.exit(1)
        logger.info(
            f"Split {len(all_docs)} source document pages into {len(chunks)} chunks."
        )

        logger.info("Generating deterministic IDs for document chunks...")
        chunk_ids = [generate_chunk_id(chunk) for chunk in chunks]
        logger.info(f"Generated {len(chunk_ids)} unique IDs.")
        if len(set(chunk_ids)) != len(chunk_ids):
            duplicate_count = len(chunk_ids) - len(set(chunk_ids))
            logger.warning(
                f"Duplicate chunk IDs generated ({duplicate_count}). This indicates identical chunks."
            )

        if chunks:
            content_preview = chunks[0].page_content[:150].replace("\n", " ") + "..."
            logger.debug(f"Sample chunk 0 preview: '{content_preview}'")
            logger.debug(f"Sample chunk 0 metadata: {chunks[0].metadata}")
            logger.debug(f"Sample chunk 0 ID: {chunk_ids[0]}")

    except Exception as e:
        logger.error(
            f"Error during text splitting or ID generation: {e}", exc_info=True
        )
        sys.exit(1)

    # 4. Initialize Embedding Model and Vector Store
    existing_items_before = 0  # Initialize a variable to store count before adding
    try:
        logger.info("Initializing embedding model...")
        embedding_model: Embeddings = get_embedding_model()
        logger.info(f"Using embedding model: {type(embedding_model).__name__}")

        logger.info("Initializing vector store...")
        # Note: get_vector_store handles directory creation. If cleared (--clear), it will be recreated here.
        vector_store: Chroma = get_vector_store(
            persist_directory_str=config.CHROMA_PERSIST_PATH,
            embedding_function=embedding_model,
        )

        # Log current state after initialization (could be 0 if cleared, or existing count if not)
        existing_items_before = vector_store._collection.count()
        if args.clear:
            logger.info(
                f"Vector store initialized fresh (after clearing). Contains {existing_items_before} items."
            )
        else:
            logger.info(
                f"Vector store initialized (or loaded existing). Contains {existing_items_before} items."
            )

    except Exception as e:
        logger.error(
            f"Error initializing embedding model or vector store: {e}", exc_info=True
        )
        sys.exit(1)

    # 5. Add Chunks to Vector Store (Idempotently, Batched)
    try:
        logger.info(
            f"Adding/updating {len(chunks)} chunks in the vector store "
            f"using generated IDs (batch size: {config.VECTOR_STORE_BATCH_SIZE})..."
        )
        processed_ids: List[str] = add_documents_to_store(
            documents=chunks,
            vector_store=vector_store,
            generated_ids=chunk_ids,
            batch_size=config.VECTOR_STORE_BATCH_SIZE,
        )
        logger.info(
            f"Chroma reported processing {len(processed_ids)} items for addition/update."
        )

        # Verify final count in the store
        final_items = vector_store._collection.count()
        logger.info(f"Vector store now contains {final_items} items.")
        # Log the difference unless it was cleared
        if not args.clear:
            newly_added_count = final_items - existing_items_before
            logger.info(
                f"(An increase of {newly_added_count} items from pre-run count of {existing_items_before})."
            )

    except Exception as e:
        logger.error(
            f"Error adding/updating documents in vector store: {e}", exc_info=True
        )
        sys.exit(1)

    logger.info("--- Data Ingestion Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
