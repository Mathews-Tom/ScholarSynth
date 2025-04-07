import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

# Vector store library
from langchain_community.vectorstores import Chroma

# LangChain components
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm.auto import tqdm

# Internal imports
from retrieval.embedding_config import get_embedding_model
from utils.config_loader import load_config

# Util for filtering metadata if needed (though we handle conversion now)
# from langchain_community.vectorstores.utils import filter_complex_metadata


# Define logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
CHROMA_PERSIST_PATH_STR: Optional[str] = None

try:
    config: SimpleNamespace = load_config()
    CHROMA_PERSIST_PATH_STR = config.CHROMA_PERSIST_PATH
    if not CHROMA_PERSIST_PATH_STR:
        raise ValueError("CHROMA_PERSIST_PATH configuration is missing.")
except (ValueError, AttributeError, Exception) as e:
    logger.exception("Configuration error during vector store setup.", exc_info=True)
    raise RuntimeError(
        "Failed to load necessary configuration for vector store."
    ) from e

# --- Caching ---
_vector_store_cache: Optional[Chroma] = None
_embedding_function_cache: Optional[Embeddings] = None
_persist_directory_cache: Optional[Path] = None  # Cache Path object


# --- Generate Chunk IDs ---
def generate_chunk_id(document: Document) -> str:
    """
    Generates a deterministic ID (SHA-256 hash) for a document chunk
    based on its page content and source metadata.

    Args:
        document (Document): The LangChain Document object (chunk).

    Returns:
        str: A unique hexadecimal SHA-256 hash string representing the chunk.
    """
    source_id = document.metadata.get("source", "unknown_source")
    page_content = document.page_content
    # Create a string that combines identifying information
    # Using a separator to avoid collisions between source and content
    identifier_string = f"source:{source_id}::content:{page_content}"
    # Encode the string to bytes
    identifier_bytes = identifier_string.encode("utf-8")
    # Create SHA-256 hash
    sha256_hash = hashlib.sha256(identifier_bytes)
    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


# --- Vector Store Initialization ---
def get_vector_store(
    persist_directory_str: Optional[str] = None,
    embedding_function: Optional[Embeddings] = None,
    force_recreate: bool = False,
) -> Chroma:
    """
    Initializes/loads the ChromaDB vector store using LangChain and pathlib.

    Args:
        persist_directory_str (Optional[str]): Path string to store/load ChromaDB data.
                                                If None, uses config.CHROMA_PERSIST_PATH.
        embedding_function (Optional[Embeddings]): LangChain embedding function instance.
                                                    If None, initializes via get_embedding_model().
        force_recreate (bool): If True, bypasses cache and forces re-initialization.

    Returns:
        Chroma: Initialized LangChain Chroma vector store instance.

    Raises:
        RuntimeError: If vector store initialization fails.
        ValueError: If persist directory config is missing.
    """
    global _vector_store_cache, _embedding_function_cache, _persist_directory_cache

    # Determine effective path string
    effective_persist_str = (
        persist_directory_str if persist_directory_str else CHROMA_PERSIST_PATH_STR
    )
    if not effective_persist_str:
        raise ValueError("Chroma persist directory is not configured.")

    # --- Convert to absolute Path object ---
    try:
        persist_path_obj = Path(effective_persist_str).resolve()
    except Exception as e:  # Catch potential errors during path resolution
        logger.error(
            f"Failed to resolve persist directory path '{effective_persist_str}': {e}"
        )
        raise ValueError("Invalid Chroma persist directory path configured.") from e
    # --- End path conversion ---

    # Initialize embedding function if not provided
    if embedding_function is None:
        embedding_function = get_embedding_model()

    # Check cache
    if not force_recreate and _vector_store_cache is not None:
        if (
            _persist_directory_cache == persist_path_obj  # Compare Path objects
            and _embedding_function_cache == embedding_function
        ):
            # logger.debug("Returning cached Chroma vector store instance.")
            return _vector_store_cache
        else:
            logger.info(
                "Cache miss: Persist directory or embedding function changed. Re-initializing."
            )

    logger.info(
        f"Initializing Chroma vector store. Persist directory: {persist_path_obj}"
    )
    logger.debug(f"Using embedding function: {type(embedding_function).__name__}")

    try:
        # --- Ensure the persist directory exists using pathlib ---
        persist_path_obj.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured persist directory exists: {persist_path_obj}")
        # --- End directory creation ---

        # Initialize ChromaDB - it expects a string path
        vector_store = Chroma(
            persist_directory=str(persist_path_obj),
            embedding_function=embedding_function,
        )

        # Update cache with Path object
        _vector_store_cache = vector_store
        _embedding_function_cache = embedding_function
        _persist_directory_cache = persist_path_obj

        logger.info("Chroma vector store initialized successfully.")
        return vector_store

    except Exception as e:
        logger.exception(
            f"Failed to initialize Chroma vector store at {persist_path_obj}: {e}",
            exc_info=True,
        )
        raise RuntimeError("Chroma vector store initialization failed.") from e


# --- Add Documents ---
def add_documents_to_store(
    documents: List[Document],
    vector_store: Chroma,
    generated_ids: Optional[List[str]] = None,
    batch_size: int = 128,  # Define a batch size for API calls
) -> List[str]:
    """
    Adds or updates documents in the vector store in batches using generated IDs
    for idempotency, showing progress with tqdm.

    Args:
        documents (List[Document]): The list of LangChain Document chunks to add/update.
        vector_store (Chroma): The initialized Chroma vector store instance.
        generated_ids (Optional[List[str]]): A list of pre-generated deterministic IDs,
                                                one for each document in the 'documents' list.
                                                If provided, enables upsert behavior.
        batch_size (int): The number of documents to process in each batch call
                            to the vector store (and underlying embedding API).

    Returns:
        List[str]: The list of IDs corresponding to the documents added or updated in Chroma.

    Raises:
        ValueError: If vector_store is invalid, parameters mismatch, or batch_size <= 0.
        RuntimeError: If adding/upserting documents fails.
    """
    if not isinstance(vector_store, Chroma):
        raise ValueError("Invalid vector_store provided. Must be a Chroma instance.")
    if not documents:
        logger.warning("No documents provided to add_documents_to_store.")
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    doc_count = len(documents)
    ids_provided = bool(generated_ids)

    if ids_provided and len(generated_ids) != doc_count:
        raise ValueError(
            f"Number of generated IDs ({len(generated_ids)}) does not match number of documents ({doc_count})."
        )

    operation = "Upserting (add/update)" if ids_provided else "Adding"
    logger.info(
        f"{operation} {doc_count} document chunks to the vector store in batches of {batch_size}..."
    )

    all_processed_ids_from_chroma: List[str] = []
    total_processed_count = 0

    # --- Iterate through documents in batches with tqdm ---
    num_batches = (doc_count + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, doc_count, batch_size),
        desc=f"{operation} documents",
        total=num_batches,
        unit="batch",
        ncols=100,
    ):
        batch_start = i
        batch_end = min(i + batch_size, doc_count)
        batch_docs = documents[batch_start:batch_end]
        batch_ids = generated_ids[batch_start:batch_end] if ids_provided else None

        # Skip empty batches (shouldn't happen with correct range logic, but safe)
        if not batch_docs:
            continue

        try:
            # Pass the batch and corresponding IDs (if any)
            added_ids_batch: List[str] = vector_store.add_documents(
                documents=batch_docs, ids=batch_ids
            )
            all_processed_ids_from_chroma.extend(added_ids_batch)
            total_processed_count += len(
                added_ids_batch
            )  # Count what Chroma reports back

            # Optional: Update tqdm postfix with current progress count
            # tqdm_batch_iter.set_postfix_str(f"Processed: {total_processed_count}/{doc_count}", refresh=True)

        except ValueError as ve:
            logger.error(
                f"Metadata validation error adding batch starting at index {batch_start}: {ve}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to add/update batch starting at {batch_start} due to invalid metadata."
            ) from ve
        except Exception as e:
            logger.exception(
                f"Error adding/updating batch starting at index {batch_start}.",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to add/update batch starting at {batch_start}."
            ) from e
    # --- End batch loop ---

    final_processed_count = len(all_processed_ids_from_chroma)
    logger.info(
        f"Successfully processed {final_processed_count} document chunks in total for addition/update."
    )

    # Sanity checks after loop completion
    if final_processed_count != doc_count:
        logger.warning(
            f"Final processed count from Chroma ({final_processed_count}) "
            f"does not match total document count ({doc_count}). This might happen if Chroma's return value varies."
        )
    if ids_provided and len(set(all_processed_ids_from_chroma)) != len(
        set(generated_ids)
    ):
        logger.warning(
            "The set of IDs returned by Chroma during upsert does not exactly match the set of generated IDs passed."
        )

    # Return the list of IDs we intended to process if using generated IDs,
    # otherwise return what Chroma reported back. Generated IDs are more reliable for tracking during upsert.
    return generated_ids if ids_provided else all_processed_ids_from_chroma


# --- Retrieve Documents ---
def get_retriever(
    vector_store: Chroma, search_type: str = "similarity", k: int = 5
) -> VectorStoreRetriever:
    """
    Creates and returns a LangChain retriever from the vector store.

    Args:
        vector_store (Chroma): The initialized Chroma vector store instance.
        search_type (str): The type of search to perform. Common options:
                            "similarity" (default), "mmr" (Maximal Marginal Relevance).
        k (int): The number of documents to retrieve (default is 5).

    Returns:
        VectorStoreRetriever: A LangChain retriever instance.

    Raises:
        ValueError: If the vector_store is not provided or invalid.
    """
    if not isinstance(vector_store, Chroma):
        raise ValueError("Invalid vector_store provided. Must be a Chroma instance.")

    logger.info(f"Creating retriever with search_type='{search_type}', k={k}")
    try:
        retriever: VectorStoreRetriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )
        return retriever
    except Exception as e:
        logger.exception("Failed to create retriever from vector store.", exc_info=True)
        raise RuntimeError("Failed to create retriever.") from e


if __name__ == "__main__":
    from utils.logging_config import setup_logging

    setup_logging()  # Config logging first
    logger.info("--- Testing vector_store.py ---")

    try:
        # 1. Get the vector store (initializes or loads from disk)
        logger.info("Attempting to initialize vector store...")
        vs = get_vector_store(force_recreate=False)  # Use cache if available
        logger.info(f"Successfully obtained vector store instance: {type(vs)}")
        logger.info(f"Vector store collection name: {vs._collection.name}")
        logger.info(
            f"Items in store before add: {vs._collection.count()}"
        )  # Access underlying chroma collection count

        # 2. Add a dummy document
        logger.info("Attempting to add a dummy document...")
        dummy_doc = Document(
            page_content="This is a test document stored in Chroma.",
            metadata={"source": "test_script", "timestamp": 12345},
        )
        added_id = add_documents_to_store([dummy_doc], vs)
        logger.info(f"Added dummy document. ID: {added_id}")
        logger.info(f"Items in store after add: {vs._collection.count()}")

        # 3. Create a retriever
        logger.info("Attempting to create a retriever...")
        retriever = get_retriever(vs, k=1)
        logger.info(f"Successfully created retriever: {type(retriever)}")

        # 4. Test retrieval (requires embeddings to work)
        logger.info("Attempting to retrieve the dummy document...")
        query = "test document content"
        try:
            retrieved_docs = retriever.invoke(
                query
            )  # Use invoke for LCEL compatibility
            logger.info(
                f"Retrieved {len(retrieved_docs)} documents for query '{query}'."
            )
            if retrieved_docs:
                logger.info(
                    f"Top retrieved doc match: {retrieved_docs[0].page_content}"
                )
                logger.info(f"Metadata: {retrieved_docs[0].metadata}")
            else:
                logger.warning(
                    "Could not retrieve the dummy document (check query/embeddings)."
                )
        except Exception as retrieve_err:
            logger.error(f"Retrieval test failed: {retrieve_err}")

        # 5. Test caching (get store again)
        logger.info("Attempting to get vector store again (testing cache)...")
        vs_cached = get_vector_store()
        logger.info(f"Cache test: Instance is the same: {vs is vs_cached}")

    except (RuntimeError, ValueError) as e:
        logger.error(f"Error during vector store test: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during vector store test: {e}", exc_info=True)

    logger.info("--- Finished testing vector_store.py ---")
