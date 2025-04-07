import logging
from typing import List

# LangChain's text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document  # LangChain's Document object type hint

# Define logger for this module
logger = logging.getLogger(__name__)

# Default configuration values (can be overridden)
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> TextSplitter:
    """
    Initializes and returns a LangChain text splitter, configured for optimal use
    with embedding models (like those from OpenAI).

    Uses RecursiveCharacterTextSplitter, which is generally recommended.

    Args:
        chunk_size (int): The target size for each text chunk (in characters).
                            Defaults to DEFAULT_CHUNK_SIZE.
        chunk_overlap (int): The number of characters to overlap between adjacent chunks.
                                Helps maintain context. Defaults to DEFAULT_CHUNK_OVERLAP.

    Returns:
        TextSplitter: An initialized LangChain TextSplitter instance.
    """
    logger.info(
        f"Initializing RecursiveCharacterTextSplitter with "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )
    try:
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Uses standard separators like "\n\n", "\n", " ", ""
            # length_function can be customized if needed, default is len()
            length_function=len,
            # is_separator_regex=False, # Default is False
        )
        return text_splitter
    except Exception as e:
        logger.exception(
            "Failed to initialize RecursiveCharacterTextSplitter.", exc_info=True
        )
        raise RuntimeError("Text splitter initialization failed.") from e


# --- Example Usage Function (Optional - can be called from __main__) ---
def split_sample_documents(splitter: TextSplitter) -> None:
    """Helper function to demonstrate splitting some sample documents."""
    logger = logging.getLogger(__name__)  # Get logger inside function if needed
    sample_text_1 = """
    This is the first document. It is relatively short.
    It discusses the importance of chunking text for RAG systems.
    Large documents need to be split before embedding.
    """
    sample_text_2 = "This is a second, very short document."
    sample_text_3 = """
    The third document is longer. It talks about different splitting strategies.
    One could split by paragraphs, sentences, or using recursive character counts.
    Recursive splitting tries to keep related text together by splitting on larger
    separators first (like double newlines) before moving to smaller ones.
    This often yields better results for semantic retrieval as meaningful units
    are less likely to be broken apart arbitrarily mid-sentence. Overlap helps
    connect chunks. Choosing the right chunk size and overlap depends on the
    embedding model's context window and the nature of the documents.
    """

    # Create LangChain Document objects (as splitter often works with these)
    docs: List[Document] = [
        Document(page_content=sample_text_1, metadata={"source": "sample1.txt"}),
        Document(page_content=sample_text_2, metadata={"source": "sample2.txt"}),
        Document(page_content=sample_text_3, metadata={"source": "sample3.txt"}),
    ]

    logger.info(f"Splitting {len(docs)} sample documents...")
    try:
        split_docs: List[Document] = splitter.split_documents(docs)
        logger.info(f"Successfully split into {len(split_docs)} chunks.")

        for i, chunk in enumerate(split_docs):
            logger.debug(
                f"--- Chunk {i + 1} (Source: {chunk.metadata.get('source', 'N/A')})---"
            )
            # Ensure page_content is string before slicing
            content_preview = (
                str(chunk.page_content)[:150].replace("\n", " ") + "..."
                if chunk.page_content
                else "[Empty Content]"
            )

            logger.debug(f"Content preview: {content_preview}")
            # Note: RecursiveCharacterTextSplitter automatically adds metadata
            # about the source document if the input is a Document object.
        logger.info("Finished splitting sample documents.")

    except Exception as e:
        logger.exception("Error during sample document splitting.", exc_info=True)


if __name__ == "__main__":
    from utils.logging_config import setup_logging

    setup_logging()  # Setup logging first
    logger.info("--- Testing text_splitter.py ---")

    try:
        # Get splitter with default settings
        default_splitter = get_text_splitter()
        logger.info(f"Created default splitter: {type(default_splitter)}")
        split_sample_documents(default_splitter)

        # Get splitter with custom settings
        custom_splitter = get_text_splitter(chunk_size=150, chunk_overlap=30)
        logger.info(f"Created custom splitter: {type(custom_splitter)}")
        split_sample_documents(custom_splitter)

    except RuntimeError as e:
        logger.error(f"Runtime Error during text splitter test: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected Error during text splitter test: {e}", exc_info=True)
    logger.info("--- Finished testing text_splitter.py ---")
