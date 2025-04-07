import logging
import sys
from pathlib import Path  # Import Path
from typing import List  # For type hints

# --- Add project root using pathlib ---
# This file is in the root, so project_root is its directory
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:  # Compare string paths
    sys.path.insert(0, str(project_root))
# --- End path modification ---

# Import Streamlit first to potentially avoid configuration conflicts
import os

import streamlit as st
from langchain_chroma import Chroma  # For type hinting
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # For type hinting
from langchain_core.output_parsers import StrOutputParser

# LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever  # For type hinting
from langchain_openai import ChatOpenAI

# Project-specific modules
# Setup logging BEFORE other modules that use logging
from utils.logging_config import setup_logging

setup_logging()  # Configure logger

from retrieval.embedding_config import get_embedding_model
from retrieval.parent_document_retriever import get_parent_document_retriever
from retrieval.vector_store import get_vector_store  # Import base function
from utils.config_loader import load_config

# Define logger for this application
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_RETRIEVER_K = 5
DEFAULT_SEARCH_TYPE = "similarity"  # Other option: "mmr"
DEFAULT_RETRIEVER_TYPE = "standard"  # Other option: "parent_document"

# --- Application State & Caching ---


# Cache the loading of configuration to avoid repeated file access
@st.cache_data
def cached_load_config():
    logger.info("Loading application configuration...")
    try:
        config = load_config()
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.critical(f"FATAL: Failed to load configuration: {e}", exc_info=True)
        st.error(
            f"CRITICAL ERROR: Could not load configuration. Please check environment variables and .env file. Error: {e}"
        )
        st.stop()  # Stop execution if config fails
        return None  # Should not be reached due to st.stop()


# Cache resources like models and vector stores to avoid reloading on every interaction
@st.cache_resource
def get_cached_embedding_model(
    _config,
):  # Pass config to ensure cache invalidation if relevant config changes
    logger.info("Initializing embedding model...")
    try:
        # We pass necessary config bits implicitly via get_embedding_model's internal config load
        model = get_embedding_model()
        logger.info("Embedding model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        st.error(
            f"ERROR: Could not initialize embedding model. Please check OpenAI API key and configuration. Error: {e}"
        )
        st.stop()
        return None


@st.cache_resource
def get_cached_vector_store(
    _config, _embedding_model
):  # Depends on config and embedding model
    logger.info("Initializing vector store...")
    persist_dir_str = _config.CHROMA_PERSIST_PATH
    if not persist_dir_str:
        logger.error("CHROMA_PERSIST_PATH not configured.")
        st.error("ERROR: Vector store path not configured. Check .env file.")
        st.stop()
        return None

    # --- Check vector store path using pathlib ---
    # Assume path in config is relative to project root
    persist_path = project_root / persist_dir_str
    persist_path = persist_path.resolve()  # Make absolute

    if not persist_path.is_dir():
        logger.error(f"Vector store directory '{persist_path}' does not exist.")
        st.error(
            f"ERROR: Vector store not found at '{persist_path}'. Please run the ingestion script first (`python scripts/ingest.py`)."
        )
        st.stop()
        return None
    # Optional: Check if directory is empty (can be slow for large dirs, uses more resources)
    # Consider skipping this check unless emptiness is a critical failure condition
    # try:
    #     is_empty = not any(persist_path.iterdir())
    # except StopIteration: # More accurate check for empty iterator
    #     is_empty = True
    # except Exception as e:
    #      logger.warning(f"Could not check if vector store directory is empty: {e}")
    #      is_empty = False # Assume not empty if check fails
    #
    # if is_empty:
    #     logger.error(f"Vector store directory '{persist_path}' exists but appears empty.")
    #     st.error(f"ERROR: Vector store directory is empty at '{persist_path}'. Please ensure the ingestion script ran successfully and added data.")
    #     st.stop()
    #     return None
    # --- End path check ---

    try:
        # Pass the string path as expected by our modified get_vector_store wrapper
        vector_store = get_vector_store(
            persist_directory_str=persist_dir_str,  # Use the string from config
            embedding_function=_embedding_model,
        )
        # Logging about path/count is handled inside get_vector_store now
        count = (
            vector_store._collection.count()
        )  # Direct Chroma collection access might change
        logger.info(
            f"Vector store initialized from '{persist_path}'. Found {count} items."
        )
        if count == 0:
            # This is a warning, not necessarily a fatal error if expecting an empty store initially
            logger.warning(
                f"Vector store loaded from '{persist_path}' but contains 0 items."
            )
            st.warning(
                "Warning: Vector store initialized, but it appears to be empty. Ensure ingestion was successful if data is expected."
            )

        return vector_store
    except Exception as e:
        logger.error(
            f"Failed to initialize vector store from '{persist_path}': {e}",
            exc_info=True,
        )
        st.error(
            f"ERROR: Could not initialize vector store from '{persist_path}'. Error: {e}"
        )
        st.stop()
        return None


@st.cache_resource
def get_cached_parent_document_retriever(
    _config, _embedding_model
):  # Depends on config and embedding model
    logger.info("Initializing parent document retriever...")

    # Get paths from config
    vector_store_path = _config.CHROMA_PERSIST_PATH
    parent_doc_store_path = getattr(
        _config, "PARENT_DOC_STORE_PATH", "data/parent_doc_store"
    )

    # Check if vector store path exists
    if not vector_store_path:
        logger.error("CHROMA_PERSIST_PATH not configured.")
        st.error("ERROR: Vector store path not configured. Check .env file.")
        st.stop()
        return None

    # Check if vector store directory exists
    vector_store_dir = project_root / vector_store_path
    if not vector_store_dir.is_dir():
        logger.error(f"Vector store directory '{vector_store_dir}' does not exist.")
        st.error(
            f"ERROR: Vector store not found at '{vector_store_dir}'. Please run the ingestion script first (`python scripts/ingest.py --use-parent-retriever`)."
        )
        st.stop()
        return None

    # Get chunk sizes from config
    parent_chunk_size = getattr(_config, "PARENT_CHUNK_SIZE", 2000)
    parent_chunk_overlap = getattr(_config, "PARENT_CHUNK_OVERLAP", 400)
    child_chunk_size = getattr(_config, "CHILD_CHUNK_SIZE", 500)
    child_chunk_overlap = getattr(_config, "CHILD_CHUNK_OVERLAP", 100)

    try:
        # Initialize parent document retriever
        retriever = get_parent_document_retriever(
            vector_store_path=vector_store_path,
            parent_doc_store_path=parent_doc_store_path,
            embedding_model=_embedding_model,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
        )

        # Get counts
        child_count = retriever.vectorstore._collection.count()
        parent_count = len(retriever.docstore.store)

        logger.info(
            f"Parent document retriever initialized successfully. "
            f"Found {child_count} child chunks and {parent_count} parent documents."
        )

        if child_count == 0 or parent_count == 0:
            logger.warning(
                "Parent document retriever initialized, but it appears to be empty."
            )
            st.warning(
                "Warning: Parent document retriever initialized, but it appears to be empty. "
                "Ensure ingestion was successful using `python scripts/ingest.py --use-parent-retriever`."
            )

        return retriever
    except Exception as e:
        logger.error(
            f"Failed to initialize parent document retriever: {e}",
            exc_info=True,
        )
        st.error(f"ERROR: Could not initialize parent document retriever. Error: {e}")
        st.stop()
        return None


class MockLLM:
    """Simple mock LLM for testing purposes."""

    def __init__(self, model_name="mock-llm", temperature=0.1):
        """Initialize the mock LLM."""
        # Store parameters but don't use them
        self._model_name = model_name
        self._temperature = temperature

    def invoke(self, prompt):
        """Generate a mock response for a prompt."""
        # Return a simple response based on the prompt
        if isinstance(prompt, str):
            if "rag" in prompt.lower():
                return "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative models. It retrieves relevant documents from a knowledge base and provides them as context to the language model, allowing it to generate more accurate and informed responses."
            else:
                return "I'm a mock LLM for testing purposes. I can't provide a real response without a valid OpenAI API key."
        else:
            # Handle the case where prompt is a list of messages or other structure
            return "I'm a mock LLM for testing purposes. I can't provide a real response without a valid OpenAI API key."


@st.cache_resource
def get_cached_llm(_config):  # Depends on config
    logger.info("Initializing Language Model...")
    try:
        # Check if we're in test mode (no API key)
        if (
            not os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY") == "YOUR_API_KEY_HERE"
        ):
            logger.warning("No valid OpenAI API key found. Using mock LLM for testing.")
            return MockLLM()

        llm = ChatOpenAI(
            model_name=_config.LLM_MODEL_NAME,
            openai_api_key=_config.OPENAI_API_KEY,
            temperature=0.1,  # Lower temperature for more factual/consistent answers
            # max_tokens= # Optional: Limit response length
        )
        logger.info(f"LLM '{_config.LLM_MODEL_NAME}' initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        logger.warning("Falling back to mock LLM for testing.")
        return MockLLM()


# --- RAG Chain Definition ---

# Define the prompt template
RAG_PROMPT_TEMPLATE = """
You are an assistant specialized in answering questions based on academic literature.
Use the following retrieved context from research papers to answer the question.
If you don't know the answer from the context, just say that you don't know.
Do not make up an answer. Keep the answer concise and relevant to the question.
Cite the source document(s) you used from the metadata provided in the context. Use the source filename or ArXiv ID. Example: [Source: my_paper.pdf] or [Source: arxiv:2301.12345].

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents into a single string for the prompt."""
    formatted_context = []
    seen_sources = set()
    for i, doc in enumerate(docs):
        # Prioritize 'filename' if available, otherwise use 'source'
        source_ref = doc.metadata.get("filename", doc.metadata.get("source", "N/A"))
        source_str = f"[Source: {source_ref}]"
        seen_sources.add(source_ref)

        # Optionally include title if available
        title = doc.metadata.get("title")
        title_str = f" [Title: {title}]" if title else ""

        content_chunk = doc.page_content.replace("\n", " ")  # Basic formatting
        formatted_context.append(
            f"--- Context Chunk {i + 1} {source_str}{title_str} ---\n{content_chunk}"
        )

    # Construct the final context string
    # final_context_str = "\n\n".join(formatted_context)
    # Optionally add a summary of sources used at the end or beginning:
    # sources_summary = f"Context derived from sources: {', '.join(sorted(list(seen_sources)))}"
    # return f"{sources_summary}\n\n{final_context_str}"
    return "\n\n".join(formatted_context)


def setup_rag_chain(retriever: VectorStoreRetriever, llm: ChatOpenAI):
    """Creates the RAG chain using LCEL."""
    logger.debug("Setting up RAG chain...")
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Parallel Runnable to fetch context and pass question through
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )

    # The main RAG chain
    rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()
    logger.debug("RAG chain setup complete.")
    return rag_chain


# --- Streamlit UI ---


def run_app():
    # --- Page Configuration ---
    st.set_page_config(
        page_title="AI Literature Review Assistant",
        page_icon="📚",
        layout="wide",  # Or "centered"
    )

    st.title("📚 AI Literature Review Assistant")
    st.markdown(
        "Ask questions about the research papers loaded into the knowledge base."
    )

    # --- Load Resources ---
    try:
        config = cached_load_config()
        if not config:
            return  # Exit if config loading failed above

        embedding_model = get_cached_embedding_model(config)
        if not embedding_model:
            return

        vector_store = get_cached_vector_store(config, embedding_model)
        if not vector_store:
            return

        llm = get_cached_llm(config)
        if not llm:
            return

    except Exception as e:
        # Catch any unexpected errors during setup not caught by cache decorators
        logger.critical(
            f"Unexpected error during application resource initialization: {e}",
            exc_info=True,
        )
        st.error(f"A critical error occurred during setup: {e}")
        st.stop()
        return

    # --- Sidebar for Configuration ---
    st.sidebar.header("Query Configuration")

    # Retriever type selection
    retriever_type = st.sidebar.selectbox(
        "Retriever type:",
        options=["standard", "parent_document"],
        index=0 if DEFAULT_RETRIEVER_TYPE == "standard" else 1,
        help="'Standard' retrieves individual chunks. 'Parent Document' retrieves smaller chunks but returns their parent documents for better context.",
    )

    # Number of chunks to retrieve
    k_value = st.sidebar.slider(
        "Number of document chunks to retrieve (k):",
        min_value=1,
        max_value=20,
        value=DEFAULT_RETRIEVER_K,
        step=1,
        help="How many relevant text chunks should be retrieved from the vector store to form the context?",
    )

    # Search type selection
    search_type = st.sidebar.selectbox(
        "Retrieval search type:",
        options=["similarity", "mmr"],
        index=0 if DEFAULT_SEARCH_TYPE == "similarity" else 1,
        help="'Similarity' finds chunks closest to the query vector. 'MMR' (Maximal Marginal Relevance) tries to find relevant *and* diverse chunks.",
    )

    # --- Initialize Retriever and RAG Chain (dependent on sidebar settings) ---
    # Recreate retriever and chain if settings change.
    try:
        # Initialize the appropriate retriever based on user selection
        if retriever_type == "standard":
            # Use the standard vector store retriever
            retriever = vector_store.as_retriever(
                search_type=search_type, search_kwargs={"k": k_value}
            )
            logger.info(
                f"Created standard retriever with search_type='{search_type}', k={k_value}"
            )
        else:
            # Use the parent document retriever
            try:
                # Initialize parent document retriever if not already done
                parent_retriever = get_cached_parent_document_retriever(
                    config, embedding_model
                )
                if not parent_retriever:
                    st.error(
                        "Failed to initialize parent document retriever. Falling back to standard retriever."
                    )
                    retriever = vector_store.as_retriever(
                        search_type=search_type, search_kwargs={"k": k_value}
                    )
                else:
                    # Update search parameters
                    retriever = parent_retriever.as_retriever(k=k_value)
                    logger.info(f"Created parent document retriever with k={k_value}")
            except Exception as e:
                logger.error(
                    f"Error initializing parent document retriever: {e}", exc_info=True
                )
                st.error(
                    f"Error initializing parent document retriever: {e}. Falling back to standard retriever."
                )
                retriever = vector_store.as_retriever(
                    search_type=search_type, search_kwargs={"k": k_value}
                )

        # Create the RAG chain with the selected retriever
        rag_chain = setup_rag_chain(retriever, llm)
    except Exception as e:
        logger.error(f"Failed to create retriever or RAG chain: {e}", exc_info=True)
        st.error(f"Error setting up the retrieval/RAG chain: {e}")
        st.stop()
        return

    # --- User Input ---
    st.markdown("---")
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main challenges of prompt engineering?",
    )

    # --- Generate and Display Response ---
    if user_question:
        logger.info(f"Received user question: '{user_question}'")
        st.markdown("---")
        st.subheader("Answer:")
        try:
            with st.spinner(
                f"Searching relevant documents and generating answer... (using k={k_value}, search='{search_type}')"
            ):
                # Invoke the RAG chain
                response = rag_chain.invoke(user_question)
                st.markdown(response)  # Display the LLM's answer

                # Optionally display retrieved documents for verification
                st.markdown("---")
                with st.expander("Show Retrieved Context Documents"):
                    # Retrieve documents again using the current retriever settings
                    retrieved_docs = retriever.invoke(user_question)
                    if retrieved_docs:
                        st.markdown(
                            f"Retrieved {len(retrieved_docs)} chunks for context:"
                        )
                        for i, doc in enumerate(retrieved_docs):
                            # Access potential retrieval score if available (depends on vector store implementation)
                            score = doc.metadata.get("_score", None)
                            score_str = (
                                f" (Score: {score:.4f})" if score is not None else ""
                            )
                            st.markdown(f"**Chunk {i + 1}{score_str}**")

                            # Display key metadata
                            source_ref = doc.metadata.get(
                                "filename", doc.metadata.get("source", "N/A")
                            )
                            st.markdown(f"*Source:* `{source_ref}`")
                            if "title" in doc.metadata:
                                st.markdown(f"*Title:* {doc.metadata.get('title')}")
                            # Add other relevant metadata if desired (e.g., authors)
                            # if 'authors' in doc.metadata:
                            #     st.markdown(f"*Authors:* {doc.metadata.get('authors')}")

                            st.text_area(
                                f"Content Chunk {i + 1}",
                                value=doc.page_content,
                                height=150,
                                key=f"doc_{i}",
                            )
                            st.markdown("---")
                    else:
                        st.markdown("No documents were retrieved for this query.")

        except Exception as e:
            logger.error(
                f"Error processing user question '{user_question}': {e}", exc_info=True
            )
            st.error(f"An error occurred while processing your question: {e}")

    # --- Footer/Info ---
    st.sidebar.markdown("---")
    try:
        # Get current item count safely
        vs_count = vector_store._collection.count()
    except Exception:
        vs_count = "N/A"  # Handle case where count fails

    st.sidebar.info(
        "This app uses a Retrieval-Augmented Generation (RAG) pipeline "
        "to answer questions based on documents loaded via `scripts/ingest.py`."
        f"\n\nVector Store: ChromaDB ({vs_count} items)"
        f"\nEmbedding Model: {config.EMBEDDING_MODEL_NAME}"
        f"\nLLM: {config.LLM_MODEL_NAME}"
    )


if __name__ == "__main__":
    run_app()
