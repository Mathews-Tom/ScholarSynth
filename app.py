import logging
import sys
from pathlib import Path  # Import Path
from typing import List, Optional  # For type hints

# --- Add project root using pathlib ---
# This file is in the root, so project_root is its directory
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:  # Compare string paths
    sys.path.insert(0, str(project_root))
# --- End path modification ---

# Import Streamlit first to potentially avoid configuration conflicts
import streamlit as st
from langchain_community.vectorstores import Chroma  # For type hinting
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # For type hinting
from langchain_core.output_parsers import StrOutputParser

# LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever  # For type hinting
from langchain_openai import ChatOpenAI

# OpenAI error types for better error handling
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from utils.error_handling import (
    ErrorHandlingCallbackHandler,
    handle_error,
    retry_with_exponential_backoff,
)

# Project-specific modules
# Setup logging BEFORE other modules that use logging
from utils.logging_config import setup_logging

setup_logging()  # Configure logger

from retrieval.embedding_config import get_embedding_model
from retrieval.vector_store import get_vector_store  # Import base function
from utils.config_loader import load_config

# Define logger for this application
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_RETRIEVER_K = 5
DEFAULT_SEARCH_TYPE = "similarity"  # Other option: "mmr"

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
        # Use retry decorator for API calls that might fail due to rate limits or connection issues
        @retry_with_exponential_backoff(max_retries=3)
        def init_embeddings():
            # We pass necessary config bits implicitly via get_embedding_model's internal config load
            return get_embedding_model()

        model = init_embeddings()
        logger.info("Embedding model initialized successfully.")
        return model
    except (APIConnectionError, APITimeoutError) as e:
        # Handle connection-related errors
        logger.error(
            f"Connection error initializing embedding model: {e}", exc_info=True
        )
        handle_error(e)
        st.stop()
        return None
    except RateLimitError as e:
        # Handle rate limit errors
        logger.error(f"Rate limit exceeded for embedding model: {e}", exc_info=True)
        handle_error(e)
        st.stop()
        return None
    except Exception as e:
        # Handle all other errors
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        handle_error(e)
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
        # Use retry decorator for operations that might fail due to temporary issues
        @retry_with_exponential_backoff(max_retries=2)
        def init_vector_store():
            # Pass the string path as expected by our modified get_vector_store wrapper
            return get_vector_store(
                persist_directory_str=persist_dir_str,  # Use the string from config
                embedding_function=_embedding_model,
            )

        vector_store = init_vector_store()

        # Safely get count for logging
        try:
            count = vector_store._collection.count()
            logger.info(
                f"Vector store initialized from '{persist_path}'. Found {count} items."
            )

            if count == 0:
                # This is a warning, not necessarily a fatal error if expecting an empty store initially
                logger.warning(
                    f"Vector store loaded from '{persist_path}' but contains 0 items."
                )
                st.warning(
                    "Warning: Vector store initialized, but it appears to be empty. "
                    "Ensure ingestion was successful if data is expected."
                )
        except Exception as count_error:
            logger.warning(f"Could not get item count from vector store: {count_error}")
            # Continue anyway since this is not critical

        return vector_store
    except (APIConnectionError, APITimeoutError) as e:
        # Handle connection-related errors
        logger.error(f"Connection error initializing vector store: {e}", exc_info=True)
        handle_error(e)
        st.stop()
        return None
    except Exception as e:
        # Handle all other errors
        logger.error(
            f"Failed to initialize vector store from '{persist_path}': {e}",
            exc_info=True,
        )
        handle_error(e)
        st.stop()
        return None


@st.cache_resource
def get_cached_llm(_config, streaming=True):  # Depends on config
    logger.info("Initializing Language Model...")
    try:
        # Use retry decorator for API calls that might fail due to rate limits or connection issues
        @retry_with_exponential_backoff(max_retries=3)
        def init_llm():
            return ChatOpenAI(
                model_name=_config.LLM_MODEL_NAME,
                openai_api_key=_config.OPENAI_API_KEY,
                temperature=0.1,  # Lower temperature for more factual/consistent answers
                streaming=streaming,  # Enable streaming for token-by-token response
                # max_tokens= # Optional: Limit response length
            )

        llm = init_llm()
        logger.info(
            f"LLM '{_config.LLM_MODEL_NAME}' initialized successfully. Streaming: {streaming}"
        )
        return llm
    except (APIConnectionError, APITimeoutError) as e:
        # Handle connection-related errors
        logger.error(f"Connection error initializing LLM: {e}", exc_info=True)
        handle_error(e)
        st.stop()
        return None
    except RateLimitError as e:
        # Handle rate limit errors
        logger.error(f"Rate limit exceeded for LLM: {e}", exc_info=True)
        handle_error(e)
        st.stop()
        return None
    except Exception as e:
        # Handle all other errors
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        handle_error(e)
        st.stop()
        return None


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
    """Creates the RAG chain using LCEL with error handling."""
    logger.debug("Setting up RAG chain...")

    # Create a more robust prompt with error handling guidance
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Create a more robust retrieval step with error handling
    @retry_with_exponential_backoff(max_retries=2)
    def retrieval_with_retry(query):
        try:
            return retriever.invoke(query)
        except Exception as e:
            logger.warning(f"Retrieval failed, falling back to empty context: {e}")
            # Return an empty list as fallback to avoid complete failure
            return []

    # Parallel Runnable to fetch context and pass question through
    # Use the retry-enabled retrieval function
    setup_and_retrieval = RunnableParallel(
        {
            "context": lambda x: format_docs(retrieval_with_retry(x)),
            "question": RunnablePassthrough(),
        }
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
    k_value = st.sidebar.slider(
        "Number of document chunks to retrieve (k):",
        min_value=1,
        max_value=20,
        value=DEFAULT_RETRIEVER_K,
        step=1,
        help="How many relevant text chunks should be retrieved from the vector store to form the context?",
    )
    search_type = st.sidebar.selectbox(
        "Retrieval search type:",
        options=["similarity", "mmr"],
        index=0 if DEFAULT_SEARCH_TYPE == "similarity" else 1,
        help="'Similarity' finds chunks closest to the query vector. 'MMR' (Maximal Marginal Relevance) tries to find relevant *and* diverse chunks.",
    )

    # --- Initialize Retriever and RAG Chain (dependent on sidebar settings) ---
    # Recreate retriever and chain if settings change.
    try:
        # Use the vector store's as_retriever method directly
        retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": k_value}
        )
        logger.info(f"Created retriever with search_type='{search_type}', k={k_value}")
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

        # Create a callback handler for error handling
        error_handler = ErrorHandlingCallbackHandler()

        try:
            # Retrieve documents for context
            with st.spinner(
                f"Searching relevant documents... (using k={k_value}, search='{search_type}')"
            ):
                try:
                    # Use retry for retrieval operations
                    @retry_with_exponential_backoff(max_retries=2)
                    def retrieve_documents():
                        return retriever.invoke(user_question)

                    retrieved_docs = retrieve_documents()
                except (APIConnectionError, APITimeoutError) as e:
                    # Handle connection-related errors
                    logger.error(
                        f"Connection error during document retrieval: {e}",
                        exc_info=True,
                    )
                    handle_error(e)
                    return
                except RateLimitError as e:
                    # Handle rate limit errors
                    logger.error(
                        f"Rate limit exceeded during document retrieval: {e}",
                        exc_info=True,
                    )
                    handle_error(e)
                    return
                except Exception as retrieval_error:
                    logger.error(
                        f"Error retrieving documents: {retrieval_error}", exc_info=True
                    )
                    handle_error(retrieval_error)
                    return

            # Create a placeholder for the streaming response
            answer_placeholder = st.empty()

            # Stream the response with error handling
            try:
                # Initialize the response string
                full_response = ""

                # Stream the response with the error handler callback
                for chunk in rag_chain.stream(
                    user_question, config={"callbacks": [error_handler]}
                ):
                    # Check if an error occurred during streaming
                    if error_handler.has_error:
                        break

                    full_response += chunk
                    answer_placeholder.markdown(full_response + "▌")

                # If no errors occurred, display the final response
                if not error_handler.has_error:
                    answer_placeholder.markdown(full_response)
            except (APIConnectionError, APITimeoutError) as e:
                # Handle connection-related errors
                logger.error(
                    f"Connection error during LLM response generation: {e}",
                    exc_info=True,
                )
                handle_error(e)
                answer_placeholder.markdown(
                    "⚠️ **Connection error occurred while generating the response.**"
                )
            except RateLimitError as e:
                # Handle rate limit errors
                logger.error(
                    f"Rate limit exceeded during LLM response generation: {e}",
                    exc_info=True,
                )
                handle_error(e)
                answer_placeholder.markdown(
                    "⚠️ **Rate limit exceeded. Please try again in a moment.**"
                )
            except Exception as e:
                # Handle all other errors
                logger.error(
                    f"Error during LLM response generation: {e}", exc_info=True
                )
                handle_error(e)
                answer_placeholder.markdown(
                    "⚠️ **An error occurred while generating the response.**"
                )

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

        except (APIConnectionError, APITimeoutError) as e:
            # Handle connection-related errors
            logger.error(
                f"Connection error processing question '{user_question}': {e}",
                exc_info=True,
            )
            handle_error(e)
            st.error(
                "⚠️ **Connection error occurred.** Please check your internet connection and try again."
            )
        except RateLimitError as e:
            # Handle rate limit errors
            logger.error(
                f"Rate limit exceeded for question '{user_question}': {e}",
                exc_info=True,
            )
            handle_error(e)
            st.error(
                "⚠️ **Rate limit exceeded.** The system is currently experiencing high demand. "
                "Please wait a moment and try again."
            )
        except Exception as e:
            # Handle all other errors
            logger.error(
                f"Error processing user question '{user_question}': {e}", exc_info=True
            )
            handle_error(e)
            st.error(
                f"⚠️ **An error occurred while processing your question.** "
                f"Please try rephrasing or ask a different question."
            )

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
