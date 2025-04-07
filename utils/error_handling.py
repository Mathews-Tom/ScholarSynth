"""
Error handling utilities for ScholarSynth.

This module provides functions and classes for handling various types of errors
that may occur during the operation of the ScholarSynth application, particularly
around API calls and user interactions.
"""

import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

import openai
import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler

# Import LangChain exceptions
from langchain_core.exceptions import LangChainException as LangChainError
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

# Configure logger
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categorization of errors for better user feedback."""

    API_CONNECTION = "API Connection Error"
    API_RATE_LIMIT = "API Rate Limit Error"
    API_TIMEOUT = "API Timeout Error"
    API_GENERAL = "API Error"
    AUTHENTICATION = "Authentication Error"
    VECTOR_STORE = "Vector Store Error"
    DOCUMENT_PROCESSING = "Document Processing Error"
    RETRIEVAL = "Retrieval Error"
    LLM_GENERATION = "LLM Generation Error"
    UNKNOWN = "Unknown Error"


def categorize_error(error: Exception) -> ErrorCategory:
    """
    Categorize an exception into a predefined error category.

    Args:
        error: The exception to categorize

    Returns:
        ErrorCategory: The categorized error type
    """
    if isinstance(error, APIConnectionError):
        return ErrorCategory.API_CONNECTION
    elif isinstance(error, RateLimitError):
        return ErrorCategory.API_RATE_LIMIT
    elif isinstance(error, APITimeoutError):
        return ErrorCategory.API_TIMEOUT
    elif isinstance(error, APIError):
        return ErrorCategory.API_GENERAL
    elif isinstance(error, openai.AuthenticationError):
        return ErrorCategory.AUTHENTICATION
    elif isinstance(error, LangChainError):
        if "retriev" in str(error).lower():
            return ErrorCategory.RETRIEVAL
        elif "vector" in str(error).lower() or "chroma" in str(error).lower():
            return ErrorCategory.VECTOR_STORE
        elif "document" in str(error).lower() or "chunk" in str(error).lower():
            return ErrorCategory.DOCUMENT_PROCESSING
        elif (
            "llm" in str(error).lower()
            or "model" in str(error).lower()
            or "generat" in str(error).lower()
        ):
            return ErrorCategory.LLM_GENERATION

    # Default case
    return ErrorCategory.UNKNOWN


def get_user_friendly_message(error: Exception, category: ErrorCategory) -> str:
    """
    Generate a user-friendly error message based on the error category.

    Args:
        error: The original exception
        category: The error category

    Returns:
        str: A user-friendly error message
    """
    base_messages = {
        ErrorCategory.API_CONNECTION: (
            "Unable to connect to the OpenAI API. Please check your internet connection "
            "and try again later."
        ),
        ErrorCategory.API_RATE_LIMIT: (
            "You've reached the rate limit for the OpenAI API. Please wait a moment "
            "before trying again."
        ),
        ErrorCategory.API_TIMEOUT: (
            "The request to the OpenAI API timed out. This might be due to high traffic "
            "or complex queries. Please try again or simplify your question."
        ),
        ErrorCategory.API_GENERAL: (
            "An error occurred while communicating with the OpenAI API. "
        ),
        ErrorCategory.AUTHENTICATION: (
            "Authentication failed. Please check your API key in the .env file."
        ),
        ErrorCategory.VECTOR_STORE: ("There was an issue with the vector database. "),
        ErrorCategory.DOCUMENT_PROCESSING: (
            "An error occurred while processing the documents. "
        ),
        ErrorCategory.RETRIEVAL: (
            "Failed to retrieve relevant documents for your query. "
        ),
        ErrorCategory.LLM_GENERATION: (
            "The AI model encountered an issue while generating a response. "
        ),
        ErrorCategory.UNKNOWN: ("An unexpected error occurred. "),
    }

    message = base_messages[category]

    # Add specific error details for technical users
    technical_details = f"\n\nTechnical details: {str(error)}"

    # Add recovery suggestions based on error type
    recovery_suggestions = {
        ErrorCategory.API_CONNECTION: (
            "\n\nSuggestions:\n"
            "• Check your internet connection\n"
            "• Verify that api.openai.com is accessible from your network\n"
            "• Try again in a few minutes"
        ),
        ErrorCategory.API_RATE_LIMIT: (
            "\n\nSuggestions:\n"
            "• Wait for a minute before trying again\n"
            "• Consider upgrading your OpenAI plan if this happens frequently\n"
            "• Try a simpler query or reduce the number of requests"
        ),
        ErrorCategory.API_TIMEOUT: (
            "\n\nSuggestions:\n"
            "• Try again with a simpler query\n"
            "• Reduce the number of documents to retrieve (k value)\n"
            "• Try during a less busy time"
        ),
        ErrorCategory.AUTHENTICATION: (
            "\n\nSuggestions:\n"
            "• Check that your OpenAI API key is correctly set in the .env file\n"
            "• Verify that your API key is still valid\n"
            "• Make sure you have sufficient credits in your OpenAI account"
        ),
        ErrorCategory.VECTOR_STORE: (
            "\n\nSuggestions:\n"
            "• Make sure you've run the ingestion script to populate the vector store\n"
            "• Check the vector store path in your .env file\n"
            "• Try restarting the application"
        ),
    }

    if category in recovery_suggestions:
        message += recovery_suggestions[category]

    return message + technical_details


def handle_error(error: Exception, show_streamlit_error: bool = True) -> None:
    """
    Handle an exception by logging it and optionally displaying a Streamlit error message.

    Args:
        error: The exception to handle
        show_streamlit_error: Whether to display an error message in the Streamlit UI
    """
    # Categorize the error
    category = categorize_error(error)

    # Log the error with appropriate level based on category
    if category in [
        ErrorCategory.API_CONNECTION,
        ErrorCategory.API_TIMEOUT,
        ErrorCategory.API_RATE_LIMIT,
    ]:
        logger.warning(f"{category.value}: {str(error)}", exc_info=True)
    else:
        logger.error(f"{category.value}: {str(error)}", exc_info=True)

    # Display user-friendly error message in Streamlit if requested
    if show_streamlit_error:
        user_message = get_user_friendly_message(error, category)
        st.error(user_message)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    errors_to_retry: List[Type[Exception]] = None,
):
    """
    Retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for the exponential backoff
        jitter: Whether to add random jitter to the delay
        errors_to_retry: List of exception types to retry on (defaults to [RateLimitError, APITimeoutError, APIConnectionError])

    Returns:
        Callable: Decorated function with retry logic
    """
    if errors_to_retry is None:
        errors_to_retry = [RateLimitError, APITimeoutError, APIConnectionError]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until max retries reached
            while True:
                try:
                    return func(*args, **kwargs)

                except tuple(errors_to_retry) as e:
                    # If we've reached max retries, re-raise the exception
                    num_retries += 1
                    if num_retries > max_retries:
                        logger.warning(
                            f"Maximum retries ({max_retries}) exceeded for function {func.__name__}. "
                            f"Last error: {str(e)}"
                        )
                        raise

                    # Log the retry attempt
                    logger.info(
                        f"Retry {num_retries}/{max_retries} for function {func.__name__} "
                        f"after error: {str(e)}. Waiting {delay:.2f} seconds..."
                    )

                    # Wait with exponential backoff and optional jitter
                    if jitter:
                        import random

                        sleep_time = delay * (1 + random.random())
                    else:
                        sleep_time = delay

                    time.sleep(sleep_time)

                    # Increase the delay for the next iteration
                    delay *= exponential_base

        return wrapper

    return decorator


class ErrorHandlingCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for error handling during LLM calls.

    This handler captures errors during LLM generation and provides
    user-friendly error messages.
    """

    def __init__(self):
        """Initialize the callback handler."""
        super().__init__()
        self.errors = []
        self.has_error = False

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """
        Handle errors that occur during LLM calls.

        Args:
            error: The exception that occurred
            **kwargs: Additional arguments
        """
        self.has_error = True
        self.errors.append(error)

        # Handle the error
        handle_error(error)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """
        Handle errors that occur during chain execution.

        Args:
            error: The exception that occurred
            **kwargs: Additional arguments
        """
        self.has_error = True
        self.errors.append(error)

        # Handle the error
        handle_error(error)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """
        Handle errors that occur during tool execution.

        Args:
            error: The exception that occurred
            **kwargs: Additional arguments
        """
        self.has_error = True
        self.errors.append(error)

        # Handle the error
        handle_error(error)

    def on_retriever_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """
        Handle errors that occur during retrieval.

        Args:
            error: The exception that occurred
            **kwargs: Additional arguments
        """
        self.has_error = True
        self.errors.append(error)

        # Handle the error
        handle_error(error)
