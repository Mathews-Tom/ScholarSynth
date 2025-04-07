"""
Text splitting utilities for document processing.

This module provides functions for splitting documents into chunks of appropriate size
for embedding and retrieval. It supports both standard text splitting and parent-child
document splitting for advanced RAG techniques.
"""

import logging
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define logger for this module
logger = logging.getLogger(__name__)


def get_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Creates and returns a RecursiveCharacterTextSplitter with the specified parameters.

    Args:
        chunk_size (int): The target size of each text chunk (in characters).
        chunk_overlap (int): The number of characters of overlap between chunks.
        separators (Optional[List[str]]): Custom separators for splitting text.
                                         If None, default separators are used.

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter instance.
    """
    # Default separators if none provided
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    logger.info(
        f"Initializing RecursiveCharacterTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def get_parent_child_splitters(
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 400,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> tuple:
    """
    Creates and returns a pair of text splitters for parent and child documents.
    
    The parent splitter creates larger chunks that provide broader context,
    while the child splitter creates smaller chunks for more precise retrieval.

    Args:
        parent_chunk_size (int): Size of parent document chunks (in characters).
        parent_chunk_overlap (int): Overlap between parent chunks.
        child_chunk_size (int): Size of child document chunks (in characters).
        child_chunk_overlap (int): Overlap between child chunks.
        separators (Optional[List[str]]): Custom separators for splitting text.
                                         If None, default separators are used.

    Returns:
        tuple: (parent_splitter, child_splitter) - Configured text splitter instances.
    """
    # Default separators if none provided
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    logger.info(
        f"Initializing parent splitter with chunk_size={parent_chunk_size}, "
        f"chunk_overlap={parent_chunk_overlap}"
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=separators,
    )

    logger.info(
        f"Initializing child splitter with chunk_size={child_chunk_size}, "
        f"chunk_overlap={child_chunk_overlap}"
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=separators,
    )

    return parent_splitter, child_splitter


def split_documents(
    documents: List[Document],
    text_splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Splits a list of documents into chunks using the provided text splitter.

    Args:
        documents (List[Document]): List of LangChain Document objects to split.
        text_splitter (RecursiveCharacterTextSplitter): The text splitter to use.

    Returns:
        List[Document]: List of split document chunks.
    """
    if not documents:
        logger.warning("No documents provided to split_documents.")
        return []

    # Split the documents
    split_docs = text_splitter.split_documents(documents)
    
    logger.info(
        f"Split {len(documents)} source documents into {len(split_docs)} chunks."
    )
    
    return split_docs
