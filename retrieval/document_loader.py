"""
Document loading utilities for ScholarSynth.

This module provides functions for loading documents from various sources,
including local PDFs and ArXiv papers.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import arxiv
from langchain_community.document_loaders import (
    ArxivLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

# Define logger for this module
logger = logging.getLogger(__name__)


def load_pdfs(directory_path: str) -> List[Document]:
    """
    Load PDF and text documents from a directory.

    Args:
        directory_path (str): Path to the directory containing PDF and text files.

    Returns:
        List[Document]: List of loaded documents.
    """
    if not directory_path:
        logger.warning("No directory path provided.")
        return []

    # Convert to Path object and resolve to absolute path
    doc_dir = Path(directory_path).resolve()

    if not doc_dir.exists():
        logger.warning(f"Directory '{doc_dir}' does not exist.")
        return []

    if not doc_dir.is_dir():
        logger.warning(f"'{doc_dir}' is not a directory.")
        return []

    documents = []

    try:
        # Load PDF files
        pdf_loader = DirectoryLoader(
            str(doc_dir),
            glob="**/*.pdf",  # Include PDFs in subdirectories
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        pdf_documents = pdf_loader.load()
        logger.info(
            f"Successfully loaded {len(pdf_documents)} pages from PDFs in '{doc_dir}'."
        )
        documents.extend(pdf_documents)
    except Exception as e:
        logger.error(f"Error loading PDFs from '{doc_dir}': {e}", exc_info=True)

    try:
        # Load text files
        text_loader = DirectoryLoader(
            str(doc_dir),
            glob="**/*.txt",  # Include text files in subdirectories
            loader_cls=TextLoader,
            show_progress=True,
        )
        text_documents = text_loader.load()
        logger.info(
            f"Successfully loaded {len(text_documents)} text documents from '{doc_dir}'."
        )
        documents.extend(text_documents)
    except Exception as e:
        logger.error(f"Error loading text files from '{doc_dir}': {e}", exc_info=True)

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def load_arxiv_papers(
    query: Optional[str] = None,
    max_results: int = 10,
    sort_by: str = arxiv.SortCriterion.Relevance,
) -> List[Document]:
    """
    Load papers from ArXiv based on a search query.

    Args:
        query (Optional[str]): ArXiv search query. If None, no papers are loaded.
        max_results (int): Maximum number of papers to load.
        sort_by (str): Sorting criterion for results.

    Returns:
        List[Document]: List of loaded documents.
    """
    if not query:
        logger.info("No ArXiv query provided. Skipping ArXiv document loading.")
        return []

    try:
        # Use LangChain's ArxivLoader
        loader = ArxivLoader(
            query=query,
            load_max_docs=max_results,
            load_all_available_meta=True,
        )
        documents = loader.load()
        logger.info(
            f"Successfully loaded {len(documents)} documents from ArXiv with query '{query}'."
        )
        return documents
    except Exception as e:
        logger.error(
            f"Error loading ArXiv papers with query '{query}': {e}", exc_info=True
        )
        return []
