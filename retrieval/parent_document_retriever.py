"""
Parent Document Retriever implementation for advanced RAG.

This module implements a retriever that stores both small chunks (for retrieval) and
their parent documents (for context). This approach allows for more precise retrieval
while providing broader context to the LLM.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter

from retrieval.embedding_config import get_embedding_model
from retrieval.text_splitter import get_parent_child_splitters
from retrieval.vector_store import get_vector_store

# Define logger for this module
logger = logging.getLogger(__name__)


class ParentDocumentRetriever:
    """
    Retriever that uses small chunks for retrieval but returns their parent documents for context.

    This retriever stores documents in two forms:
    1. Small chunks for precise retrieval (stored in a vector store)
    2. Larger parent documents for context (stored in a document store)

    When a query is made, the retriever:
    1. Finds the most relevant small chunks
    2. Returns the parent documents of those chunks for broader context
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        docstore: InMemoryDocstore,
        parent_splitter: TextSplitter,
        child_splitter: TextSplitter,
        id_key: str = "doc_id",
        search_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the ParentDocumentRetriever.

        Args:
            vectorstore (VectorStore): Vector store for storing and retrieving child chunks.
            docstore (InMemoryDocstore): Document store for storing parent documents.
            parent_splitter (TextSplitter): Splitter for creating parent documents.
            child_splitter (TextSplitter): Splitter for creating child chunks.
            id_key (str): Metadata key used to match child chunks to parent documents.
            search_kwargs (Optional[Dict]): Search parameters for the vector store.
        """
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.parent_splitter = parent_splitter
        self.child_splitter = child_splitter
        self.id_key = id_key
        self.search_kwargs = search_kwargs or {"k": 5}
        logger.info(
            f"Initialized ParentDocumentRetriever with search_kwargs={self.search_kwargs}"
        )

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Process documents and add them to the retriever.

        This method:
        1. Splits documents into parent chunks
        2. Stores parent chunks in the document store
        3. Splits parent chunks into child chunks
        4. Stores child chunks in the vector store with references to their parents

        Args:
            documents (List[Document]): Documents to add.
            **kwargs: Additional arguments to pass to the vector store.

        Returns:
            List[str]: List of IDs of the added documents.
        """
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return []

        # First, split documents into parent documents
        logger.info(f"Splitting {len(documents)} documents into parent chunks...")
        parent_documents = self.parent_splitter.split_documents(documents)
        logger.info(f"Created {len(parent_documents)} parent chunks.")

        # Generate IDs for parent documents and add them to the docstore
        doc_ids = []
        for doc in parent_documents:
            # Generate a deterministic ID based on content
            doc_id = self._generate_id(doc)
            doc.metadata[self.id_key] = doc_id
            self.docstore.add({doc_id: doc})
            doc_ids.append(doc_id)

        # Now, split parent documents into child documents
        logger.info(
            f"Splitting {len(parent_documents)} parent chunks into child chunks..."
        )
        child_documents = []
        for parent_doc in parent_documents:
            # Get the parent document's ID
            doc_id = parent_doc.metadata[self.id_key]

            # Split the parent into children
            children = self.child_splitter.split_documents([parent_doc])

            # Add the parent's ID to each child's metadata
            for child in children:
                child.metadata[self.id_key] = doc_id

            child_documents.extend(children)

        logger.info(f"Created {len(child_documents)} child chunks.")

        # Add child documents to the vector store
        logger.info(f"Adding {len(child_documents)} child chunks to vector store...")
        self.vectorstore.add_documents(child_documents, **kwargs)
        logger.info("Successfully added child chunks to vector store.")

        return doc_ids

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        This method:
        1. Retrieves the most relevant child chunks from the vector store
        2. Looks up the parent documents of those chunks
        3. Returns the parent documents for broader context

        Args:
            query (str): Query string.

        Returns:
            List[Document]: List of relevant parent documents.
        """
        # Search for relevant child documents
        child_documents = self.vectorstore.similarity_search(
            query, **self.search_kwargs
        )

        # Get the unique parent document IDs
        parent_ids = list(set(doc.metadata[self.id_key] for doc in child_documents))
        logger.info(
            f"Retrieved {len(child_documents)} child chunks with {len(parent_ids)} unique parent documents."
        )

        # Retrieve the parent documents
        parent_documents = []
        for doc_id in parent_ids:
            if doc_id in self.docstore.store:
                parent_documents.append(self.docstore.store[doc_id])
            else:
                logger.warning(
                    f"Parent document with ID {doc_id} not found in docstore."
                )

        return parent_documents

    def _generate_id(self, doc: Document) -> str:
        """
        Generate a deterministic ID for a document based on its content and metadata.

        Args:
            doc (Document): The document to generate an ID for.

        Returns:
            str: A unique ID for the document.
        """
        # Create a string that combines content and key metadata
        content_str = doc.page_content

        # Add important metadata if available
        if "source" in doc.metadata:
            content_str += doc.metadata["source"]
        if "page" in doc.metadata:
            content_str += str(doc.metadata["page"])

        # Generate a hash
        return hashlib.md5(content_str.encode()).hexdigest()

    def get_parent_document_count(self) -> int:
        """
        Get the number of parent documents in the docstore.

        Returns:
            int: Number of parent documents.
        """
        try:
            # Try different ways to access the document count based on the docstore implementation
            if hasattr(self.docstore, "store"):
                return len(self.docstore.store)
            elif hasattr(self.docstore, "_dict"):
                return len(self.docstore._dict)
            elif hasattr(self.docstore, "__len__"):
                return len(self.docstore)
            else:
                logger.warning(
                    "Could not determine parent document count. Unknown docstore implementation."
                )
                return 0
        except Exception as e:
            logger.warning(f"Error getting parent document count: {e}")
            return 0

    def as_retriever(self, **kwargs):
        """
        Return self as a retriever, allowing this class to be used directly in LangChain chains.

        Args:
            **kwargs: Additional arguments to override search parameters.

        Returns:
            ParentDocumentRetriever: Self, but with potentially updated search parameters.
        """
        if kwargs:
            # Update search parameters if provided
            search_kwargs = {**self.search_kwargs, **kwargs}
            self.search_kwargs = search_kwargs
            logger.info(f"Updated search parameters: {search_kwargs}")
        return self


def get_parent_document_retriever(
    vector_store_path: str,
    parent_doc_store_path: str,
    embedding_model: Optional[Embeddings] = None,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 400,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
    search_kwargs: Optional[Dict] = None,
) -> ParentDocumentRetriever:
    """
    Create and initialize a ParentDocumentRetriever with the specified parameters.

    Args:
        vector_store_path (str): Path to the vector store for child chunks.
        parent_doc_store_path (str): Path to store parent documents.
        embedding_model (Optional[Embeddings]): Embedding model to use.
        parent_chunk_size (int): Size of parent document chunks.
        parent_chunk_overlap (int): Overlap between parent chunks.
        child_chunk_size (int): Size of child document chunks.
        child_chunk_overlap (int): Overlap between child chunks.
        search_kwargs (Optional[Dict]): Search parameters for retrieval.

    Returns:
        ParentDocumentRetriever: Initialized retriever.
    """
    # Get embedding model if not provided
    if embedding_model is None:
        embedding_model = get_embedding_model()

    # Get vector store for child chunks
    vector_store = get_vector_store(
        persist_directory_str=vector_store_path,
        embedding_function=embedding_model,
    )

    # Create parent and child splitters
    parent_splitter, child_splitter = get_parent_child_splitters(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )

    # Create document store for parent documents
    docstore = InMemoryDocstore({})

    # Set default search parameters if not provided
    if search_kwargs is None:
        search_kwargs = {"k": 5}

    # Create and return the retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs=search_kwargs,
    )

    logger.info(
        f"Created ParentDocumentRetriever with parent_chunk_size={parent_chunk_size}, "
        f"child_chunk_size={child_chunk_size}, search_kwargs={search_kwargs}"
    )

    return retriever
