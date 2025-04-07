"""
Parent Document Retriever implementation for ScholarSynth.

This module provides a retriever that stores both small chunks and their parent documents.
When retrieving, it finds the most relevant small chunks but returns their parent documents
for better context.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from retrieval.text_splitter import get_parent_child_splitters

# Define logger for this module
logger = logging.getLogger(__name__)


def get_parent_document_retriever(
    vector_store_path: str,
    parent_doc_store_path: str,
    embedding_model: Optional[Embeddings] = None,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 400,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
    search_type: str = "similarity",
    search_kwargs: Optional[Dict] = None,
) -> "ParentDocumentRetriever":
    """
    Initialize and return a ParentDocumentRetriever.

    Args:
        vector_store_path (str): Path to the vector store directory.
        parent_doc_store_path (str): Path to the parent document store directory.
        embedding_model (Optional[Embeddings]): Embedding model to use.
        parent_chunk_size (int): Size of parent chunks in characters.
        parent_chunk_overlap (int): Overlap between parent chunks in characters.
        child_chunk_size (int): Size of child chunks in characters.
        child_chunk_overlap (int): Overlap between child chunks in characters.
        search_type (str): Type of search to use ("similarity" or "mmr").
        search_kwargs (Optional[Dict]): Additional search parameters.

    Returns:
        ParentDocumentRetriever: Initialized retriever.
    """
    # Create directories if they don't exist
    os.makedirs(vector_store_path, exist_ok=True)
    os.makedirs(parent_doc_store_path, exist_ok=True)

    # Get text splitters for parent and child documents
    parent_splitter, child_splitter = get_parent_child_splitters(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )

    # Initialize vector store for child chunks
    vectorstore = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embedding_model,
        collection_name="child_chunks",
    )

    # Initialize document store for parent documents
    docstore = InMemoryDocstore()

    # Set default search kwargs if not provided
    if search_kwargs is None:
        search_kwargs = {"k": 4}

    # Initialize and return the retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        parent_doc_store_path=parent_doc_store_path,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    return retriever


class ParentDocumentRetriever:
    """
    Retriever that uses small chunks for retrieval but returns their parent documents.

    This retriever:
    1. Splits documents into parent chunks
    2. Stores parent chunks in a document store
    3. Further splits parent chunks into child chunks
    4. Embeds and stores child chunks in a vector store with references to their parent chunks
    5. When retrieving, finds the most relevant child chunks but returns their parent documents
    """

    def __init__(
        self,
        vectorstore: Chroma,
        docstore: InMemoryDocstore,
        parent_splitter: any,
        child_splitter: any,
        parent_doc_store_path: str,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the ParentDocumentRetriever.

        Args:
            vectorstore (Chroma): Vector store for child chunks.
            docstore (InMemoryDocstore): Document store for parent documents.
            parent_splitter: Text splitter for creating parent chunks.
            child_splitter: Text splitter for creating child chunks from parent chunks.
            parent_doc_store_path (str): Path to the parent document store directory.
            search_type (str): Type of search to use ("similarity" or "mmr").
            search_kwargs (Optional[Dict]): Additional search parameters.
        """
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.parent_splitter = parent_splitter
        self.child_splitter = child_splitter
        self.parent_doc_store_path = parent_doc_store_path
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {"k": 4}

        # Load existing parent documents if any
        self._load_parent_documents()

    def _load_parent_documents(self) -> None:
        """
        Load existing parent documents from the parent document store directory.
        """
        parent_doc_dir = Path(self.parent_doc_store_path)
        if not parent_doc_dir.exists():
            logger.info(
                f"Parent document store directory '{parent_doc_dir}' does not exist."
            )
            return

        try:
            # Count files in the directory
            doc_files = list(parent_doc_dir.glob("*.txt"))
            if not doc_files:
                logger.info("No parent documents found in the store.")
                return

            logger.info(
                f"Loading {len(doc_files)} parent documents from '{parent_doc_dir}'..."
            )

            # Load each document
            for doc_file in doc_files:
                try:
                    # Extract document ID from filename
                    doc_id = doc_file.stem

                    # Read document content
                    with open(doc_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create document and add to docstore
                    doc = Document(page_content=content)
                    self.docstore.store[doc_id] = doc
                except Exception as e:
                    logger.warning(f"Error loading parent document '{doc_file}': {e}")

            logger.info(
                f"Successfully loaded {len(self.docstore.store)} parent documents."
            )
        except Exception as e:
            logger.error(f"Error loading parent documents: {e}", exc_info=True)

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Process documents and add them to the retriever.

        Args:
            documents (List[Document]): Documents to add.
            ids (Optional[List[str]]): Optional list of IDs for the documents.

        Returns:
            List[str]: List of parent document IDs.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return []

        # Split documents into parent chunks
        logger.info(f"Splitting {len(documents)} documents into parent chunks...")
        parent_documents = self.parent_splitter.split_documents(documents)
        logger.info(f"Created {len(parent_documents)} parent chunks.")

        # Generate IDs for parent documents if not provided
        if ids is None:
            ids = [self._generate_id(doc) for doc in parent_documents]

        # Process each parent document
        all_child_docs = []
        parent_doc_ids = []

        for i, (parent_doc, doc_id) in enumerate(zip(parent_documents, ids)):
            try:
                # Store parent document
                self.docstore.store[doc_id] = parent_doc
                parent_doc_ids.append(doc_id)

                # Save parent document to disk
                self._save_parent_document(doc_id, parent_doc)

                # Split parent into child chunks
                child_docs = self.child_splitter.split_documents([parent_doc])

                # Add parent_id to each child's metadata
                for child_doc in child_docs:
                    child_doc.metadata["parent_id"] = doc_id

                all_child_docs.extend(child_docs)

                # Log progress for large document sets
                if (i + 1) % 100 == 0 or i + 1 == len(parent_documents):
                    logger.info(
                        f"Processed {i + 1}/{len(parent_documents)} parent documents."
                    )
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}", exc_info=True)

        # Add child documents to vector store
        if all_child_docs:
            logger.info(f"Adding {len(all_child_docs)} child chunks to vector store...")
            self.vectorstore.add_documents(all_child_docs)
            logger.info("Successfully added child chunks to vector store.")

        return parent_doc_ids

    def _generate_id(self, document: Document) -> str:
        """
        Generate a deterministic ID for a document based on its content and metadata.

        Args:
            document (Document): Document to generate ID for.

        Returns:
            str: Generated ID.
        """
        # Combine content and key metadata fields for hashing
        content = document.page_content
        metadata_str = ""

        # Add key metadata fields if available
        if document.metadata:
            if "source" in document.metadata:
                metadata_str += document.metadata["source"]
            if "page" in document.metadata:
                metadata_str += str(document.metadata["page"])

        # Generate hash
        text_to_hash = (content + metadata_str).encode("utf-8")
        return hashlib.md5(text_to_hash).hexdigest()

    def _save_parent_document(self, doc_id: str, document: Document) -> None:
        """
        Save a parent document to disk.

        Args:
            doc_id (str): Document ID.
            document (Document): Document to save.
        """
        try:
            parent_doc_dir = Path(self.parent_doc_store_path)
            parent_doc_dir.mkdir(parents=True, exist_ok=True)

            file_path = parent_doc_dir / f"{doc_id}.txt"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(document.page_content)

            logger.debug(f"Saved parent document to {file_path}")
        except Exception as e:
            logger.error(f"Error saving parent document {doc_id}: {e}", exc_info=True)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to the query.

        Args:
            query (str): Query string.

        Returns:
            List[Document]: List of relevant parent documents.
        """
        # Search for relevant child chunks
        if self.search_type == "mmr":
            child_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:  # Default to similarity search
            child_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # Get unique parent IDs
        parent_ids = set()
        for doc in child_docs:
            if "parent_id" in doc.metadata:
                parent_ids.add(doc.metadata["parent_id"])

        # Retrieve parent documents
        parent_docs = []
        for parent_id in parent_ids:
            if parent_id in self.docstore.store:
                parent_docs.append(self.docstore.store[parent_id])
            else:
                logger.warning(
                    f"Parent document with ID '{parent_id}' not found in docstore."
                )

        return parent_docs

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
