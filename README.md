# ScholarSynth: Generative AI Literature Review Assistant

## Description

ScholarSynth implements a Retrieval-Augmented Generation (RAG) system designed to assist research scientists with literature reviews. Users can ask questions or request summaries based on content retrieved from local PDF documents and articles fetched from ArXiv. The system uses semantic search to find relevant context and then leverages a Large Language Model (LLM) via LangChain to generate answers grounded in the retrieved information.

## Core Features

- **Query Local PDFs:** Ingest and search across PDF documents stored locally.
- **Query ArXiv:** Fetch and ingest recent papers from ArXiv based on specified queries.
- **Semantic Search:** Utilizes embeddings and a vector store (ChromaDB) to find document chunks most relevant to the user's query based on meaning, not just keywords.
- **Advanced RAG with Parent Document Retriever:** Supports an advanced retrieval technique that uses smaller chunks for precise retrieval while providing larger parent documents for better context.
- **Grounded Generation:** Uses OpenAI's language models (via LangChain) to answer questions or summarize information based *only* on the context retrieved from the documents.
- **Web Interface:** Provides a simple user interface built with Streamlit for interaction.

## Architecture & Technologies

- **Architecture:** Retrieval-Augmented Generation (RAG) with Parent Document Retriever
- **Language:** Python 3.12+
- **Core Framework:** LangChain
- **LLM Provider:** OpenAI
- **Embedding Model:** OpenAI
- **Vector Store:** ChromaDB (persistent local storage)
- **Document Store:** InMemoryDocstore for parent documents
- **Data Loaders:** PyPDFLoader, TextLoader, ArxivLoader (via LangChain)
- **UI:** Streamlit

## Setup & Installation

1. **Prerequisites:**
    - Python 3.12 or higher installed.
    - `git` installed (for cloning).

2. **Clone Repository:**

    ```bash
    git clone <your-repository-url> # Replace with your actual repo URL if applicable
    cd lit_review_assistant
    ```

3. **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment:

    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

4. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: The `requirements.txt` file will be created in a later step).*

5. **Set Up Environment Variables:**
    - Locate the `.env-sample` file in the project root directory.
    - **Copy** this file and rename the copy to `.env`.

        ```bash
        # Linux/macOS
        cp .env-sample .env

        # Windows (Command Prompt)
        copy .env-sample .env
        ```

    - Open the newly created `.env` file
    - **Crucially, replace placeholder values (especially `YOUR_API_KEY_HERE` for `OPENAI_API_KEY`) with your actual credentials and adjust other settings like ArXiv queries if needed.** The comments within the `.env` file provide guidance on each variable.
    - **Important:** Ensure `.env` is added to your `.gitignore` file to prevent accidentally committing your API keys or secrets. (We will create `.gitignore` next).

## Data Ingestion

Before running the main application, you need to process your source documents (PDFs and ArXiv papers) and store their embeddings in the vector database.

1. **Place PDFs:** Put any PDF documents you want to query into the `data/local_pdfs/` directory (create this directory if it doesn't exist).
2. **Configure ArXiv:** Adjust `ARXIV_QUERY` and `ARXIV_MAX_RESULTS` in your `.env` file to fetch relevant papers from ArXiv.
3. **Run Ingestion Script:** Execute one of the following commands from the project root directory:

    ```bash
    # Standard ingestion (single chunk size)
    python scripts/ingest.py

    # Advanced ingestion with Parent Document Retriever
    python scripts/ingest.py --use-parent-retriever
    ```

    To clear existing data and start fresh, add the `--clear` flag:

    ```bash
    python scripts/ingest.py --clear
    # or
    python scripts/ingest.py --use-parent-retriever --clear
    ```

    This script will:
    - Load documents from the PDF directory and text files.
    - Fetch documents from ArXiv based on your query.
    - Split the documents into chunks (with different strategies depending on the ingestion method).
    - Generate embeddings for each chunk using the specified OpenAI model.
    - Store the chunks and their embeddings in the ChromaDB vector store.
    - If using Parent Document Retriever, also store parent documents in a separate document store.

## Usage

Once the data ingestion is complete, you can start the interactive web application:

1. **Run Streamlit:**

    ```bash
    streamlit run app.py
    ```

2. **Configure Retrieval Settings:** In the sidebar, you can adjust various settings:
   - **Retriever Type:** Choose between "standard" or "parent_document" retriever
   - **Number of chunks to retrieve (k):** Adjust how many document chunks to retrieve
   - **Search Type:** Choose between "similarity" or "MMR" (Maximal Marginal Relevance) search

3. **Interact:** Enter your questions about the ingested literature into the text input box and press Enter or click the submit button. The application will retrieve relevant context and generate an answer based on it.

## Configuration Details (`.env`)

### Basic Configuration

- `OPENAI_API_KEY`: **Required.** Your secret API key from OpenAI.
- `EMBEDDING_MODEL_NAME`: The model used to create vector embeddings for semantic search.
- `LLM_MODEL_NAME`: The language model used to generate answers based on retrieved context.
- `CHROMA_PERSIST_PATH`: The directory where the vector database will be stored locally.
- `PDF_SOURCE_DIR`: The directory where the ingestion script looks for input PDF files.
- `ARXIV_QUERY`: The search query used to fetch papers from ArXiv during ingestion. Follow ArXiv's advanced search syntax.
- `ARXIV_MAX_RESULTS`: The maximum number of papers to download from ArXiv during ingestion.

### Parent Document Retriever Configuration

- `PARENT_DOC_STORE_PATH`: Directory where parent documents are stored (default: "data/parent_doc_store").
- `PARENT_CHUNK_SIZE`: Size of parent document chunks in characters (default: 2000).
- `PARENT_CHUNK_OVERLAP`: Overlap between parent chunks in characters (default: 400).
- `CHILD_CHUNK_SIZE`: Size of child document chunks in characters (default: 500).
- `CHILD_CHUNK_OVERLAP`: Overlap between child chunks in characters (default: 100).

## Repository Structure

```bash
ScholarSynth/
├── scripts/
│   └── ingest.py
├── retrieval/
│   ├── document_loader.py
│   ├── embedding_config.py
│   ├── parent_document_retriever.py
│   ├── text_splitter.py
│   └── vector_store.py
├── utils/
│   ├── config_loader.py
│   └── logging_config.py
├── data/
│   ├── local_pdfs/
│   ├── parent_doc_store/
│   └── vector_store_db/
├── app.py
├── requirements.txt
├── .env-sample
└── .gitignore
```

## Advanced RAG with Parent Document Retriever

ScholarSynth implements an advanced RAG technique called Parent Document Retriever that combines the benefits of:

1. **Precise Retrieval**: Uses small chunks (e.g., 500 characters) for more accurate retrieval
2. **Rich Context**: Provides the LLM with larger parent documents (e.g., 2000 characters) that contain the retrieved chunks

This approach helps solve a common problem in RAG systems: the tradeoff between retrieval precision and context richness.

### How It Works

The Parent Document Retriever works in two stages:

1. **Ingestion Stage**:
   - Documents are split into larger "parent" chunks
   - Parent chunks are stored in a document store
   - Parent chunks are further split into smaller "child" chunks
   - Child chunks are embedded and stored in the vector store with references to their parent chunks

2. **Retrieval Stage**:
   - When a query is made, the system finds the most relevant child chunks
   - Instead of returning these child chunks, it returns their parent documents
   - The LLM receives the broader context from the parent documents

To use this feature, run the ingestion with the `--use-parent-retriever` flag and select "parent_document" as the Retriever type in the UI.
