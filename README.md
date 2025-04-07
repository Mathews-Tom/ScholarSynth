# Generative AI Literature Review Assistant

## Description

This project implements a Retrieval-Augmented Generation (RAG) system designed to assist research scientists with literature reviews. Users can ask questions or request summaries based on content retrieved from local PDF documents and articles fetched from ArXiv. The system uses semantic search to find relevant context and then leverages a Large Language Model (LLM) via LangChain to generate answers grounded in the retrieved information.

## Core Features

- **Query Local PDFs:** Ingest and search across PDF documents stored locally.
- **Query ArXiv:** Fetch and ingest recent papers from ArXiv based on specified queries.
- **Semantic Search:** Utilizes embeddings and a vector store (ChromaDB) to find document chunks most relevant to the user's query based on meaning, not just keywords.
- **Grounded Generation:** Uses OpenAI's language models (via LangChain) to answer questions or summarize information based *only* on the context retrieved from the documents.
- **Web Interface:** Provides a simple user interface built with Streamlit for interaction.

## Architecture & Technologies

- **Architecture:** Retrieval-Augmented Generation (RAG)
- **Language:** Python 3.9+
- **Core Framework:** LangChain
- **LLM Provider:** OpenAI
- **Embedding Model:** OpenAI
- **Vector Store:** ChromaDB (persistent local storage)
- **Data Loaders:** PyPDFLoader, ArxivLoader (via LangChain)
- **UI:** Streamlit

## Setup & Installation

1. **Prerequisites:**
    - Python 3.9 or higher installed.
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

1. **Place PDFs:** Put any PDF documents you want to query into the `data/raw_pdfs/` directory (create this directory if it doesn't exist).
2. **Configure ArXiv:** Adjust `ARXIV_QUERY` and `ARXIV_MAX_RESULTS` in your `.env` file to fetch relevant papers from ArXiv.
3. **Run Ingestion Script:** Execute the following command from the project root directory:

    ```bash
    python scripts/ingest.py
    ```

    *(Note: The `scripts/ingest.py` file will be created in a later step).*

    This script will:
    - Load documents from the PDF directory.
    - Fetch documents from ArXiv based on your query.
    - Split the documents into smaller, manageable chunks.
    - Generate embeddings for each chunk using the specified OpenAI model.
    - Store the chunks and their embeddings in the ChromaDB vector store located at `data/vector_store_db/`.

## Usage

Once the data ingestion is complete, you can start the interactive web application:

1. **Run Streamlit:**

    ```bash
    streamlit run app.py
    ```

    *(Note: The `app.py` file will be created in a later step).*

2. **Interact:** Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`). Enter your questions about the ingested literature into the text input box and press Enter or click the submit button. The application will retrieve relevant context and generate an answer based on it.

## Configuration Details (`.env`)

- `OPENAI_API_KEY`: **Required.** Your secret API key from OpenAI.
- `EMBEDDING_MODEL_NAME`: The model used to create vector embeddings for semantic search.
- `LLM_MODEL_NAME`: The language model used to generate answers based on retrieved context.
- `CHROMA_PERSIST_PATH`: The directory where the vector database will be stored locally.
- `PDF_SOURCE_DIR`: The directory where the ingestion script looks for input PDF files.
- `ARXIV_QUERY`: The search query used to fetch papers from ArXiv during ingestion. Follow ArXiv's advanced search syntax.
- `ARXIV_MAX_RESULTS`: The maximum number of papers to download from ArXiv during ingestion.

## Repository Structure

```bash
lit_review_assistant/
├── scripts/
├── retrieval/
├── utils/
├── data/
│   ├── arxiv_pdfs/
│   ├── local_pdfs/
│   └── vector_store_db/
├── scripts/
├── app.py
├── requirements.txt
├── .env-sample
└── .gitignore
```
