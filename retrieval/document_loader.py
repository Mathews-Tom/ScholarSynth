import logging
import shutil  # Keep shutil for potential directory removal
from pathlib import Path  # Use pathlib for path operations
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# External libraries
import arxiv  # ArXiv API client
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from tqdm.auto import tqdm  # Progress bar

# Internal imports
from utils.config_loader import load_config

# Define logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
PDF_SOURCE_DIR_STR: Optional[str] = None  # Store config value as string initially
ARXIV_QUERY: Optional[str] = None
ARXIV_MAX_RESULTS: Optional[int] = None

try:
    config: SimpleNamespace = load_config()
    PDF_SOURCE_DIR_STR = config.PDF_SOURCE_DIR
    ARXIV_QUERY = config.ARXIV_QUERY
    ARXIV_MAX_RESULTS = config.ARXIV_MAX_RESULTS

    # Basic path validation (using pathlib)
    pdf_source_path: Optional[Path] = None
    if PDF_SOURCE_DIR_STR:
        pdf_source_path = Path(PDF_SOURCE_DIR_STR).resolve()  # Resolve to absolute path
        if not pdf_source_path.is_dir():
            logger.warning(
                f"PDF_SOURCE_DIR '{pdf_source_path}' not found or is not a directory."
            )
            pdf_source_path = None  # Invalidate path if not a directory
    else:
        logger.warning("PDF_SOURCE_DIR not configured.")

    if not ARXIV_QUERY:
        logger.warning("ARXIV_QUERY not configured. ArXiv loading will be skipped.")
    if ARXIV_MAX_RESULTS is None or ARXIV_MAX_RESULTS <= 0:
        logger.warning("ARXIV_MAX_RESULTS not configured or invalid. Defaulting to 10.")
        ARXIV_MAX_RESULTS = 10

except (ValueError, AttributeError, Exception) as e:
    logger.exception("Configuration error during document loader setup.", exc_info=True)
    logger.error(
        "Proceeding with potentially incomplete configuration for document loading."
    )
    # Reset potentially invalid path from config error
    pdf_source_path = None


# --- PDF Loading Function (using pathlib) ---
def load_pdfs(source_dir_path: Optional[Path]) -> List[Document]:
    """
    Loads all PDF files from the specified directory using PyPDFLoader and pathlib.

    Args:
        source_dir_path (Optional[Path]): A Path object representing the absolute path
                                            to the directory containing PDF files.
                                            If None or not a valid directory, returns an empty list.

    Returns:
        List[Document]: A list of LangChain Document objects, each representing a page
                        from the loaded PDFs. Returns empty list if directory is invalid
                        or contains no PDFs.
    """
    if not source_dir_path or not source_dir_path.is_dir():
        logger.warning(
            f"Invalid or missing PDF source directory: {source_dir_path}. Skipping PDF loading."
        )
        return []

    # Use pathlib's glob to find PDF files (returns a generator)
    pdf_files: List[Path] = list(source_dir_path.glob("*.pdf"))

    if not pdf_files:
        logger.info(f"No PDF files found in directory: {source_dir_path}")
        return []

    logger.info(
        f"Found {len(pdf_files)} PDF files in {source_dir_path}. Starting loading process..."
    )
    all_docs: List[Document] = []
    loaded_count: int = 0
    failed_count: int = 0

    for pdf_path_obj in pdf_files:
        try:
            # PyPDFLoader generally expects a string path
            pdf_path_str = str(pdf_path_obj)
            loader = PyPDFLoader(pdf_path_str, extract_images=False)
            docs: List[Document] = loader.load()
            logger.debug(
                f"Loaded {len(docs)} pages from '{pdf_path_obj.name}'"
            )  # Use path object's name property

            # Add string representation of path to metadata
            for doc in docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = pdf_path_str
                # Optionally add just the filename
                doc.metadata["filename"] = pdf_path_obj.name
            all_docs.extend(docs)
            loaded_count += 1
        except Exception as e:
            # Log using the path object's name for clarity
            logger.error(
                f"Error loading PDF file '{pdf_path_obj.name}': {e}",
                exc_info=False,  # Set True for detailed traceback if needed
            )
            failed_count += 1

    logger.info(
        f"Finished loading local PDFs. Successfully loaded: {loaded_count}. Failed: {failed_count}."
    )
    return all_docs


# --- ArXiv Loading Function ---
def load_arxiv_papers(
    query: Optional[str],
    max_results: Optional[int],
    download_dir_name: str = "data/arxiv_pdfs",  # Just the relative directory name
) -> List[Document]:
    """
    Searches ArXiv, downloads PDFs (if needed) using pathlib, loads them, adds metadata,
    and displays progress using tqdm.

    Args:
        query (Optional[str]): The search query for ArXiv.
        max_results (Optional[int]): Max papers to fetch metadata for.
        download_dir_name (str): The relative directory name within the project to store PDFs.
                                Defaults to 'data/arxiv_pdfs'.

    Returns:
        List[Document]: List of Document objects from processed ArXiv papers.
    """
    if not query or not max_results or max_results <= 0:
        logger.warning("ArXiv query/max_results invalid. Skipping ArXiv loading.")
        return []

    # Determine project root and full download path using pathlib
    try:
        # Resolve finds the absolute path, parent gets the directory containing this file (retrieval)
        # parent again gets the project root
        project_root = Path(__file__).resolve().parent.parent
        full_download_path = project_root / download_dir_name
        full_download_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using ArXiv download directory: {full_download_path}")
    except OSError as e:
        logger.error(f"Could not create ArXiv directory '{full_download_path}': {e}.")
        return []
    except Exception as e:  # Catch potential errors resolving path
        logger.error(f"Could not determine project/download path: {e}")
        return []

    all_docs: List[Document] = []
    processed_count: int = 0
    failed_count: int = 0
    skipped_download_count: int = 0

    try:
        logger.info(f"Searching ArXiv: query='{query}', max_results={max_results}")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        # Fetch results eagerly to get the total count for tqdm
        results = list(client.results(search))

        if not results:
            logger.info("No results found on ArXiv.")
            return []

        logger.info(
            f"Found {len(results)} ArXiv papers. Checking local cache and processing..."
        )

        # --- Wrap the main processing loop with tqdm ---
        # Create the tqdm iterator
        result_iterator = tqdm(
            results,  # Iterate through the fetched results list
            desc="Processing ArXiv papers",
            total=len(results),  # Use the length of the fetched results
            unit="paper",
            ncols=100,  # Adjust width as needed
            leave=True,  # Keep the bar on screen after completion
        )

        for result in result_iterator:
            # --- End tqdm wrapper ---

            # Filename generation using pathlib compatible strings
            entry_id_slug = result.get_short_id().replace("/", "-").replace(".", "_")
            pdf_filename = f"{entry_id_slug}.pdf"
            pdf_path: Path = full_download_path / pdf_filename

            # logger.debug(...) # Reduced logging frequency as tqdm shows progress

            try:
                # --- Download Step (Conditional) ---
                download_attempted = False
                if pdf_path.exists():
                    # Log less verbosely now
                    # logger.info(f"Skipping download, PDF exists: '{pdf_path.name}'")
                    skipped_download_count += 1
                else:
                    # Optional: Update tqdm description dynamically before download
                    # result_iterator.set_description(f"Downloading {pdf_path.name}", refresh=True)
                    logger.debug(f"Attempting download: '{pdf_path.name}'...")
                    download_attempted = True
                    result.download_pdf(
                        dirpath=str(full_download_path), filename=pdf_filename
                    )
                    logger.debug(f"Successfully downloaded '{pdf_path.name}'")
                    # Restore description after download if changed
                    # result_iterator.set_description("Processing ArXiv papers", refresh=True)

                # --- Loading Step ---
                if not pdf_path.exists():
                    logger.error(
                        f"File not found after check/download: '{pdf_path.name}'"
                    )
                    raise FileNotFoundError(f"File not found: {pdf_path}")

                # Optional: Update description before loading
                # result_iterator.set_description(f"Loading {pdf_path.name}", refresh=True)
                loader = PyPDFLoader(str(pdf_path), extract_images=False)
                docs_from_paper: List[Document] = loader.load()
                # Restore description after loading if changed
                # result_iterator.set_description("Processing ArXiv papers", refresh=True)

                # --- Metadata Step ---
                # Ensure all added metadata values are primitive types (str, int, float, bool)
                arxiv_metadata: Dict[str, Any] = {
                    "source": f"arxiv:{result.entry_id}",
                    "arxiv_id": result.entry_id,
                    "title": str(result.title) if result.title else "N/A",
                    "authors": ", ".join(str(a) for a in result.authors),
                    "published_date": result.published.strftime("%Y-%m-%d")
                    if result.published
                    else None,
                    "summary": str(result.summary) if result.summary else "",
                    "pdf_url": str(result.pdf_url) if result.pdf_url else "",
                    "categories": ", ".join(result.categories),
                    "filename": pdf_path.name,
                    "local_path": str(pdf_path.relative_to(project_root)),
                }
                # Filter out None values before updating metadata, as Chroma might disallow them
                filtered_metadata = {
                    k: v for k, v in arxiv_metadata.items() if v is not None
                }

                for doc in docs_from_paper:
                    doc.metadata.update(filtered_metadata)

                all_docs.extend(docs_from_paper)
                processed_count += 1

                # Optional: Update tqdm postfix with counts
                result_iterator.set_postfix(
                    Processed=processed_count,
                    Skipped=skipped_download_count,
                    Failed=failed_count,
                    refresh=True,
                )

            # --- Exception Handling for this paper ---
            except FileNotFoundError as fnf_err:
                logger.error(
                    f"File not found error processing {result.entry_id} ('{pdf_path.name}'): {fnf_err}"
                )
                failed_count += 1
                # Optional: Update tqdm postfix
                result_iterator.set_postfix(
                    Processed=processed_count,
                    Skipped=skipped_download_count,
                    Failed=failed_count,
                    refresh=True,
                )
            except arxiv.arxiv.UnexpectedEmptyPageError as empty_page_err:
                logger.error(
                    f"ArXiv download error for {result.entry_id} ('{pdf_path.name}'): {empty_page_err}"
                )
                failed_count += 1
                if download_attempted and pdf_path.exists():
                    try:
                        pdf_path.unlink()  # Use Path object's unlink method
                        logger.info(
                            f"Deleted potentially corrupt file: '{pdf_path.name}'"
                        )
                    except OSError as del_err:
                        logger.warning(
                            f"Could not delete corrupt file '{pdf_path.name}': {del_err}"
                        )
                # Optional: Update tqdm postfix
                result_iterator.set_postfix(
                    Processed=processed_count,
                    Skipped=skipped_download_count,
                    Failed=failed_count,
                    refresh=True,
                )
            except Exception as e:
                logger.error(
                    f"Failed to process ArXiv paper {result.entry_id} ('{pdf_path.name}'): {e}",
                    exc_info=False,  # Set True for more detail if needed
                )
                failed_count += 1
                # Optional: Update tqdm postfix
                result_iterator.set_postfix(
                    Processed=processed_count,
                    Skipped=skipped_download_count,
                    Failed=failed_count,
                    refresh=True,
                )

    except Exception as e:
        logger.error(f"Error during ArXiv search/setup: {e}", exc_info=True)
        logger.warning("Processing stopped due to ArXiv search/setup error.")

    # Final Summary Log
    logger.info("--- ArXiv Processing Summary ---")
    logger.info(f"Successfully processed: {processed_count} papers.")
    logger.info(
        f"Skipped downloading (already existed): {skipped_download_count} papers."
    )
    logger.info(f"Failed during processing: {failed_count} papers.")
    logger.info(f"Total documents generated: {len(all_docs)} pages.")
    logger.info("--------------------------------")
    return all_docs


# --- Direct Execution Block (updated for pathlib) ---
if __name__ == "__main__":
    # Note: config loading happens at module level, including pdf_source_path validation
    # setup_logging() should be called *before* any logging happens, including config load warnings
    # Placing it here means initial config load errors might not be logged via our setup
    # A better pattern is to have a separate entry point function or setup logging ASAP
    from utils.logging_config import setup_logging

    setup_logging()

    logger.info("--- Testing document_loader.py ---")

    # --- Test PDF Loading ---
    logger.info("--- Testing PDF Loading ---")
    # pdf_source_path is already loaded and validated (as Path object or None) at module level
    pdf_dir_to_test_path = pdf_source_path

    if pdf_dir_to_test_path and not pdf_dir_to_test_path.exists():
        try:
            pdf_dir_to_test_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created test PDF directory: {pdf_dir_to_test_path}")
            # Example dummy file creation with pathlib:
            # dummy_pdf = pdf_dir_to_test_path / "dummy.pdf"
            # dummy_pdf.write_text("%PDF-1.0\n...") # Use write_text or write_bytes
        except OSError as e:
            logger.error(
                f"Could not create test PDF directory {pdf_dir_to_test_path}: {e}"
            )
            pdf_dir_to_test_path = None  # Don't proceed if creation failed

    try:
        # Pass the Path object (or None) to the function
        pdf_docs = load_pdfs(pdf_dir_to_test_path)
        logger.info(f"Loaded {len(pdf_docs)} pages from local PDFs.")
        if pdf_docs:
            logger.debug(f"First PDF page metadata: {pdf_docs[0].metadata}")
            logger.debug(
                f"First PDF page content preview: {pdf_docs[0].page_content[:100]}..."
            )
    except Exception as e:
        logger.exception("Error testing PDF loading", exc_info=True)

    # --- Test ArXiv Loading ---
    logger.info("--- Testing ArXiv Loading ---")
    try:
        test_max_results = 2  # Keep test small
        logger.info(
            f"Testing ArXiv with query='{ARXIV_QUERY}', max_results={test_max_results}"
        )
        # Pass the relative dir name
        arxiv_docs = load_arxiv_papers(
            ARXIV_QUERY, test_max_results, download_dir_name="data/arxiv_pdfs"
        )
        logger.info(f"Loaded {len(arxiv_docs)} pages from ArXiv papers.")
        if arxiv_docs:
            logger.debug(f"First ArXiv page metadata: {arxiv_docs[0].metadata}")
            logger.debug(
                f"First ArXiv page content preview: {arxiv_docs[0].page_content[:100]}..."
            )
    except Exception as e:
        logger.exception("Error testing ArXiv loading", exc_info=True)

    logger.info("--- Finished testing document_loader.py ---")
