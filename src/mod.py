# mod.py
import os
from PyPDF2 import PdfReader
import re
import logging
import pytesseract
from PIL import Image
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    logging.warning("pdf2image not found. PDF OCR functionality will be limited without it. Please install `pip install pdf2image` and Poppler.")
    PDF2IMAGE_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path to the Tesseract executable if it's not in your PATH
# For Windows, uncomment and modify the line below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Make sure to adjust the path to your Tesseract installation directory.

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file, attempting PyPDF2 first, then OCR if no text found.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    pypdf2_extracted_any_text = False

    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            logging.info(f"Attempting to extract text from {num_pages} pages of PDF: {file_path}")

            for i, page in enumerate(reader.pages):
                # Try PyPDF2 extraction first
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    pypdf2_extracted_any_text = True
                    logging.debug(f"Extracted text from page {i+1} using PyPDF2.")
                else:
                    logging.warning(f"No text extracted from page {i+1} using PyPDF2. This page might be image-based or empty.")
            
            # If PyPDF2 extracted nothing at all, try full OCR fallback
            if not pypdf2_extracted_any_text and PDF2IMAGE_AVAILABLE:
                logging.info("PyPDF2 extracted no text. Attempting full PDF OCR using pdf2image and Tesseract.")
                ocr_full_text = ""
                try:
                    images = convert_from_path(file_path)
                    if images:
                        for img_idx, image in enumerate(images):
                            page_ocr_text = pytesseract.image_to_string(image)
                            if page_ocr_text.strip():
                                ocr_full_text += page_ocr_text + "\n"
                                logging.info(f"OCR extracted text from PDF page {img_idx+1}.")
                            else:
                                logging.warning(f"OCR failed for PDF page {img_idx+1}.")
                        if ocr_full_text.strip():
                            text = ocr_full_text
                            logging.info("Successfully extracted text from PDF using OCR fallback.")
                        else:
                            logging.error("OCR fallback also failed to extract any text from the PDF.")
                    else:
                        logging.error("pdf2image could not convert PDF to images for OCR.")
                except Exception as full_ocr_e:
                    logging.error(f"Error during full PDF OCR fallback: {full_ocr_e}", exc_info=True)
            elif not pypdf2_extracted_any_text and not PDF2IMAGE_AVAILABLE:
                logging.warning("PDF2IMAGE is not available. Cannot perform full PDF OCR. Please install `pip install pdf2image` and Poppler.")

    except Exception as e:
        logging.error(f"Error opening or reading PDF '{file_path}': {e}", exc_info=True)
        return ""
    
    if not text.strip():
        logging.error("No significant text could be extracted from the PDF using either PyPDF2 or OCR fallback. The document might be unreadable.")
        return ""

    return text

def extract_text_from_txt(file_path: str) -> str:
    """
    Extracts text from a TXT file.

    Args:
        file_path (str): The path to the TXT file.

    Returns:
        str: The extracted text from the TXT file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            if not text.strip():
                logging.warning(f"TXT file '{file_path}' is empty or contains only whitespace.")
    except Exception as e:
        logging.error(f"Error extracting text from TXT '{file_path}': {e}", exc_info=True)
        return ""
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits a long text into smaller chunks with optional overlap.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        logging.warning("No text provided to chunk.")
        return []

    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        logging.warning("Text became empty after cleaning whitespace. No chunks generated.")
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text) and chunks:
            break
        if start < len(text) and start + (chunk_size - overlap) >= len(text) and len(text) > end:
            if text[end:].strip():
                chunks.append(text[end:])
            break
        
    logging.info(f"Text chunked into {len(chunks)} chunks.")
    return chunks

def process_document(file_path: str, file_type: str) -> list[str]:
    """
    Extracts text from a document based on its type and chunks it.

    Args:
        file_path (str): The path to the uploaded document.
        file_type (str): The type of the file ('pdf' or 'txt' or 'plain').

    Returns:
        list[str]: A list of processed text chunks.
    """
    text = ""
    logging.info(f"Starting document processing for file: {file_path}, type: {file_type}")
    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    # Handle both 'txt' and 'plain' for text files
    elif file_type == "txt" or file_type == "plain": # <--- FIXED THIS LINE
        text = extract_text_from_txt(file_path)
    else:
        logging.error(f"Unsupported file type: {file_type}")
        return []

    if not text.strip():
        logging.error("No significant text extracted from the document. Cannot proceed with chunking.")
        return []

    chunks = chunk_text(text)
    logging.info(f"Finished processing document. Generated {len(chunks)} chunks.")
    return chunks

if __name__ == '__main__':
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write("This is a test document. It has multiple sentences. We will try to chunk this text. This is the fourth sentence.")
    
    print("--- Testing TXT processing ---")
    txt_chunks = process_document("test_doc.txt", "txt")
    for i, chunk in enumerate(txt_chunks):
        print(f"Chunk {i+1}: {chunk}")
    os.remove("test_doc.txt")

    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. " * 5
    print("\n--- Testing long text chunking ---")
    long_text_chunks = chunk_text(long_text, chunk_size=200, overlap=50)
    for i, chunk in enumerate(long_text_chunks):
        print(f"Chunk {i+1} (length {len(chunk)}): {chunk[:50]}...")
