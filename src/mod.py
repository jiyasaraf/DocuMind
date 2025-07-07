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
# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
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
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    pypdf2_extracted_any_text = True
    except Exception as e:
        logging.error(f"Error reading PDF with PyPDF2: {e}")
        pypdf2_extracted_any_text = False # Reset if error occurred during PyPDF2 extraction

    # If PyPDF2 extracted no text or failed, try OCR with pytesseract
    if not pypdf2_extracted_any_text and PDF2IMAGE_AVAILABLE:
        logging.info(f"PyPDF2 extracted no text or failed for {file_path}. Attempting OCR...")
        try:
            # Convert PDF pages to images
            images = convert_from_path(file_path)
            for i, image in enumerate(images):
                # Use Tesseract to do OCR on the image
                page_text = pytesseract.image_to_string(image)
                text += page_text
                logging.info(f"OCR processed page {i+1}")
        except Exception as e:
            logging.error(f"Error during OCR extraction with pdf2image/pytesseract: {e}")
            text = "" # Clear text if OCR also fails, to indicate no text could be extracted
    elif not PDF2IMAGE_AVAILABLE:
        logging.warning("pdf2image is not available, skipping OCR for PDF.")

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
        logging.info(f"Successfully read {len(text)} characters from TXT file: {file_path}")
        return text
    except Exception as e:
        logging.error(f"Error reading TXT file {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of a specified size with overlap.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The desired size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []

    # Replace multiple newlines with single spaces to treat paragraphs as continuous text
    # This helps in creating more coherent chunks across paragraph breaks
    text = re.sub(r'\s+', ' ', text).strip()

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text) and end < len(text): # Ensure the very last part is also added if it's smaller than chunk_size
            chunks.append(text[end:])
            break
    return chunks

def process_document(file_path: str, file_type: str) -> list[str]:
    """
    Processes a document (PDF or TXT) to extract and chunk its text content.

    Args:
        file_path (str): The path to the document file.
        file_type (str): The type of the file ('pdf' or 'txt').

    Returns:
        list[str]: A list of text chunks from the document.
    """
    logging.info(f"Starting document processing for file: {file_path}, type: {file_type}")
    text = ""
    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_type == "txt": # Now explicitly handles 'txt' which will also cover 'plain' from main.py
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
        print(f"Chunk {i+1}: {chunk}")
