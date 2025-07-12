import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from langchain.docstore.document import Document
import io
import os

# Set Tesseract path and validate
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
except Exception as e:
    print(f"Error: Tesseract not found or misconfigured: {e}")
    raise

def preprocess_image(image):
    """Preprocess image for better OCR results."""
    # Convert to grayscale
    image = image.convert('L')
    # Apply thresholding (binarization)
    image = image.point(lambda x: 0 if x < 128 else 255)
    # Increase contrast and convert to RGB
    image = image.convert('RGB')
    return image

def extract_text_from_image(image_data):
    """Extract text from an image file or bytes using Tesseract OCR."""
    try:
        image = Image.open(io.BytesIO(image_data))
        image = preprocess_image(image)
        
        # Try multiple PSM settings for robustness
        for psm in [6, 3]:  # PSM 6 (single block), PSM 3 (auto)
            text = pytesseract.image_to_string(image, config=f'--psm {psm} --oem 3', lang='eng')
            if text.strip():
                print(f"Extracted text with PSM {psm}: {text[:200]}...")
                return Document(page_content=text, metadata={"source": "image", "psm_used": psm})
        
        print(f"[WARNING] No text extracted from image with any PSM setting.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        return None

def extract_text_from_scanned_pdf(pdf_data):
    """Extract text from a scanned PDF using Tesseract OCR."""
    try:
        # Convert PDF to images with higher DPI
        images = convert_from_bytes(pdf_data, dpi=300)  # Increased DPI for better quality
        if not images:
            print(f"[ERROR] PDF to image conversion failed. Ensure poppler is installed and configured.")
            return None

        text = ""
        for i, page in enumerate(images):
            try:
                page = preprocess_image(page)
                # Try multiple OCR settings
                for psm in [6, 3]:  # PSM 6 (single block), PSM 3 (auto)
                    page_text = pytesseract.image_to_string(page, config=f'--psm {psm} --oem 3', lang='eng')
                    if page_text.strip():
                        text += f"\nPage {i+1}:\n{page_text}"
                        break
                    else:
                        print(f"[WARNING] No text extracted from page {i+1} with PSM {psm}.")
                if not page_text.strip():
                    print(f"[WARNING] No text extracted from page {i+1} with any PSM setting.")
            except Exception as e:
                print(f"[WARNING] Failed to process page {i+1}: {e}")
                continue

        if not text.strip():
            print(f"[ERROR] No text extracted from the PDF. The PDF may be unreadable, encrypted, or not image-based.")
            return None

        return Document(page_content=text, metadata={"source": "pdf", "page_count": len(images)})
    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}")
        return None

if __name__ == "__main__":
    # Example usage (uncomment and adjust paths for testing)
    # with open("sample_image.jpg", 'rb') as f:
    #     image_doc = extract_text_from_image(f.read())
    #     if image_doc:
    #         print(f"Extracted text from image: {image_doc.page_content[:200]}...")
    #
    # with open("sample_scanned.pdf", 'rb') as f:
    #     pdf_doc = extract_text_from_scanned_pdf(f.read())
    #     if pdf_doc:
    #         print(f"Extracted text from PDF: {pdf_doc.page_content[:200]}...")
    pass
