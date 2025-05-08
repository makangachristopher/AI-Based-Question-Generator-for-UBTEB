import os
import PyPDF2
import docx
import traceback

def process_document(file_path):
    """
    Extract text from PDF or Word document
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Extracted text from the document
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        print(traceback.format_exc())
        raise Exception(f"Failed to process document: {str(e)}")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                raise ValueError("Cannot process encrypted PDF files")
            
            # Check if PDF has extractable text
            if len(pdf_reader.pages) == 0:
                raise ValueError("PDF has no pages")
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # If page has no text, add a placeholder to avoid empty document
                if not page_text:
                    page_text = f"[Page {page_num + 1} contains no extractable text]"
                    
                text += page_text + "\n\n"
        
        # If entire document has no text, raise error
        if not text.strip():
            raise ValueError("PDF does not contain extractable text. It may be scanned or image-based.")
            
        return text
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Failed to read PDF file: {str(e)}")
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        print(traceback.format_exc())
        raise

def extract_text_from_docx(docx_path):
    """
    Extract text from Word document
    
    Args:
        docx_path (str): Path to the Word document
        
    Returns:
        str: Extracted text from the Word document
    """
    try:
        doc = docx.Document(docx_path)
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
            text += "\n"
        
        # If document has no text, raise error
        if not text.strip():
            raise ValueError("Word document does not contain any text")
            
        return text
    except Exception as e:
        print(f"Error extracting text from Word document: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"Failed to process Word document: {str(e)}") 