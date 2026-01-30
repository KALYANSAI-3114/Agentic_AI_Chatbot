import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class PDFIngester:
    #Initialization
    def __init__(self, chunk_size=1200, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    

    #Reading PDF
        
    def read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    # Chunking PDF
    def chunk_text(self, text: str) -> List[Document]:
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"source": "Agentic-AI-eBook"}) for chunk in chunks]
