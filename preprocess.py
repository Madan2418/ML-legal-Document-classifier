"""
Document preprocessing module for legal document analysis.
Handles loading and preprocessing of various document formats.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import magic
from PyPDF2 import PdfReader
from docx import Document
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and text preprocessing."""
    
    def __init__(self):
        """Initialize the document processor with NLP models and utilities."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.error("Spacy model 'en_core_web_sm' not found. Please install it.")
            raise
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.mime = magic.Magic(mime=True)
    
    def process_document(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Process a document file and return its cleaned text content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Cleaned text content or None if processing fails
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        try:
            # Determine file type and extract text
            mime_type = self.mime.from_file(str(file_path))
            
            if 'pdf' in mime_type.lower():
                text = self._extract_pdf_text(file_path)
            elif 'word' in mime_type.lower() or 'document' in mime_type.lower():
                text = self._extract_docx_text(file_path)
            elif 'text' in mime_type.lower() or 'plain' in mime_type.lower():
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                logger.warning(f"Unsupported file type: {mime_type}")
                return None
                
            # Clean and preprocess the extracted text
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _extract_pdf_text(self, file_path: Union[str, Path]) -> str:
        """Extract text from a PDF file."""
        text = []
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or '')
        return '\n'.join(text)
    
    def _extract_docx_text(self, file_path: Union[str, Path]) -> str:
        """Extract text from a DOCX file."""
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess the extracted text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""
            
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
                
        return entities

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    test_text = "This is a test document. It contains some legal terms like Plaintiff and Defendant."
    print("Original text:", test_text)
    print("Cleaned text:", processor._clean_text(test_text))
    print("Entities:", processor.get_entities(test_text))
