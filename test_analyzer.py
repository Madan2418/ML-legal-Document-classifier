"""
Test script for the Legal Document Analyzer.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_preprocessing():
    """Test the document preprocessing functionality."""
    print("\n=== Testing Document Preprocessing ===")
    from preprocess import DocumentProcessor
    
    processor = DocumentProcessor()
    test_text = "This is a test document. It contains some legal terms like Plaintiff and Defendant."
    
    print("Original text:", test_text)
    print("Cleaned text:", processor._clean_text(test_text))
    print("Entities:", processor.get_entities(test_text))

def test_summarization():
    """Test the document summarization functionality."""
    print("\n=== Testing Document Summarization ===")
    from summarizer import DocumentSummarizer
    
    summarizer = DocumentSummarizer()
    test_text = """
    The plaintiff, John Doe, filed a lawsuit against the defendant, Acme Corp, 
    alleging breach of contract. The contract in question was signed on January 1, 2023, 
    and was for the delivery of 1000 widgets by March 1, 2023. The plaintiff claims 
    that only 800 widgets were delivered by the deadline, causing financial damages. 
    The defendant counters that the delay was due to unforeseen circumstances and 
    offered to deliver the remaining widgets by April 1, 2023. The plaintiff rejected 
    this offer and is seeking damages of $10,000.
    """
    
    print("Original text:", test_text)
    print("Summary:", summarizer.summarize(test_text))

def test_risk_detection():
    """Test the risk detection functionality."""
    print("\n=== Testing Risk Detection ===")
    from risk_detector import RiskDetector
    
    detector = RiskDetector()
    test_text = """
    This agreement may be terminated by either party at any time without notice. 
    The Company shall have no liability for any damages arising from the use of this software.
    The User agrees to indemnify and hold harmless the Company from any claims.
    The software is provided "as is" without any warranties, express or implied.
    """
    
    print("Test text:", test_text)
    risks = detector.detect_risks(test_text)
    print("Detected risks:")
    for i, risk in enumerate(risks, 1):
        print(f"{i}. {risk['category']}: {risk['term']} (Severity: {risk['severity']})")

if __name__ == "__main__":
    test_preprocessing()
    test_summarization()
    test_risk_detection()
