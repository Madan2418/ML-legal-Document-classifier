"""
Test script for the DocumentSummarizer class.
"""

import os
from summarizer import DocumentSummarizer

def test_summarizer():
    """Test the DocumentSummarizer with a sample legal document."""
    # Initialize the summarizer
    summarizer = DocumentSummarizer(algorithm="lexrank")
    
    # Sample legal document text
    sample_text = """
    This Agreement ("Agreement") is made and entered into as of [Date] (the "Effective Date") by and between 
    [Company Name], a corporation organized and existing under the laws of [State/Country], with its principal 
    office at [Address] ("Company"), and [Client Name], with an address at [Address] ("Client").
    
    WHEREAS, Company is engaged in the business of providing [description of services]; and
    WHEREAS, Client desires to retain Company to provide such services, and Company desires to provide such 
    services to Client, subject to the terms and conditions set forth herein.
    
    NOW, THEREFORE, in consideration of the mutual covenants and agreements hereinafter set forth and for other 
    good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the parties 
    hereto agree as follows:
    
    1. SERVICES. Company shall provide to Client the following services (the "Services"): [detailed description 
    of services to be provided].
    
    2. COMPENSATION. In consideration for the Services to be performed by Company, Client shall pay Company a fee 
    of [amount] (the "Fee") payable as follows: [payment terms].
    
    3. TERM AND TERMINATION. This Agreement shall commence on the Effective Date and shall continue until [end date]
    unless earlier terminated as provided herein. Either party may terminate this Agreement upon [number] days' 
    written notice to the other party.
    
    4. CONFIDENTIALITY. Each party agrees to maintain the confidentiality of the other party's proprietary 
    information ("Confidential Information") and not to disclose such information to any third party without 
    the prior written consent of the disclosing party.
    
    5. GOVERNING LAW. This Agreement shall be governed by and construed in accordance with the laws of the State 
    of [State], without regard to its conflict of laws principles.
    
    IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the Effective Date.
    
    [Company Name]                              [Client Name]
    By: ___________________________    By: ___________________________
    Name: _________________________    Name: _________________________
    Title: ________________________    Title: ________________________
    Date: _________________________    Date: _________________________
    """
    
    # Generate summary
    summary = summarizer.summarize(sample_text, target_words=200)
    print("=== SUMMARY ===")
    print(summary)
    
    # Get key phrases
    key_phrases = summarizer.extract_key_phrases(sample_text, top_n=5)
    print("\n=== KEY PHRASES ===")
    print(", ".join(key_phrases))
    
    # Get summary with key phrases
    result = summarizer.get_summary_with_key_phrases(sample_text)
    print("\n=== SUMMARY WITH KEY PHRASES ===")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Language: {result['language']}")
    print("\nSummary:", result['summary'])
    print("\nKey Phrases:", ", ".join(result['key_phrases']))

if __name__ == "__main__":
    test_summarizer()
