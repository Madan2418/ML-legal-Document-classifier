"""
Document summarization module for legal document analysis.
Implements both extractive and abstractive summarization techniques.
"""

import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import spacy

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """Handles document summarization using various algorithms."""
    
    def __init__(self, language: str = "english", 
                 algorithm: str = "lexrank",
                 summary_ratio: float = 0.2):
        """Initialize the summarizer with specified parameters.
        
        Args:
            language: Language of the text (default: "english")
            algorithm: Summarization algorithm to use (lexrank, lsa, or textrank)
            summary_ratio: Ratio of sentences to include in summary (0-1)
        """
        self.language = language
        self.algorithm = algorithm.lower()
        self.summary_ratio = summary_ratio
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize the appropriate summarizer
        if self.algorithm == "lexrank":
            self.summarizer = LexRankSummarizer(Stemmer(self.language))
        elif self.algorithm == "lsa":
            self.summarizer = LsaSummarizer(Stemmer(self.language))
        elif self.algorithm == "textrank":
            self.summarizer = TextRankSummarizer(Stemmer(self.language))
        else:
            logger.warning(f"Unknown algorithm '{algorithm}'. Using LexRank as default.")
            self.summarizer = LexRankSummarizer(Stemmer(self.language))
        
        # Set stop words
        self.summarizer.stop_words = get_stop_words(self.language)
    
    def summarize(self, text: str, target_words: int = 500) -> str:
        """
        Generate a comprehensive and detailed summary of the given legal text.
        
        Args:
            text: The text to summarize
            target_words: Target number of words for the summary (default: 500)
            
        Returns:
            A detailed summary of the text with key legal points
        """
        if not text.strip():
            return "No content to summarize."
            
        # Tokenize the text into sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        if not sentences:
            return "No valid sentences found in the document."
            
        # Calculate the compression ratio based on target words
        total_words = len(words)
        if total_words <= target_words:
            return text  # Return full text if it's already within target length
            
        compression_ratio = target_words / total_words
        
        # Process the document with spaCy for better analysis
        doc = self.nlp(text)
        
        # Extract key legal terms and concepts with comprehensive entity recognition
        legal_terms = set()
        legal_phrases = set()
        
        # Enhanced legal entity extraction
        for ent in doc.ents:
            if ent.label_ in ['LAW', 'ORG', 'GPE', 'DATE', 'MONEY', 'PERCENT', 'TIME', 'CARDINAL', 'ORDINAL']:
                legal_terms.add(ent.text.lower())
                
        # Extract noun chunks that might be legal concepts
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if len(chunk_text.split()) > 1:  # Multi-word phrases
                legal_phrases.add(chunk_text)
                
        # Add common legal terms with their variations
        common_legal_terms = {
            'plaintiff', 'defendant', 'appellant', 'respondent', 'petitioner',
            'witness', 'testimony', 'evidence', 'exhibit', 'affidavit', 'summons', 'subpoena',
            'judgment', 'order', 'decree', 'verdict', 'appeal', 'revision', 'writ', 'injunction',
            'stay', 'bail', 'parole', 'probation', 'sentence', 'conviction', 'acquittal',
            'allegation', 'complaint', 'fir', 'chargesheet', 'indictment', 'pleading',
            'cause of action', 'res judicata', 'stare decisis', 'bona fide', 'prima facie',
            'mens rea', 'actus reus', 'habeas corpus', 'certiorari', 'mandamus', 'quo warranto',
            'section', 'article', 'clause', 'subsection', 'paragraph', 'statute', 'regulation',
            'act', 'rule', 'provision', 'amendment', 'constitution', 'law', 'legal', 'court'
        }
        legal_terms.update(common_legal_terms)
        
        # Calculate word frequencies with enhanced weighting
        word_frequencies = {}
        stop_words = set(get_stop_words(self.language))
        
        for word in words:
            word = word.lower()
            if (word not in stop_words and word.isalnum() and len(word) > 2 and 
                word not in ['court', 'said', 'would', 'could', 'shall', 'may', 'also']):
                
                # Weight based on term importance
                if word in legal_terms:
                    weight = 3.0  # Highest weight for legal terms
                elif any(phrase.startswith(word) or phrase.endswith(word) for phrase in legal_phrases):
                    weight = 2.5  # High weight for parts of legal phrases
                elif word.isupper() and len(word) > 1:  # Acronyms
                    weight = 2.0
                else:
                    weight = 1.0
                    
                word_frequencies[word] = word_frequencies.get(word, 0) + weight
        
        # Normalize frequencies
        if word_frequencies:
            max_freq = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] = word_frequencies[word] / max_freq
        
        # Enhanced sentence scoring with multiple factors
        sentence_scores = {}
        
        # Calculate average sentence length for normalization
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # First pass: score all sentences
        for i, sentence in enumerate(sentences):
            # Position-based weighting (U-shaped curve - higher at start and end)
            position = i / len(sentences)
            if position < 0.1:  # First 10%
                position_weight = 2.0 - (position * 5)  # 2.0 at start, ~1.5 at 10%
            elif position > 0.9:  # Last 10%
                position_weight = 1.0 + ((position - 0.9) * 10)  # 1.0 at 90%, ~2.0 at end
            else:  # Middle 80%
                position_weight = 1.0
                
            # Length-based weighting (prefer medium-length sentences)
            sentence_length = len(sentence.split())
            length_weight = 1.0 - (abs(sentence_length - avg_sentence_length) / avg_sentence_length)
            
            # Content-based weighting
            content_score = 0
            sentence_words = word_tokenize(sentence.lower())
            for word in sentence_words:
                if word in word_frequencies:
                    content_score += word_frequencies[word]
            
            # Normalize content score by sentence length
            if sentence_length > 0:
                content_score /= sentence_length
            
            # Combine scores with weights
            sentence_scores[i] = {
                'score': (position_weight * 0.3) + (length_weight * 0.2) + (content_score * 0.5),
                'position': i,
                'length': sentence_length,
                'text': sentence
            }
        
        # Sort sentences by score (highest first)
        sorted_sentences = sorted(sentence_scores.values(), key=lambda x: -x['score'])
        
        # Select top sentences to reach target word count
        selected_indices = set()
        total_words = 0
        summary_sentences = []
        
        for sent_info in sorted_sentences:
            if total_words >= target_words:
                break
                
            # Add the sentence if it's not already selected
            if sent_info['position'] not in selected_indices:
                summary_sentences.append((sent_info['position'], sent_info['text']))
                selected_indices.add(sent_info['position'])
                total_words += sent_info['length']
        
        # Sort selected sentences by their original position
        summary_sentences.sort(key=lambda x: x[0])
        
        # Combine sentences into the final summary
        summary = ' '.join([s[1] for s in summary_sentences])
        
        # Ensure the summary is coherent and flows well
        summary = self._post_process_summary(summary)
        
        return summary
    
    def _post_process_summary(self, text: str) -> str:
        """
        Post-process the summary to improve coherence and readability.
        
        Args:
            text: The generated summary text
            
        Returns:
            Processed summary with improved coherence
        """
        # Fix common spacing and punctuation issues
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        
        # Ensure proper paragraph breaks for better readability
        sentences = sent_tokenize(text)
        processed_text = []
        
        for i, sentence in enumerate(sentences):
            # Add paragraph breaks for better structure
            if i > 0 and i % 5 == 0:  # New paragraph every 5 sentences
                processed_text.append('\n\n' + sentence)
            else:
                processed_text.append(sentence)
        
        # Join with proper spacing and return the processed text
        return ' '.join(processed_text).replace(' .', '.').replace(' ,', ',').replace(' ;', ';')
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases from the text using noun chunks.
        
        Args:
            text: Input text
            top_n: Number of key phrases to return
            
        Returns:
            List of key phrases
        """
        doc = self.nlp(text)
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Count phrase frequencies
        phrase_freq = defaultdict(int)
        for chunk in noun_chunks:
            if len(chunk.split()) > 1:  # Only consider multi-word phrases
                phrase_freq[chunk] += 1
        
        # Sort by frequency and return top N
        sorted_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases[:top_n]]
    
    def get_summary_with_key_phrases(self, text: str) -> Dict[str, Any]:
        """Generate a summary along with key phrases.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing summary and key phrases
        """
        summary = self.summarize(text)
        key_phrases = self.extract_key_phrases(text)
        
        return {
            "summary": summary,
            "key_phrases": key_phrases,
            "algorithm": self.algorithm,
            "language": self.language
        }


if __name__ == "__main__":
    # Example usage
    summarizer = DocumentSummarizer(algorithm="lexrank", summary_ratio=0.2)
    
    sample_text = """
    This is a sample legal document that discusses various aspects of contract law.
    The document outlines the terms and conditions that both parties must agree to.
    It includes clauses about payment terms, delivery schedules, and dispute resolution.
    The agreement is governed by the laws of the state of California.
    Any disputes arising from this agreement will be settled through arbitration.
    """
    
    result = summarizer.get_summary_with_key_phrases(sample_text)
    print("Summary:", result["summary"])
    print("\nKey Phrases:", ", ".join(result["key_phrases"]))
