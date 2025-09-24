"""
Risk detection module for legal document analysis.
Identifies and highlights risk factors in legal documents.
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
import spacy
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskDetector:
    """
    Detects risk factors in legal documents.
    Identifies potentially risky clauses, terms, and phrases.
    """
    
    def __init__(self, risk_terms_file: Optional[str] = None):
        """
        Initialize the risk detector with risk terms and patterns.
        
        Args:
            risk_terms_file: Path to a JSON file containing risk terms and patterns
        """
        # Comprehensive risk terms and patterns for legal document analysis
        self.risk_terms = {
            # Domestic violence and harassment related
            'domestic_violence': [
                'dowry', 'dowry death', 'dowry harassment', 'dowry demand',
                'harassment', 'mental harassment', 'mental cruelty', 
                'cruelty', 'torture', 'abuse', 'domestic abuse', 'physical abuse',
                'mental torture', 'emotional abuse', 'violence', 'domestic violence',
                'threat', 'intimidation', 'coercion', 'force', 'assault', 'battery'
            ],
            
            # Financial and property risks
            'financial_abuse': [
                'dowry demand', 'illegal demand', 'unlawful demand',
                'money demand', 'cash demand', 'property demand',
                'stridhan', 'dowry articles', 'dowry items', 'gifts',
                'financial control', 'economic abuse', 'deprivation of property',
                'disclaimer of liability', 'liability cap',
                'liability limitation', 'liability exclusion',
                'not liable for', 'no responsibility for',
                'to the fullest extent permitted by law',
                'consequential damages', 'indirect damages',
                'incidental damages', 'punitive damages'
            ],
            
            # Legal and criminal terms
            'criminal_offenses': [
                'suicide', 'suicidal', 'dowry death', 'unnatural death',
                'murder', 'homicide', 'culpable homicide', 'dowry murder',
                'abetment', 'abetment to suicide', 'criminal conspiracy',
                'criminal intimidation', 'criminal breach of trust',
                'forgery', 'fabrication', 'false evidence', 'perjury',
                'concealment', 'fraud', 'cheating', 'misrepresentation'
            ],
            
            # Legal proceedings and judgments
            'legal_risks': [
                'convicted', 'sentenced', 'imprisonment', 'jail', 'punishment',
                'penalty', 'fine', 'compensation', 'damages', 'liability',
                'breach', 'violation', 'non-compliance', 'contempt',
                'prosecution', 'trial', 'conviction', 'acquittal', 'bail',
                'remand', 'custody', 'arrest', 'fir', 'charge sheet',
                'cognizable offense', 'non-bailable offense'
            ],
            
            # Contract and agreement risks
            'contract_risks': [
                'termination without notice', 'termination without cause',
                'arbitrary termination', 'unilateral termination',
                'indemnify', 'indemnification', 'hold harmless', 'waiver',
                'limitation of liability', 'exclusion of liability',
                'force majeure', 'act of god', 'unforeseen circumstances',
                'breach of contract', 'material breach', 'default',
                'liquidated damages', 'penalty clause', 'confidentiality',
                'non-disclosure', 'non-compete', 'restrictive covenant',
                'change at any time', 'modification at any time',
                'right to modify', 'right to amend',
                'without notice', 'at our sole discretion',
                'at our discretion', 'in our discretion'
            ],
            
            # Personal safety and well-being
            'safety_risks': [
                'threat to life', 'danger to life', 'risk of harm',
                'physical safety', 'mental health', 'psychological impact',
                'harassment', 'stalking', 'cyber stalking', 'defamation',
                'character assassination', 'blackmail', 'extortion',
                'molestation', 'eve teasing', 'sexual harassment', 'rape',
                'marital rape', 'unnatural offenses', 'outraging modesty'
            ],
            
            # Family and relationship risks
            'family_risks': [
                'desertion', 'abandonment', 'willful neglect',
                'deprivation of conjugal rights', 'marital discord',
                'irretrievable breakdown', 'mental cruelty', 'adultery',
                'bigamy', 'polygamy', 'illegal relationship', 'affair',
                'child custody', 'child abuse', 'child neglect',
                'child marriage', 'forced marriage', 'dowry prohibition',
                'domestic violence', 'protection order', 'restraining order'
            ],
            
            # Legal jurisdiction and dispute resolution
            'jurisdiction': [
                'governing law', 'jurisdiction', 'venue',
                'exclusive jurisdiction', 'dispute resolution',
                'arbitration clause', 'waiver of jury trial',
                'class action waiver', 'waiver of class actions'
            ],
            
            # Intellectual property risks
            'intellectual_property': [
                'intellectual property rights', 'IP rights',
                'assignment of rights', 'work for hire',
                'ownership of work product', 'proprietary rights',
                'confidential information', 'non-disclosure'
            ],
            
            # Financial penalties and charges
            'penalties': [
                'liquidated damages', 'penalty clause',
                'late fees', 'interest charges',
                'default interest', 'penalty interest',
                'breach penalties', 'contractual penalties'
            ],
            
            # Additional contract terms
            'warranties': [
                r'(no|without) (warrant|guarantee|assurance)',
                'as is', 'as available', 'without warranty',
                'disclaimer of warranties', 'no warranty',
                'express or implied', 'warranty disclaimer',
                'all warranties', 'exclusion of warranties'
            ],
            
            'amendment': [
                'change at any time', 'modification at any time',
                'right to modify', 'right to amend',
                'without notice', 'at our sole discretion',
                'at our discretion', 'in our discretion'
            ]
        }
        
        # Load custom risk terms if provided
        if risk_terms_file and Path(risk_terms_file).exists():
            try:
                with open(risk_terms_file, 'r', encoding='utf-8') as f:
                    custom_terms = json.load(f)
                    self.risk_terms.update(custom_terms)
                logger.info(f"Loaded custom risk terms from {risk_terms_file}")
            except Exception as e:
                logger.error(f"Failed to load custom risk terms: {str(e)}")
        
        # Initialize spaCy for NLP processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.error("Spacy model 'en_core_web_sm' not found. Please install it.")
            raise
            
        # Compile regex patterns for faster matching
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for risk terms."""
        patterns = {}
        for category, terms in self.risk_terms.items():
            # Create a pattern that matches any of the terms in the category
            pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms) + r')\b'
            patterns[category] = re.compile(pattern, re.IGNORECASE)
        return patterns
    
    def _get_context(self, text: str, position: int, window: int = 100) -> str:
        """
        Extract context around a given position in the text.
        
        Args:
            text: The full text content
            position: The position in the text to center the context around
            window: Number of characters to include on each side of the position
            
        Returns:
            A string containing the context around the specified position
        """
        if not text or position < 0 or position > len(text):
            return ""
            
        # Calculate start and end positions
        start = max(0, position - window)
        end = min(len(text), position + window)
        
        # Try to start at a word boundary
        while start > 0 and not text[start].isspace() and (position - start) < (window * 1.5):
            start -= 1
            
        # Try to end at a word boundary
        while end < len(text) and not text[end].isspace() and (end - position) < (window * 1.5):
            end += 1
            
        # Extract the context
        context = text[start:end].strip()
        
        # Add ellipsis if we didn't capture the full text
        if start > 0:
            context = '...' + context
        if end < len(text):
            context = context + '...'
            
        # Clean up any extra whitespace
        context = ' '.join(context.split())
        
        return context

    def detect_risks(self, text: str, min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect and analyze risk factors in the given text with comprehensive context.
        
        Args:
            text: The text to analyze for risk factors
            min_confidence: Minimum confidence score (0-1) for a risk to be included
            
        Returns:
            List of dictionaries containing risk details including term, category, 
            context, and severity score
        """
        if not text or not text.strip():
            return []
        
        # Ensure spaCy is initialized (used for potential later enhancements)
        if not hasattr(self, 'nlp'):
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.error("Spacy model 'en_core_web_sm' not found. Please install it.")
                raise
        
        # Compile regex patterns for faster matching from list-based risk_terms
        self.patterns = self._compile_patterns()
        
        detected: List[Dict[str, Any]] = []
        seen_spans: Set[Tuple[int, int, str]] = set()  # (start, end, category)
        
        # Base confidence by category (fallbacks)
        base_confidence_map = {
            'domestic_violence': 0.8,
            'criminal_offenses': 0.9,
            'legal_risks': 0.7,
            'contract_risks': 0.6,
            'jurisdiction': 0.5,
            'financial_abuse': 0.75,
            'safety_risks': 0.8,
            'family_risks': 0.75,
            'intellectual_property': 0.6,
            'penalties': 0.7,
        }
        
        for category, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                term = match.group(0)
                if (start, end, category) in seen_spans:
                    continue
                seen_spans.add((start, end, category))
                
                context = self._get_context(text, start, window=120)
                severity = self._get_severity(category, term, context)
                
                # Confidence heuristic: base + small boost for severity and phrase length
                base_conf = base_confidence_map.get(category, 0.6)
                sev_boost = {'low': 0.0, 'medium': 0.05, 'high': 0.1}.get(severity, 0.05)
                len_boost = min(len(term.split()) * 0.02, 0.08)
                confidence = max(0.0, min(1.0, base_conf + sev_boost + len_boost))
                
                if confidence >= min_confidence:
                    detected.append({
                        'category': category,
                        'term': term,
                        'start': start,
                        'end': end,
                        'context': context,
                        'severity': severity,
                        'confidence': round(confidence, 2)
                    })
        
        # Sort by severity and then by confidence
        severity_rank = {'high': 2, 'medium': 1, 'low': 0}
        detected.sort(key=lambda x: (severity_rank.get(x['severity'], 0), x['confidence']), reverse=True)
        return detected
    
    def _get_severity(self, category: str, term: str, context: str) -> str:
        """
        Determine the severity of a detected risk based on category, term, and context.
        
        Args:
            category: Risk category (e.g., 'domestic_violence', 'criminal_offenses')
            term: The specific term that was matched
            context: The surrounding context of the match
            
        Returns:
            Severity level ('high', 'medium', 'low')
        """
        # Default severity levels by category
        category_severity = {
            'domestic_violence': 'high',
            'criminal_offenses': 'high',
            'legal_risks': 'medium',
            'contract_risks': 'medium',
            'jurisdiction': 'low',
            'financial_abuse': 'high',
            'safety_risks': 'high',
            'family_risks': 'high',
            'intellectual_property': 'medium',
            'penalties': 'high'
        }
        
        # Start with the base severity for the category
        severity = category_severity.get(category.lower().replace(' ', '_'), 'medium')
        
        # Adjust severity based on specific high-risk terms
        high_risk_terms = [
            'dowry death', 'murder', 'homicide', 'suicide', 'rape', 'torture',
            'assault', 'battery', 'coercion', 'blackmail', 'extortion', 'molestation',
            'sexual harassment', 'marital rape', 'domestic violence', 'child abuse'
        ]
        
        if any(term.lower() in high_risk_terms for term in [term] + term.split()):
            severity = 'high'
        
        # Check for intensifying context
        intensifiers = [
            'severe', 'extreme', 'serious', 'grave', 'critical',
            'violent', 'brutal', 'heinous', 'atrocious', 'inhuman'
        ]
        
        if any(word in context.lower() for word in intensifiers):
            # Increase severity if not already high
            if severity == 'low':
                severity = 'medium'
            elif severity == 'medium':
                severity = 'high'
        
        # Check for mitigating context
        mitigators = [
            'not', 'no', 'without', 'never', 'none', 'neither', 'nor',
            'dismissed', 'acquitted', 'cleared', 'exonerated', 'vindicated'
        ]
        
        if any(word in context.lower().split() for word in mitigators):
            # Decrease severity if not already low
            if severity == 'high':
                severity = 'medium'
            elif severity == 'medium':
                severity = 'low'
        
        return severity
    
    def get_risk_summary(self, risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of detected risks.
        
        Args:
            risks: List of risk dictionaries from detect_risks()
            
        Returns:
            Dictionary with risk summary statistics
        """
        if not risks:
            return {
                'total_risks': 0,
                'by_category': {},
                'by_severity': {},
                'high_risk_terms': []
            }
        
        summary = {
            'total_risks': len(risks),
            'by_category': defaultdict(int),
            'by_severity': defaultdict(int),
            'high_risk_terms': []
        }
        
        unique_terms = set()
        
        for risk in risks:
            summary['by_category'][risk['category']] += 1
            summary['by_severity'][risk['severity']] += 1
            
            if risk['severity'] == 'high':
                term_info = {
                    'term': risk['term'],
                    'category': risk['category'],
                    'context': risk['context']
                }
                if risk['term'] not in unique_terms:
                    summary['high_risk_terms'].append(term_info)
                    unique_terms.add(risk['term'])
        
        # Convert defaultdict to regular dict for JSON serialization
        summary['by_category'] = dict(summary['by_category'])
        summary['by_severity'] = dict(summary['by_severity'])
        
        return summary

if __name__ == "__main__":
    # Example usage
    detector = RiskDetector()
    
    test_text = """
    This agreement may be terminated by either party at any time without notice. 
    The Company shall have no liability for any damages arising from the use of this software.
    The User agrees to indemnify and hold harmless the Company from any claims.
    The software is provided "as is" without any warranties, express or implied.
    The Company reserves the right to modify these terms at its sole discretion.
    Any disputes shall be governed by the laws of the State of California.
    The User assigns all intellectual property rights to the Company.
    Late payments shall incur a penalty of 1.5% per month.
    """
    
    print("Detecting risks in test text...")
    risks = detector.detect_risks(test_text)
    summary = detector.get_risk_summary(risks)
    
    print(f"\nFound {summary['total_risks']} risk instances:")
    print(f"By category: {summary['by_category']}")
    print(f"By severity: {summary['by_severity']}")
    
    print("\nHigh risk terms:")
    for i, term in enumerate(summary['high_risk_terms'], 1):
        print(f"{i}. {term['term']} ({term['category']})")
        print(f"   Context: {term['context']}\n")
