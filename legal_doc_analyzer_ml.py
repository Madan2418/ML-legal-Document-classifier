#!/usr/bin/env python3
"""
Legal Document Analyzer ML
A tool to analyze legal documents, extract summaries, and detect risk factors.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from preprocess import DocumentProcessor
from summarizer import DocumentSummarizer
from risk_detector import RiskDetector
from models import DocumentClassifier
from visualize import DocumentVisualizer

class LegalDocumentAnalyzer:
    """Main class for the Legal Document Analyzer application."""
    
    def __init__(self, model_path: str = "models/document_classifier.joblib"):
        """
        Initialize the analyzer with required components.
        
        Args:
            model_path: Path to the trained classifier model
        """
        self.processor = DocumentProcessor()
        self.summarizer = DocumentSummarizer()
        self.risk_detector = RiskDetector()
        
        # Initialize and load the classifier
        self.classifier = DocumentClassifier()
        try:
            if os.path.exists(model_path):
                self.classifier.load_model(model_path)
                logger.info(f"Loaded classifier model from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}. Please train the model first.")
        except Exception as e:
            logger.error(f"Error loading classifier model: {str(e)}")
            
        self.visualizer = DocumentVisualizer()
        
    def _format_summary(self, text: str, max_length: int = 1200) -> str:
        """Format summary to a reasonable length and add ellipsis if truncated."""
        if len(text) > max_length:
            return text[:max_length].rsplit(' ', 1)[0] + '...'
        return text
        
    def _format_risk_factors(self, risk_factors: List[str]) -> str:
        """Format risk factors for display."""
        if not risk_factors:
            return "No significant risk factors identified."
        return "\n- " + "\n- ".join(risk_factors)
        
    def analyze_document(self, file_path: str) -> Dict:
        """
        Analyze a single legal document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing analysis results
        """
        try:
            # 1. Load and preprocess the document
            logger.info(f"Processing document: {file_path}")
            doc_text = self.processor.process_document(file_path)
            
            if not doc_text:
                logger.error(f"Failed to extract text from {file_path}")
                return {}
                
            # 2. Generate summary
            logger.info("Generating document summary...")
            summary = self.summarizer.summarize(doc_text, target_words=700)
            
            # 3. Detect risk factors with improved detection
            logger.info("Analyzing potential risk factors...")
            
            # Get risk factors as a list
            risk_factors = self.risk_detector.detect_risks(doc_text)
            
            # Convert risk factors to a more readable format (keep dicts, prepare display later)
            formatted_risks = []
            if isinstance(risk_factors, list):
                formatted_risks = risk_factors
            elif isinstance(risk_factors, dict):
                # Handle the case where risk_factors is a dictionary
                for category, risks in risk_factors.items():
                    if isinstance(risks, list):
                        for risk in risks:
                            if isinstance(risk, dict):
                                formatted_risks.append({
                                    **risk,
                                    'category': category
                                })
                            else:
                                formatted_risks.append({'category': category, 'term': str(risk)})
                    else:
                        formatted_risks.append({'category': category, 'term': str(risks)})
            
            # 4. Classify document
            logger.info("Classifying document...")
            category = self.classifier.predict_category(doc_text) if hasattr(self.classifier, 'predict_category') else "Unknown"
            
            # 5. Generate visualizations
            logger.info("Generating visualizations...")
            viz_path = {}
            if hasattr(self, 'visualizer') and hasattr(self.visualizer, 'create_visualizations'):
                try:
                    viz_path = self.visualizer.create_visualizations(doc_text, file_path, risk_factors=formatted_risks) or {}
                except Exception as e:
                    logger.error(f"Error generating visualizations: {str(e)}")
                    viz_path = {"error": f"Failed to generate visualizations: {str(e)}"}
            
            # 6. Format and return results
            return {
                'file_path': file_path,
                'summary': self._format_summary(summary) if hasattr(self, '_format_summary') else summary[:500] + '...',
                'risk_factors': formatted_risks if formatted_risks else ["No significant risk factors identified."],
                'category': category,
                'visualization_path': viz_path
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'error': str(e),
                'summary': '',
                'risk_factors': [f"Error during analysis: {str(e)}"],
                'category': 'Error',
                'visualization_path': {}
            }

def print_section(title: str, content: str, width: int = 80):
    """Print a formatted section with title and content."""
    print("\n" + title)
    print("-" * width)
    print(content)

def print_analysis_results(results: Dict, width: int = 80):
    """Print the analysis results in a clean, readable format."""
    # Document header
    print("\n" + "=" * width)
    print(f"DOCUMENT ANALYSIS: {Path(results['file_path']).name}".center(width))
    print("=" * width)
    
    # Document category
    print_section("DOCUMENT CATEGORY", results['category'])
    
    # Summary section
    print_section("SUMMARY", results['summary'])
    
    # Risk factors section
    risk_content = results['risk_factors']
    if isinstance(risk_content, list):
        lines = []
        for item in risk_content:
            if isinstance(item, dict):
                cat = item.get('category', 'Unknown')
                term = item.get('term', '')
                sev = item.get('severity')
                conf = item.get('confidence')
                line = f"- {cat}: {term}"
                meta = []
                if sev:
                    meta.append(f"severity={sev}")
                if conf is not None:
                    meta.append(f"confidence={conf}")
                if meta:
                    line += " (" + ", ".join(meta) + ")"
                lines.append(line)
            else:
                lines.append(f"- {item}")
        risk_content = "\n".join(lines) if lines else "No significant risk factors identified."
    print_section("RISK FACTORS", risk_content)
    
    # Visualizations section
    if results['visualization_path']:
        if isinstance(results['visualization_path'], dict):
            viz_content = "\n".join(f"- {viz_type}: {path}" 
                                  for viz_type, path in results['visualization_path'].items())
        else:
            viz_content = str(results['visualization_path'])
        print_section("VISUALIZATIONS", viz_content)
    
    print("=" * width + "\n")

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description='Legal Document Analyzer')
    parser.add_argument('file_paths', nargs='+', help='Path(s) to document(s) to analyze')
    parser.add_argument('--output-dir', default='results', help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the analyzer
    analyzer = LegalDocumentAnalyzer()
    
    # Process each file
    for file_path in args.file_paths:
        try:
            # Analyze the document
            results = analyzer.analyze_document(file_path)
            
            # Print formatted results
            print_analysis_results(results)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
