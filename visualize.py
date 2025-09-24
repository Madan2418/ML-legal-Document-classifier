"""
Visualization module for legal document analysis.
Generates charts and visualizations for document analysis results.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class DocumentVisualizer:
    """
    Handles visualization of document analysis results.
    Generates charts, word clouds, and other visualizations.
    """
    
    def __init__(self, output_dir: str = 'visualizations', 
                 style: str = 'whitegrid',
                 palette: str = 'viridis'):
        """
        Initialize the document visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Matplotlib style to use for plots
            palette: Color palette to use for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style='darkgrid')
        sns.set_palette('viridis')
        sns.set_palette(palette)
        
        # Save palette for later usage
        self.palette = palette
        
        # Default figure size
        self.figsize = (12, 6)
    
    def create_visualizations(self, text: str, document_name: str, 
                            risk_factors: Optional[List[Dict]] = None,
                            categories: Optional[Dict] = None) -> Dict[str, str]:
        """
        Create visualizations for a document.
        
        Args:
            text: Document text
            document_name: Name of the document (used for output filenames)
            risk_factors: List of detected risk factors (from RiskDetector)
            categories: Document categories and confidence scores
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        # Clean the document name for use in filenames
        safe_name = self._sanitize_filename(document_name)
        
        visualizations = {}
        
        try:
            # Generate word cloud
            wc_path = self.generate_wordcloud(text, safe_name)
            if wc_path:
                visualizations['wordcloud'] = wc_path
            
            # Generate risk factors visualization if available
            if risk_factors:
                risk_path = self.plot_risk_factors(risk_factors, safe_name)
                if risk_path:
                    visualizations['risk_factors'] = risk_path
            
            # Generate category distribution if available
            if categories:
                cat_path = self.plot_category_distribution(categories, safe_name)
                if cat_path:
                    visualizations['categories'] = cat_path
            
            # Generate text statistics
            stats_path = self.plot_text_statistics(text, safe_name)
            if stats_path:
                visualizations['statistics'] = stats_path
                
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return visualizations
    
    def generate_wordcloud(self, text: str, document_name: str) -> Optional[str]:
        """
        Generate a word cloud from the document text.
        
        Args:
            text: Document text
            document_name: Sanitized document name for output filename
            
        Returns:
            Path to the generated word cloud image, or None if failed
        """
        try:
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                contour_width=1,
                contour_color='steelblue'
            ).generate(text)
            
            # Save to file
            output_path = self.output_dir / f"{document_name}_wordcloud.png"
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Frequent Terms', fontsize=16)
            plt.tight_layout(pad=0)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            return None
    
    def plot_risk_factors(self, risk_factors: List[Dict], 
                         document_name: str) -> Optional[str]:
        """
        Create a bar plot of risk factors by category and severity.
        
        Args:
            risk_factors: List of risk factor dictionaries from RiskDetector
            document_name: Sanitized document name for output filename
            
        Returns:
            Path to the generated plot, or None if failed
        """
        if not risk_factors:
            return None
            
        try:
            # Convert to DataFrame for easier plotting
            import pandas as pd
            
            df = pd.DataFrame(risk_factors)
            
            # Count risks by category and severity
            risk_counts = df.groupby(['category', 'severity']).size().reset_index(name='count')
            
            # Create plot
            plt.figure(figsize=self.figsize)
            
            # Use seaborn for better styling
            ax = sns.barplot(
                x='category',
                y='count',
                hue='severity',
                data=risk_counts,
                palette=self.palette,
                dodge=True
            )
            
            # Customize plot
            plt.title('Risk Factors by Category and Severity', fontsize=16)
            plt.xlabel('Risk Category', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Severity')
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"{document_name}_risks.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting risk factors: {str(e)}")
            return None
    
    def plot_category_distribution(self, categories: Dict[str, float], 
                                 document_name: str) -> Optional[str]:
        """
        Create a pie chart of document category distribution.
        
        Args:
            categories: Dictionary of category names and confidence scores
            document_name: Sanitized document name for output filename
            
        Returns:
            Path to the generated plot, or None if failed
        """
        if not categories:
            return None
            
        try:
            # Prepare data
            labels = list(categories.keys())
            sizes = list(categories.values())
            
            # Create plot
            plt.figure(figsize=(8, 8))
            
            # Use seaborn for better styling
            colors = sns.color_palette(self.palette, len(labels))
            
            # Plot pie chart
            plt.pie(
                sizes, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.5, edgecolor='w'),
                textprops={'fontsize': 10}
            )
            
            # Add title
            plt.title('Document Category Distribution', fontsize=16)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            plt.axis('equal')
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"{document_name}_categories.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting category distribution: {str(e)}")
            return None
    
    def plot_text_statistics(self, text: str, document_name: str) -> Optional[str]:
        """
        Generate visualizations for text statistics.
        
        Args:
            text: Document text
            document_name: Sanitized document name for output filename
            
        Returns:
            Path to the generated plot, or None if failed
        """
        try:
            # Calculate basic statistics
            words = text.split()
            sentences = text.split('.')
            
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            
            # Calculate word length distribution
            word_lengths = [len(word) for word in words]
            unique_lengths = sorted(set(word_lengths))
            length_counts = [word_lengths.count(l) for l in unique_lengths]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot word length distribution
            sns.barplot(x=unique_lengths, y=length_counts, ax=ax1, palette=self.palette)
            ax1.set_title('Word Length Distribution')
            ax1.set_xlabel('Word Length')
            ax1.set_ylabel('Count')
            
            # Plot basic statistics
            stats_text = (
                f"Total Words: {word_count:,}\n"
                f"Total Sentences: {sentence_count:,}\n"
                f"Avg. Word Length: {avg_word_length:.2f} characters"
            )
            
            ax2.text(0.1, 0.5, stats_text, fontsize=12, va='center')
            ax2.axis('off')
            
            plt.suptitle('Document Statistics', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"{document_name}_stats.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating text statistics: {str(e)}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be used as a filename.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*' + ''.join(chr(i) for i in range(32))
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        max_length = 100
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename.strip('_.')

if __name__ == "__main__":
    # Example usage
    visualizer = DocumentVisualizer()
    
    # Sample text
    sample_text = """
    This is a sample legal document that contains various risk factors and legal terms.
    The parties agree to the terms and conditions outlined herein. The company shall
    not be liable for any damages arising from the use of this document. This agreement
    may be terminated at any time without notice. All intellectual property rights
    are reserved by the respective owners.
    """
    
    # Sample risk factors
    sample_risks = [
        {"category": "Termination", "term": "terminated at any time", "severity": "High", "context": "This agreement may be terminated at any time without notice."},
        {"category": "Liability", "term": "not be liable", "severity": "High", "context": "The company shall not be liable for any damages."},
        {"category": "IP Rights", "term": "intellectual property rights", "severity": "Medium", "context": "All intellectual property rights are reserved."}
    ]
    
    # Sample categories
    sample_categories = {"Contract": 0.6, "Agreement": 0.25, "Legal": 0.15}
    
    # Generate visualizations
    print("Generating sample visualizations...")
    visualizations = visualizer.create_visualizations(
        text=sample_text,
        document_name="sample_document",
        risk_factors=sample_risks,
        categories=sample_categories
    )
    
    print("\nGenerated visualizations:")
    for name, path in visualizations.items():
        print(f"- {name}: {path}")
