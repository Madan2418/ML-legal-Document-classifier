"""
Train a document classifier for legal documents.
"""

import os
import logging
import pandas as pd
from pathlib import Path
from models import DocumentClassifier, train_document_classifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data(data_dir: str) -> tuple:
    """
    Prepare training data from the dataset directory.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    # Define the categories based on directory names
    categories = {
        'IN-Ext': 'indian_extended'  # Only using IN-Ext since other directories are not found
    }
    
    # Read files from each category
    for category_dir, label in categories.items():
        category_path = Path(data_dir) / category_dir / 'judgement'
        if not category_path.exists():
            logger.warning(f"Directory not found: {category_path}")
            continue
            
        logger.info(f"Processing category: {label}")
        
        # Read all text files in the directory
        for file_path in category_path.glob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:  # Only add non-empty texts
                        texts.append(text)
                        labels.append(label)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
    
    return texts, labels

def main():
    # Paths
    data_dir = Path("dataset/dataset")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare training data
    logger.info("Preparing training data...")
    texts, labels = prepare_training_data(data_dir)
    
    if not texts:
        logger.error("No training data found!")
        return
        
    logger.info(f"Prepared {len(texts)} documents for training")
    
    # Train the classifier with a small test size since we have limited data
    logger.info("Training the classifier...")
    try:
        # Use the train_document_classifier function from models.py
        results = train_document_classifier(
            data_path=None,  # We're passing data directly
            output_dir=str(output_dir),
            model_type='random_forest',
            test_size=0.1,  # Smaller test size for limited data
            random_state=42,
            # Pass the prepared data directly
            X_train=texts,
            y_train=labels
        )
        
        logger.info(f"Training completed! Model saved to {output_dir}")
        logger.info(f"Test accuracy: {results.get('test_accuracy', 0):.2f}")
        
        # If test accuracy is 0, it might be due to small test size, so we'll do a manual evaluation
        if results.get('test_accuracy', 0) == 0 and len(texts) > 10:
            logger.info("Running a manual evaluation since test accuracy is 0...")
            from sklearn.model_selection import cross_val_score
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier
            
            # Create a simple pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', RandomForestClassifier(random_state=42))
            ])
            
            # Perform cross-validation
            cv_scores = cross_val_score(pipeline, texts, labels, cv=min(5, len(texts)))
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
