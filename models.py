"""
Machine learning models for legal document classification and analysis.
"""

import os
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """
    Class for training and using document classification models.
    Supports multiple classifier types and feature extraction methods.
    """
    
    def __init__(self, model_type: str = 'random_forest', model_path: Optional[str] = None):
        """
        Initialize the document classifier.
        
        Args:
            model_type: Type of classifier to use ('random_forest', 'svm', 'logistic_regression')
            model_path: Path to load a pre-trained model (optional)
        """
        self.model_type = model_type.lower()
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.classes_ = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, X: List[str], y: List[Any], test_size: float = 0.2, 
              random_state: int = 42, **kwargs) -> Dict[str, Any]:
        """
        Train the document classifier.
        
        Args:
            X: List of document texts
            y: List of corresponding labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for the classifier
            
        Returns:
            Dictionary containing training results and metrics
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty lists of the same length")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Create pipeline with TF-IDF vectorizer and classifier
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        # Initialize classifier
        if self.model_type == 'svm':
            classifier = SVC(
                kernel='linear',
                probability=True,
                random_state=random_state,
                **kwargs
            )
        elif self.model_type == 'logistic_regression':
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                multi_class='multinomial',
                **kwargs
            )
        else:  # Default to Random Forest
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', classifier)
        ])
        
        # Train the model
        logger.info(f"Training {self.model_type} classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'num_classes': len(self.classes_)
        }
    
    def predict_category(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict the category of one or more documents.
        
        Args:
            text: Input text or list of texts to classify
            
        Returns:
            Predicted category or list of categories
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        if not text:
            return [] if isinstance(text, list) else ""
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Make predictions
        predictions = self.model.predict(texts)
        
        # Convert numeric predictions back to original labels
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions[0] if is_single else predictions
    
    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Predict class probabilities for one or more documents.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        if not text:
            return np.array([])
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        return self.model.predict_proba(texts)
    
    def save_model(self, output_dir: str, model_name: str = 'document_classifier') -> str:
        """
        Save the trained model to disk.
        
        Args:
            output_dir: Directory to save the model
            model_name: Base name for the model files
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded")
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}.joblib")
        
        # Save model components
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model components
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.model_type = model_data.get('model_type', 'random_forest')
        self.label_encoder = model_data.get('label_encoder')
        self.classes_ = model_data.get('classes_')
        
        # Extract vectorizer from the pipeline
        if hasattr(self.model, 'named_steps') and 'tfidf' in self.model.named_steps:
            self.vectorizer = self.model.named_steps['tfidf']
        
        logger.info(f"Loaded {self.model_type} model from {model_path}")

def train_document_classifier(data_path: str = None, output_dir: str = "models",
                            model_type: str = 'random_forest',
                            test_size: float = 0.2,
                            random_state: int = 42,
                            X_train: List[str] = None,
                            y_train: List[str] = None) -> Dict[str, Any]:
    """
    Train a document classifier from a CSV/JSON file or directly from data.
    
    Args:
        data_path: Path to the training data file (CSV or JSON) - either this or (X_train, y_train) must be provided
        output_dir: Directory to save the trained model
        model_type: Type of classifier to train
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        X_train: List of training texts (alternative to data_path)
        y_train: List of training labels (alternative to data_path)
        
    Returns:
        Dictionary containing training results and model information
    """
    if data_path is None and (X_train is None or y_train is None):
        raise ValueError("Either data_path or (X_train and y_train) must be provided")
    
    if data_path is not None:
        # Load data from file
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix.lower() == '.json':
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
        
        # Check required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Data must contain 'text' and 'label' columns")
        
        X = df['text'].tolist()
        y = df['label'].tolist()
    else:
        # Use provided data directly
        X = X_train
        y = y_train
    
    # Initialize and train classifier
    classifier = DocumentClassifier(model_type=model_type)
    
    # Use a small test size if we don't have much data
    if len(X) < 100:
        test_size = min(0.1, test_size)
    
    results = classifier.train(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Save the trained model
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = classifier.save_model(str(output_dir))
    results['model_path'] = model_path
    
    return results

if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create sample data
    data = {
        'text': [
            "This is a legal contract about intellectual property rights.",
            "The parties agree to the terms and conditions of this agreement.",
            "This document outlines the liability limitations for the service.",
            "The software is provided as-is without any warranties.",
            "The agreement may be terminated by either party with 30 days notice.",
            "All intellectual property rights are reserved by the company.",
            "The service provider disclaims all liability for any damages.",
            "This is a sample legal document for demonstration purposes."
        ],
        'label': [
            'ip', 'general', 'liability', 'warranties',
            'termination', 'ip', 'liability', 'general'
        ]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create a temporary directory for the model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save sample data to a temporary file
        temp_csv = os.path.join(temp_dir, 'sample_data.csv')
        df.to_csv(temp_csv, index=False)
        
        # Train a classifier
        print("Training sample classifier...")
        results = train_document_classifier(
            data_path=temp_csv,
            output_dir=temp_dir,
            model_type='random_forest'
        )
        
        print(f"\nTraining results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Model saved to: {results['model_path']}")
        
        # Test the classifier
        test_texts = [
            "This agreement may be terminated at any time.",
            "All rights to the software are reserved.",
            "The company is not liable for any damages."
        ]
        
        classifier = DocumentClassifier(model_path=results['model_path'])
        predictions = classifier.predict_category(test_texts)
        
        print("\nTest predictions:")
        for text, pred in zip(test_texts, predictions):
            print(f"\nText: {text}")
            print(f"Predicted category: {pred}")
