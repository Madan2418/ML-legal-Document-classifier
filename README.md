# Legal Document Analyzer

An automated tool for analyzing legal documents, extracting key information, and identifying potential risks.

## Features

- **Document Processing**: Handles various document formats (TXT, PDF, DOCX)
- **Text Summarization**: Generates concise summaries of legal documents
- **Risk Detection**: Identifies and highlights potential risk factors in legal text
- **Document Classification**: Categorizes documents based on their content
- **Visualization**: Creates visual representations of document analysis

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd legal-document-analyzer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. Install the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage

To analyze a single legal document:

```bash
python legal_doc_analyzer_ml.py path/to/your/document.pdf
```

### Command Line Arguments

- `file_path`: Path to the legal document file (required)
- `--output`, `-o`: Output directory for results (default: 'results/')
- `--verbose`, `-v`: Enable verbose output

Example:

```bash
python legal_doc_analyzer_ml.py data/contracts/agreement.pdf --output analysis_results --verbose
```

### Output

The tool will generate the following outputs:

1. **Console Output**:
   - Document summary
   - Detected risk factors
   - Document category
   - Path to generated visualizations

2. **Files**:
   - Summary text file
   - Risk analysis report
   - Visualizations (word cloud, risk factors chart, etc.)
   - Full analysis report (JSON)

## Project Structure

```
legal-document-analyzer/
├── legal_doc_analyzer_ml.py  # Main script
├── preprocess.py            # Document loading and preprocessing
├── summarizer.py            # Text summarization
├── risk_detector.py         # Risk factor detection
├── models.py                # Document classification
├── visualize.py             # Data visualization
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Customization

### Adding Custom Risk Terms

You can add custom risk terms by creating a JSON file with the following structure:

```json
{
    "custom_category_1": [
        "term 1",
        "term 2",
        "term 3"
    ],
    "custom_category_2": [
        "term 4",
        "term 5"
    ]
}
```

Then specify the path to this file when initializing the `RiskDetector` class.

### Training Custom Models

To train a custom document classifier, prepare a CSV or JSON file with 'text' and 'label' columns, then run:

```python
from models import train_document_classifier

results = train_document_classifier(
    data_path='path/to/training_data.csv',
    output_dir='models',
    model_type='random_forest'  # or 'svm', 'logistic_regression'
)
```

## Dependencies

- Python 3.7+
- See `requirements.txt` for a complete list of Python dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with ❤️ using Python
- Uses spaCy for NLP processing
- Visualizations powered by Matplotlib and Seaborn
