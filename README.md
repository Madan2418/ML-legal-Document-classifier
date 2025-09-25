# ML Legal Document Classifier

An automated tool for analyzing legal documents, extracting key information, identifying potential risks, and classifying documents with machine learning.

---

## 🚀 Features Overview

| Feature                | Description                                                                 | Visual Output                |
|------------------------|-----------------------------------------------------------------------------|------------------------------|
| **Document Processing**| Handles TXT, PDF, DOCX; cleans and extracts text.                           | 📄                           |
| **Text Summarization** | Generates concise summaries of lengthy legal docs.                          | 🗒️ Summary Block             |
| **Risk Detection**     | Identifies and highlights potential risk factors in legal language.          | ⚠️ Risk Chart                |
| **Document Classification** | Categorizes documents by content (Contract, IP, Liability, etc.).      | 🏷️ Category Pie Chart        |
| **Visualization**      | Creates visual representations (word cloud, stats, risk bar charts, etc.).  | 🌥️ Word Cloud, 📊 Bar Graphs |

---

## 🗂 Project Structure

```text
legal-document-analyzer/
├── legal_doc_analyzer_ml.py   # Main workflow script
├── preprocess.py              # Document loading & preprocessing (PDF, DOCX, TXT)
├── summarizer.py              # Text summarization logic
├── risk_detector.py           # Detects legal risks in docs
├── models.py                  # ML models for document classification
├── visualize.py               # All visualizations (word clouds, charts, etc.)
├── train_classifier.py        # Utility script to train new classifiers
├── test_analyzer.py           # Basic tests for preprocessing, summarization, risk detection
├── requirements.txt           # Python dependencies
├── README.md                  # (This file)
└── dataset/                   # Example & training legal documents
```

---

## 🖥️ Workflow Visualization

```mermaid
flowchart TD
    A[User provides document] --> B[Preprocess (clean/load)]
    B --> C[Summarize text]
    C --> D[Detect risks]
    D --> E[Classify document type]
    E --> F[Visualize results]
    F --> G[Output: Console + Files]
```

**Typical Outputs:**
- **Console:** Summary, risk factors, classification, visualization paths
- **Files:** Summary `.txt`, risk report, visualizations (word cloud, risk chart, category distribution), full analysis `.json`

---

## 🛠️ Installation

```bash
git clone https://github.com/Madan2418/ML-legal-Document-classifier.git
cd ML-legal-Document-classifier
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

---

## ⚡ Usage

To analyze a legal document:
```bash
python legal_doc_analyzer_ml.py path/to/document.pdf --output results --verbose
```

**Command Line Options:**
- `file_path` – Path to the document [required]
- `--output, -o` – Output directory for results (default: `results/`)
- `--verbose, -v` – Prints detailed logs

---

## 🔍 Customization

### Add Custom Risk Terms
Create a JSON file:
```json
{
    "termination": ["terminate", "expiry", "notice period"],
    "liability": ["indemnify", "damages", "hold harmless"]
}
```
Pass this file to `RiskDetector`.

### Train Custom Models
Prepare a CSV/JSON with 'text' and 'label', then:
```python
from models import train_document_classifier
results = train_document_classifier(
    data_path='path/to/training_data.csv',
    output_dir='models',
    model_type='random_forest'
)
```

---

## 📊 Visualization Examples

- **Word Cloud:** Most frequent terms from document.
- **Risk Factors Chart:** Severity/category bar chart for detected risks.
- **Category Distribution:** Pie chart showing classification confidence.
- **Text Statistics:** Word count, sentence count, average word length.

---

## 🧑‍💻 For Developers

- **Test Core Modules:**  
  Run `test_analyzer.py` for basic component checks.
- **Retrain Model:**  
  Use `train_classifier.py` for new training data.

---

## 📚 Dependencies

- Python 3.7+
- NLTK, spaCy, scikit-learn, matplotlib, seaborn, wordcloud, PyPDF2, python-magic, etc.

---

## 📄 License

MIT License

---

## ❤️ Acknowledgments

- Built with Python
- NLP powered by spaCy & NLTK
- Visualizations with Matplotlib & Seaborn

---

## 👋 Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
