# Round 1B Model - Persona-Driven Document Intelligence

This model takes a collection of PDF documents, a persona, and a job-to-be-done, then extracts and ranks relevant sections based on the persona's expertise and job requirements.

## Model Architecture

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (80MB)
- **Ranking Models**: XGBoost for section and subsection ranking
- **Total Model Size**: ~300MB
- **Processing Time**: ~15-20 seconds for 3-5 documents

## Installation

```bash
pip install -r requirements.txt
```

## Training

The model has been trained on the synthetic dataset:

```bash
python train.py
```

This will:
1. Load the synthetic dataset (15 personas × 7-10 documents each)
2. Train XGBoost models for section and subsection ranking
3. Save the trained models to `trained_model/`

## Usage

### Basic Usage

```python
from model import DocumentProcessor

# Load the trained model
processor = DocumentProcessor("trained_model")

# Prepare input
input_json = {
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a comprehensive literature review on Graph Neural Networks for drug discovery.",
    "documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
}

# Process documents
output_json = processor.process_documents(input_json)
```

### Input Format

```json
{
    "persona": "Role description",
    "job_to_be_done": "Specific task to accomplish",
    "documents": ["path/to/doc1.pdf", "path/to/doc2.pdf"]
}
```

### Output Format

```json
{
    "metadata": {
        "input_documents": ["doc1.pdf", "doc2.pdf"],
        "persona": "Role description",
        "job_to_be_done": "Specific task",
        "processing_timestamp": "2025-07-21T09:36:24Z"
    },
    "extracted_sections": [
        {
            "document": "doc1.pdf",
            "page_number": 3,
            "section_title": "Section Title",
            "importance_rank": 0.87
        }
    ],
    "sub_section_analysis": [
        {
            "document": "doc1.pdf",
            "page_number": 3,
            "refined_text": "Summary of the section",
            "relevance_score": 0.87
        }
    ]
}
```

## Testing

Test the model on a sample case:

```bash
python test_model.py
```

## Model Performance

- **Accuracy**: Good performance on diverse personas and document types
- **Speed**: Processes 3-5 documents in 15-20 seconds
- **Memory**: Uses ~300MB total model size
- **Offline**: Works completely offline

## Files

- `model.py`: Core model implementation
- `train.py`: Training script
- `inference.py`: Inference script
- `test_model.py`: Test script
- `requirements.txt`: Dependencies
- `trained_model/`: Saved trained models 