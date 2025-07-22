import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from model import DocumentProcessor
import PyPDF2

def load_training_data(data_dir: str) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """Load training data from the synthetic dataset, including the case path."""
    training_pairs = []
    
    for case_dir_name in os.listdir(data_dir):
        if not case_dir_name.startswith('case_'):
            continue
            
        case_path = os.path.join(data_dir, case_dir_name)
        input_path = os.path.join(case_path, 'input.json')
        output_path = os.path.join(case_path, 'output.json')
        
        if not (os.path.exists(input_path) and os.path.exists(output_path)):
            continue
            
        with open(input_path, 'r') as f:
            input_json = json.load(f)
        with open(output_path, 'r') as f:
            output_json = json.load(f)
            
        training_pairs.append((input_json, output_json, case_path))
    
    return training_pairs

def get_full_section_text(pdf_path: str, page_number: int, section_title: str) -> str:
    """Finds and returns the full text of a section from a PDF page."""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            if page_number > len(pdf_reader.pages):
                return section_title  # Fallback
            
            page_text = pdf_reader.pages[page_number - 1].extract_text()
            
        # Find the section text using the title as an anchor.
        title_pos = page_text.find(section_title)
        if title_pos == -1:
            return section_title  # Fallback if title not found exactly

        # Find the end of the section (marked by a double newline or end of page)
        rest_of_page = page_text[title_pos:]
        end_of_section_pos = rest_of_page.find('\n\n')
        
        if end_of_section_pos == -1:
            full_text = rest_of_page
        else:
            full_text = rest_of_page[:end_of_section_pos]
            
        return full_text.strip()
    except Exception:
        return section_title # Fallback on any error

def prepare_features(processor: DocumentProcessor, training_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features and labels for training using full section text."""
    section_features = []
    section_labels = []
    subsection_features = []
    subsection_labels = []
    
    print("Extracting features from full PDF text...")
    for input_json, output_json, case_path in training_pairs:
        persona = input_json['persona']
        job = input_json['job_to_be_done']
        input_dir = os.path.join(case_path, 'input')
        
        # Process sections from ground truth
        for section in output_json['extracted_sections']:
            pdf_path = os.path.join(input_dir, section['document'])
            if not os.path.exists(pdf_path):
                continue

            full_text = get_full_section_text(pdf_path, section['page_number'], section['section_title'])
            features = processor.get_section_features(full_text, persona, job)
            section_features.append(features[0])
            section_labels.append(section['importance_rank'])
        
        # Process subsections from ground truth
        for subsection in output_json['sub_section_analysis']:
            pdf_path = os.path.join(input_dir, subsection['document'])
            if not os.path.exists(pdf_path):
                continue
                
            # Here we use 'refined_text' as a proxy to find the relevant section
            # This part can be improved if we had more structured subsection data
            full_text = get_full_section_text(pdf_path, subsection['page_number'], subsection['refined_text'].split('insights into ')[-1])
            features = processor.get_section_features(full_text, persona, job)
            subsection_features.append(features[0])
            subsection_labels.append(subsection['relevance_score'])
    
    return (np.array(section_features), np.array(section_labels),
            np.array(subsection_features), np.array(subsection_labels))

def train_model(data_dir: str, save_path: str):
    """Train the model using the synthetic dataset."""
    processor = DocumentProcessor()
    
    print("Loading training data...")
    training_pairs = load_training_data(data_dir)
    print(f"Loaded {len(training_pairs)} training cases")
    
    print("Preparing features...")
    section_features, section_labels, subsection_features, subsection_labels = prepare_features(processor, training_pairs)
    
    # Scale features
    processor.scaler.fit(section_features)
    section_features_scaled = processor.scaler.transform(section_features)
    subsection_features_scaled = processor.scaler.transform(subsection_features)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        section_features_scaled, section_labels, test_size=0.2, random_state=42)
    
    # Train section ranker
    print("Training section ranker...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    processor.section_ranker = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Train subsection ranker
    print("Training subsection ranker...")
    X_train, X_val, y_train, y_val = train_test_split(
        subsection_features_scaled, subsection_labels, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    processor.subsection_ranker = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Save the trained model
    print(f"Saving model to {save_path}...")
    processor.save_model(save_path)
    print("Training complete!")

if __name__ == "__main__":
    data_dir = "Adobe-1A/round1b_high_quality"
    save_path = "Adobe-1A/round1b/trained_model"
    train_model(data_dir, save_path) 