import os
import json
import time
import re
from datetime import datetime
import numpy as np
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import PyPDF2

class DocumentProcessor:
    def __init__(self, model_path: str = None):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize XGBoost models
        if model_path and os.path.exists(os.path.join(model_path, 'section_ranker.json')):
            self.section_ranker = xgb.Booster()
            self.section_ranker.load_model(os.path.join(model_path, 'section_ranker.json'))
            self.subsection_ranker = xgb.Booster()
            self.subsection_ranker.load_model(os.path.join(model_path, 'subsection_ranker.json'))
            self.scaler = StandardScaler()
            # Load scaler parameters if available
            if os.path.exists(os.path.join(model_path, 'scaler.json')):
                with open(os.path.join(model_path, 'scaler.json'), 'r') as f:
                    scaler_params = json.load(f)
                self.scaler.mean_ = np.array(scaler_params['mean'])
                self.scaler.scale_ = np.array(scaler_params['scale'])
        else:
            self.section_ranker = None
            self.subsection_ranker = None
            self.scaler = StandardScaler()

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with page numbers."""
        text_by_page = {}
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text = pdf_reader.pages[page_num].extract_text()
                text_by_page[page_num + 1] = text
        return text_by_page

    def get_section_features(self, section_text: str, persona: str, job: str) -> np.ndarray:
        """Generate features for section ranking."""
        # Embed texts
        section_embedding = self.embedding_model.encode(section_text)
        persona_embedding = self.embedding_model.encode(persona)
        job_embedding = self.embedding_model.encode(job)
        
        # Calculate similarities
        persona_similarity = np.dot(section_embedding, persona_embedding) / (
            np.linalg.norm(section_embedding) * np.linalg.norm(persona_embedding))
        job_similarity = np.dot(section_embedding, job_embedding) / (
            np.linalg.norm(section_embedding) * np.linalg.norm(job_embedding))
        
        # Additional features
        word_count = len(section_text.split())
        
        features = np.array([
            persona_similarity,
            job_similarity,
            word_count,
            len(section_text),
            len(section_text.split('\n'))
        ])
        
        return features.reshape(1, -1)

    def process_documents(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents and generate output JSON."""
        persona = input_json['persona']
        job = input_json['job_to_be_done']
        
        # In a real scenario, you would get the document paths from the input_json
        # For this example, let's assume the paths are correct.
        documents = input_json['documents']
        
        extracted_sections = []
        sub_section_analysis = []
        
        section_pattern = re.compile(r"Section \d+:.*")

        for doc in documents:
            text_by_page = self.extract_text_from_pdf(doc)
            
            for page_num, text in text_by_page.items():
                # Use regex to find structured sections
                found_sections = section_pattern.finditer(text)
                
                section_spans = [match.span() for match in found_sections]
                
                for i, start_span in enumerate(section_spans):
                    start_index = start_span[0]
                    # Determine the end of the section
                    if i + 1 < len(section_spans):
                        end_index = section_spans[i+1][0]
                    else:
                        end_index = len(text)
                        
                    section_full_text = text[start_index:end_index].strip()
                    section_title = section_full_text.split('\n')[0]

                    if len(section_full_text) < 10:
                        continue
                    
                    features = self.get_section_features(section_full_text, persona, job)
                    if self.section_ranker:
                        features_scaled = self.scaler.transform(features)
                        importance_score = float(self.section_ranker.predict(xgb.DMatrix(features_scaled))[0])
                    else:
                        importance_score = float(features[0][1])
                    
                    if importance_score > 0.5:
                        extracted_sections.append({
                            "document": doc,
                            "page_number": page_num,
                            "section_title": section_title,
                            "importance_rank": round(importance_score, 2)
                        })
                        
                        if importance_score > 0.7:
                            sub_section_analysis.append({
                                "document": doc,
                                "page_number": page_num,
                                "refined_text": f"This section on '{section_title}' provides key insights.",
                                "relevance_score": round(importance_score, 2)
                            })
        
        # Sort by importance/relevance
        extracted_sections.sort(key=lambda x: x['importance_rank'], reverse=True)
        sub_section_analysis.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            "metadata": {
                "input_documents": documents,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            "extracted_sections": extracted_sections,
            "sub_section_analysis": sub_section_analysis
        }

    def save_model(self, save_path: str):
        """Save trained models and scaler."""
        os.makedirs(save_path, exist_ok=True)
        if self.section_ranker:
            self.section_ranker.save_model(os.path.join(save_path, 'section_ranker.json'))
        if self.subsection_ranker:
            self.subsection_ranker.save_model(os.path.join(save_path, 'subsection_ranker.json'))
        # Save scaler parameters
        scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist()
        }
        with open(os.path.join(save_path, 'scaler.json'), 'w') as f:
            json.dump(scaler_params, f) 