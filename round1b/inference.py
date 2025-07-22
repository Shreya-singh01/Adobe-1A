import os
import json
import time
from model import DocumentProcessor

def process_input(input_path: str, output_path: str, model_path: str):
    """Process a single input JSON file and generate output."""
    # Load the trained model
    processor = DocumentProcessor(model_path)
    
    # Read input JSON
    with open(input_path, 'r') as f:
        input_json = json.load(f)
    
    # Process documents
    start_time = time.time()
    output_json = processor.process_documents(input_json)
    processing_time = time.time() - start_time
    
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Save output JSON
    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    model_path = "Adobe-1A/round1b/trained_model"
    input_path = "path/to/input.json"
    output_path = "path/to/output.json"
    
    if not os.path.exists(model_path):
        print("Error: Model not found. Please train the model first.")
        exit(1)
    
    process_input(input_path, output_path, model_path) 