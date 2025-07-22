import os
import json
from model import DocumentProcessor

def test_and_compare_model(case_path: str):
    """Test the trained model on a sample case and compare with ground truth."""
    # Load the trained model
    model_path = "Adobe-1A/round1b/trained_model"
    processor = DocumentProcessor(model_path)
    
    # Load input and ground truth output
    input_path = os.path.join(case_path, 'input.json')
    ground_truth_path = os.path.join(case_path, 'output.json')

    with open(input_path, 'r') as f:
        input_json = json.load(f)
    with open(ground_truth_path, 'r') as f:
        ground_truth_json = json.load(f)
    
    # Update document paths to be absolute
    input_dir = os.path.join(case_path, 'input')
    input_json['documents'] = [os.path.join(input_dir, doc) for doc in input_json['documents']]
    
    print("--- Test Case ---")
    print(f"Persona: {input_json['persona']}")
    print(f"Job: {input_json['job_to_be_done']}")
    print("-" * 20)
    
    # Process documents to get model's prediction
    print("Running model...")
    model_output_json = processor.process_documents(input_json)
    print("Model processing complete.\n")

    # --- Comparison ---
    print("--- Comparison of Top 5 Extracted Sections ---")
    
    print("\n--- Ground Truth Output ---")
    if ground_truth_json['extracted_sections']:
        for i, section in enumerate(ground_truth_json['extracted_sections'][:5]):
            print(f"{i+1}. Rank: {section['importance_rank']} | Page: {section['page_number']} | Doc: {os.path.basename(section['document'])}")
            print(f"   Title: {section['section_title']}")
    else:
        print("No sections in ground truth.")

    print("\n--- Model Generated Output ---")
    if model_output_json['extracted_sections']:
        for i, section in enumerate(model_output_json['extracted_sections'][:5]):
            print(f"{i+1}. Rank: {section['importance_rank']} | Page: {section['page_number']} | Doc: {os.path.basename(section['document'])}")
            print(f"   Title: {section['section_title']}")
    else:
        print("No sections extracted by the model.")

    print("\n--- End of Comparison ---")


if __name__ == "__main__":
    # Specify the test case to use
    test_case_path = "Adobe-1A/round1b_high_quality/case_02_investment_analyst"
    test_and_compare_model(test_case_path) 