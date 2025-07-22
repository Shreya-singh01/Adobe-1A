#!/usr/bin/env python3
"""
Round 1B Model Runner - Persona-Driven Document Intelligence

This script allows you to easily run the trained model on your own PDFs.
Simply provide the PDF files, define a persona and job, and get ranked results.
"""

import os
import json
import argparse
from datetime import datetime
from model import DocumentProcessor

def run_model_on_pdfs(pdf_paths, persona, job_to_be_done, output_file="output.json"):
    """
    Run the trained model on a collection of PDFs.
    
    Args:
        pdf_paths (list): List of paths to PDF files
        persona (str): Description of the persona (e.g., "Investment Analyst")
        job_to_be_done (str): Specific task to accomplish
        output_file (str): Path to save the output JSON
    """
    
    # Validate PDF files exist
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {pdf_path}")
    
    print("=" * 60)
    print("ROUND 1B MODEL - PERSONA-DRIVEN DOCUMENT INTELLIGENCE")
    print("=" * 60)
    print(f"Persona: {persona}")
    print(f"Job to be done: {job_to_be_done}")
    print(f"Number of PDFs: {len(pdf_paths)}")
    print("-" * 60)
    
    # --- FIX: Build path relative to this script's location ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, "trained_model")
    
    # Load the trained model
    print("Loading trained model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")
    
    processor = DocumentProcessor(model_path)
    print("Model loaded successfully!")
    
    # Prepare input JSON
    input_json = {
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "documents": pdf_paths
    }
    
    # Process documents
    print(f"\nProcessing {len(pdf_paths)} PDF documents...")
    import time
    start_time = time.time()
    
    output_json = processor.process_documents(input_json)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Display results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total sections extracted: {len(output_json['extracted_sections'])}")
    print(f"Sub-section analyses: {len(output_json['sub_section_analysis'])}")
    
    if output_json['extracted_sections']:
        print(f"\nTop 5 Most Relevant Sections:")
        for i, section in enumerate(output_json['extracted_sections'][:5]):
            doc_name = os.path.basename(section['document'])
            print(f"{i+1}. {doc_name} (Page {section['page_number']})")
            print(f"   Title: {section['section_title']}")
            print(f"   Importance: {section['importance_rank']}")
            print()
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    
    print(f"Full results saved to: {output_file}")
    print("=" * 60)
    
    return output_json

def main():
    parser = argparse.ArgumentParser(
        description="Run Round 1B model on PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_model.py --pdfs doc1.pdf doc2.pdf --persona "Investment Analyst" --job "Analyze revenue trends"
  
  # With custom output file
  python run_model.py --pdfs *.pdf --persona "PhD Researcher" --job "Literature review" --output results.json
  
  # Interactive mode
  python run_model.py --interactive
        """
    )
    
    parser.add_argument('--pdfs', nargs='+', help='Paths to PDF files')
    parser.add_argument('--persona', help='Description of the persona')
    parser.add_argument('--job', help='Job to be done')
    parser.add_argument('--output', default='output.json', help='Output JSON file path')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        print("=" * 60)
        print("ROUND 1B MODEL - INTERACTIVE MODE")
        print("=" * 60)
        
        # Get PDF paths
        print("\nEnter the paths to your PDF files (one per line, press Enter twice when done):")
        pdf_paths = []
        while True:
            path = input("PDF path: ").strip()
            if not path:
                break
            pdf_paths.append(path)
        
        if not pdf_paths:
            print("No PDF files provided. Exiting.")
            return
        
        # Get persona
        persona = input("\nEnter the persona description: ").strip()
        if not persona:
            print("No persona provided. Exiting.")
            return
        
        # Get job
        job = input("Enter the job to be done: ").strip()
        if not job:
            print("No job provided. Exiting.")
            return
        
        # Get output file
        output_file = input("Enter output file path (default: output.json): ").strip()
        if not output_file:
            output_file = "output.json"
    
    else:
        # Command line mode
        if not args.pdfs:
            parser.error("--pdfs is required (or use --interactive)")
        if not args.persona:
            parser.error("--persona is required (or use --interactive)")
        if not args.job:
            parser.error("--job is required (or use --interactive)")
        
        pdf_paths = args.pdfs
        persona = args.persona
        job = args.job
        output_file = args.output
    
    # --- FIX: Check if output path is a directory ---
    if os.path.isdir(output_file):
        print(f"Warning: Output path is a directory. Appending default filename 'output.json'.")
        output_file = os.path.join(output_file, "output.json")

    try:
        # Run the model
        result = run_model_on_pdfs(pdf_paths, persona, job, output_file)
        print("\n✅ Model execution completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 