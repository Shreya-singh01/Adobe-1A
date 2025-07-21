# Round 1B Dataset - Adobe India Hackathon 2025

## Overview
This dataset contains 10 different test cases for Round 1B of the Adobe India Hackathon 2025. Each test case includes:
- A specific persona with role, expertise, and focus areas
- A job-to-be-done description
- Multiple PDF documents related to the domain
- Sample input and output JSON files

## Dataset Structure
```
round1b_dataset/
├── input/                    # Input JSON files for each test case
├── output/                   # Sample output JSON files
└── README.md                 # This file
```

## Test Cases

### 1. Academic Research - Machine Learning
- **Persona:** PhD Researcher in Machine Learning
- **Job:** Literature review on deep learning advances
- **Documents:** 3 research papers on ML topics
- **Domain:** Academic Research

### 2. Business Analysis - Tech Companies
- **Persona:** Investment Analyst
- **Job:** Analyze tech company performance and strategies
- **Documents:** 3 annual reports from major tech companies
- **Domain:** Business Analysis

### 3. Educational Content - Chemistry
- **Persona:** Undergraduate Chemistry Student
- **Job:** Exam preparation on organic chemistry
- **Documents:** 3 chemistry textbooks/guides
- **Domain:** Educational Content

### 4. Healthcare Research - Drug Discovery
- **Persona:** Drug Discovery Researcher
- **Job:** Evaluate AI applications in drug discovery
- **Documents:** 2 research papers on AI in healthcare
- **Domain:** Healthcare Research

### 5. Financial Analysis - Banking Sector
- **Persona:** Financial Analyst
- **Job:** Assess banking sector performance and digital transformation
- **Documents:** 2 annual reports from major banks
- **Domain:** Financial Analysis

### 6. Environmental Science - Climate Change
- **Persona:** Environmental Scientist
- **Job:** Analyze climate change impacts and renewable energy
- **Documents:** 2 research papers on environmental topics
- **Domain:** Environmental Science

### 7. Marketing Research - Consumer Behavior
- **Persona:** Digital Marketing Manager
- **Job:** Develop insights into digital marketing trends
- **Documents:** 2 marketing research papers
- **Domain:** Marketing Research

### 8. Legal Research - Intellectual Property
- **Persona:** Intellectual Property Lawyer
- **Job:** Research IP law challenges in technology sector
- **Documents:** 2 legal research papers
- **Domain:** Legal Research

### 9. Supply Chain Management
- **Persona:** Supply Chain Analyst
- **Job:** Evaluate supply chain optimization and challenges
- **Documents:** 2 supply chain research papers
- **Domain:** Supply Chain Management

### 10. Cybersecurity Research
- **Persona:** Cybersecurity Researcher
- **Job:** Analyze cybersecurity threats and AI applications
- **Documents:** 2 cybersecurity research papers
- **Domain:** Cybersecurity Research

## Input Format
Each input JSON file contains:
```json
{
  "test_case_id": "unique_identifier",
  "persona": {
    "role": "Job title/role",
    "expertise": "Areas of expertise",
    "focus_areas": "Specific focus areas"
  },
  "job_to_be_done": "Detailed description of the task",
  "documents": ["list", "of", "pdf", "files"],
  "domain": "Domain category"
}
```

## Expected Output Format
Your model should output JSON in the format specified in the hackathon requirements:
```json
{
  "metadata": {
    "input_documents": ["list", "of", "input", "documents"],
    "persona": {...},
    "job_to_be_done": "task description",
    "processing_timestamp": "ISO timestamp"
  },
  "extracted_sections": [
    {
      "document": "filename.pdf",
      "page_number": 1,
      "section_title": "Section Title",
      "importance_rank": 0.95
    }
  ],
  "sub_section_analysis": [
    {
      "document": "filename.pdf",
      "page_number": 1,
      "refined_text": "Extracted and refined text content",
      "relevance_score": 0.92
    }
  ]
}
```

## Usage Instructions

### For Testing Your Round 1B Model:
1. **Input Processing:**
   - Read the input JSON files from `input/` directory
   - Extract persona, job-to-be-done, and document list
   - Process the referenced PDF documents

2. **Model Execution:**
   - Use your Round 1A model to extract document structure
   - Apply persona-driven analysis to identify relevant sections
   - Rank sections by importance based on job requirements
   - Extract and refine relevant text content

3. **Output Generation:**
   - Generate output JSON matching the expected format
   - Include metadata with processing timestamp
   - Provide extracted sections with importance rankings
   - Include sub-section analysis with relevance scores

### For Evaluation:
- Compare your output with the sample output files
- Ensure all required fields are present
- Verify that extracted sections are relevant to the persona and job
- Check that importance rankings and relevance scores are reasonable

## Key Requirements for Round 1B:
- **Processing Time:** ≤ 60 seconds for 3-5 documents
- **Model Size:** ≤ 1GB
- **No Internet Access:** Must work offline
- **CPU Only:** No GPU dependencies
- **Generic Solution:** Must work across diverse domains

## Integration with Round 1A:
Your Round 1B solution should:
1. Use your Round 1A model to extract document structure
2. Apply persona-driven filtering to identify relevant sections
3. Rank sections by importance for the specific job
4. Extract and refine text content based on relevance

## Notes:
- The PDF files referenced in the input are placeholders
- You'll need to create or obtain actual PDF documents for testing
- Consider creating text versions of documents for easier processing
- Test with various document types and formats
- Ensure your solution generalizes across different domains and personas 