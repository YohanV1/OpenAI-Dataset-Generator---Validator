# OpenAI Dataset Generator & Validator

A Streamlit application for creating, validating, and managing fine-tuning datasets for OpenAI models.

## Features

- Create JSONL format datasets for fine-tuning
- Validate dataset format and structure
- Upload datasets directly to OpenAI
- Create and monitor fine-tuning jobs
- Track fine-tuned model status
- Analyze token distribution and dataset statistics

## Prerequisites

- Python 3.7+
- OpenAI API key
- Streamlit
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd openai-dataset-generator
```

2. Install required packages:
```bash
pip install streamlit openai python-dotenv tiktoken numpy
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Follow the step-by-step process in the web interface:

   - **Step 1:** Create a fine-tuning dataset
     - Add a system message
     - Provide training examples (minimum 10)
     - Generate and download JSONL file
   
   - **Step 2:** Verify data format and structure
     - Check for format errors
     - Review token distribution
     - Analyze conversation lengths
   
   - **Step 3:** Upload dataset to OpenAI
     - Automatically uploads JSONL file
     - Receives file ID for tracking
   
   - **Step 4:** Create fine-tuning job
     - Initialize training process
     - Uses gpt-3.5-turbo model
   
   - **Step 5:** Monitor fine-tuned model
     - Track training progress
     - View model statistics and parameters
     - Get fine-tuned model identifier

## Dataset Validation Checks

The validator performs comprehensive checks including:
- Data format and structure validation
- Token count analysis
- Message distribution statistics
- System and user message presence
- Token limit compliance (16,385 tokens per example)
- Pricing estimation based on token usage