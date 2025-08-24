# Challenge 1B: Persona-Driven Document Intelligence

## Overview
This solution analyzes document collections based on specific personas and tasks, extracting and ranking the most relevant sections for informed decision-making.

## Quick Start

### Docker Execution
```bash
# Build the image
docker build --platform linux/amd64 -t challenge1b:latest .

# Run with document collection
docker run --rm \
  -v $(pwd)/collection:/app/collection \
  --network none \
  challenge1b:latest
```

### Input Structure
Your collection directory should contain:
```
collection/
├── challenge1b_input.json    # Configuration file
├── PDFs/                     # PDF documents
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
└── challenge1b_output.json  # Generated output
```

### Input Configuration Format
Create `challenge1b_input.json`:
```json
{
  "documents": [
    {"filename": "document1.pdf"},
    {"filename": "document2.pdf"}
  ],
  "persona": {
    "role": "Your persona description"
  },
  "job_to_be_done": {
    "task": "Your specific task"
  }
}
```

## Local Development

### Prerequisites
```bash
pip install PyMuPDF langid
```

### Run Locally
```bash
# Setup your collection directory
mkdir -p my_collection/PDFs

# Create input configuration
# (Copy sample_challenge1b_input.json as template)
cp sample_challenge1b_input.json my_collection/challenge1b_input.json

# Add your PDF files to my_collection/PDFs/
# cp your_pdfs/*.pdf my_collection/PDFs/

# Run the analysis
python main.py my_collection

# Check output: my_collection/challenge1b_output.json
```

### Example Collection Setup
```bash
my_collection/
├── challenge1b_input.json    # Your persona/task configuration
├── PDFs/                     # Your PDF documents
│   ├── document1.pdf
│   └── document2.pdf
└── challenge1b_output.json  # Generated analysis (after running)
```

## Output Format
The system generates `challenge1b_output.json` containing:
- **Metadata**: Input documents, persona, task, timestamp
- **Extracted Sections**: Ranked relevant sections with importance scores
- **Subsection Analysis**: Detailed text analysis of top sections

## Performance
- **Processing Time**: <60 seconds for 3-5 documents
- **Model Size**: <1GB
- **Architecture**: CPU-only, AMD64 