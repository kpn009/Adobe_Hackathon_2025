# Challenge 1A: PDF Structure Extraction

## Overview
This solution extracts document titles and hierarchical headings (H1, H2, H3) from PDF files with multilingual support and robust document structure analysis.

## Approach

### Key Features
- **Multilingual Support**: 80+ languages with automatic detection
- **Smart Heading Detection**: Font analysis with content validation
- **Hierarchical Structure**: Automatic heading level determination
- **TOC Integration**: Built-in table of contents support

## Dependencies
- **PyMuPDF**: PDF parsing and text extraction
- **langid**: Language detection

## Directory Structure
```
/app/
├── input/          # Input PDF files
│   ├── document1.pdf
│   └── document2.pdf
├── output/         # Generated JSON outputs
│   ├── document1.json
│   └── document2.json
└── process_pdfs.py # Main processing script
```

## Build and Run Instructions

### Docker Build
```bash
docker build --platform linux/amd64 -t challenge1a:latest .
```

### Docker Run
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1a:latest
```

### Local Development
```bash
# Install dependencies
pip install PyMuPDF langid

# Create input/output directories
mkdir -p input output

# Add PDF files to input directory
# cp your_pdfs/*.pdf input/

# Run directly (processes from ./input to ./output locally)
python process_pdfs.py
```

### Testing Your Setup
```bash
# 1. Place PDF files in input directory
# 2. Run the Docker container
# 3. Check output directory for generated JSON files
# Each filename.pdf will generate filename.json
```

## Output Format
```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Chapter 1", "page": 1 },
    { "level": "H2", "text": "Section 1.1", "page": 2 },
    { "level": "H3", "text": "Subsection 1.1.1", "page": 3 }
  ]
}
```

## Performance
- **Processing Time**: <10 seconds for 50-page PDFs
- **CPU Architecture**: AMD64 (x86_64)
- **Memory**: <200MB model size 