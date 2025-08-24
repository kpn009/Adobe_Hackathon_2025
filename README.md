# Adobe PDF Challenge Solutions - TEAM AK97

This repository contains solutions for Adobe's PDF processing challenges, implementing advanced document structure analysis and persona-driven intelligence systems.

## Repository Structure

```
root
├── challenge1a/              # PDF Structure Extraction Challenge
│   ├── process_pdfs.py       # Main processing script
│   ├── Dockerfile            # Container configuration
│   ├── requirements.txt      # Python dependencies
│   └── README.md            # Detailed documentation
│
├── challenge1b/              # Persona-Driven Document Intelligence
   ├── main.py              # Main analysis script
   ├── Dockerfile           # Container configuration
   ├── requirements.txt     # Python dependencies
   ├── approach_explanation.md  # Methodology explanation
   ├── README.md            # Execution instructions
   └── sample_collection/   # Example directory structure


```

## Challenge 1A: PDF Structure Extraction

**Objective**: Extract document titles and hierarchical headings (H1, H2, H3) from PDF files.

**Key Features**:
- Multilingual support (80+ languages)
- Font-based and content-based heading detection
- Built-in table of contents support
- Robust filtering of metadata and boilerplate content

**Quick Start**:
```bash
cd challenge1a
docker build --platform linux/amd64 -t challenge1a:latest .
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1a:latest
```

## Challenge 1B: Persona-Driven Document Intelligence

**Objective**: Analyze document collections and extract relevant sections based on user personas and specific tasks.

**Key Features**:
- Dynamic relevance scoring without domain-specific knowledge
- Multi-faceted semantic analysis
- Hierarchical content extraction
- Support for diverse domains (academic, business, educational)

**Quick Start**:
```bash
cd challenge1b
docker build --platform linux/amd64 -t challenge1b:latest .
docker run --rm -v $(pwd)/collection:/app/collection --network none challenge1b:latest
```

## Technical Specifications

### Common Requirements
- **Python Version**: 3.6+ (recommended 3.9+)
- **Dependencies**: PyMuPDF, langid
- **Architecture**: AMD64 (x86_64)
- **Network**: Offline operation
- **Memory**: Optimized for 16GB RAM systems

### Challenge 1A Constraints
- **Processing Time**: ≤10 seconds per 50-page PDF
- **Model Size**: ≤200MB
- **Input**: Individual PDF files
- **Output**: JSON with title and hierarchical outline

### Challenge 1B Constraints
- **Processing Time**: ≤60 seconds for 3-5 documents
- **Model Size**: ≤1GB
- **Input**: Document collection with persona/task configuration
- **Output**: Ranked sections and subsection analysis

## Installation

### Prerequisites
```bash
# Install Python 3.9+
pip install PyMuPDF>=1.20.0 langid>=1.1.6
```

### Docker (Recommended)
Each challenge includes its own Dockerfile for containerized execution with all dependencies pre-configured.

## Development

### Local Testing
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare input data according to challenge specifications
3. Run the appropriate script with required arguments

## Innovation Highlights

- **Zero-shot Domain Adaptation**: No hardcoded domain knowledge
- **Multilingual Intelligence**: Support for major world languages
- **Robust Content Analysis**: Advanced filtering and validation
- **Hierarchical Understanding**: Maintains document structure relationships
- **Performance Optimization**: Efficient processing within time constraints

## License

This solution is developed for Adobe's PDF Challenge competition.

## Support

For technical questions or issues, refer to the individual challenge README files for detailed documentation and troubleshooting guides. 