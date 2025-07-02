# DocuLogix

> Engineering document/submittal tracker with PaddleOCR integration

## Overview
DocuLogix is a modern engineering document/submittal tracker enhanced with state-of-the-art OCR capabilities using PaddleOCR 3.x. This integration provides comprehensive document analysis including text recognition, table extraction, and layout analysis for engineering workflows.

## Features
- **Advanced OCR Integration**: PaddleOCR 3.x with PP-OCRv5, PP-StructureV3, and PP-ChatOCRv4
- **Multi-language Support**: 13+ languages including Chinese, English, Korean, Japanese
- **Document Structure Analysis**: Table recognition, layout detection, and form processing
- **MCP Server**: Model Context Protocol server for AI assistant integration
- **REST API**: FastAPI-based web service for document processing
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Quick Start

### Prerequisites
- Python 3.8+ 
- Git
- Optional: CUDA-capable GPU for acceleration

### Installation

```bash
git clone https://github.com/zbest1000/DocuLogix.git
cd DocuLogix

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install paddlepaddle-gpu>=3.0.0
```

### Basic Usage

#### 1. Python API
```python
from src.ocr import PaddleOCREngine

# Initialize OCR engine
ocr = PaddleOCREngine(
    lang="en",  # or "ch" for Chinese
    use_gpu=True,  # Enable GPU acceleration
    enable_structure=True  # Enable table/layout analysis
)

# Process an image
result = ocr.process_image("path/to/document.jpg")

# Get extracted text
print(result.get_full_text())

# Get table data
for table in result.tables:
    print(table.to_csv_rows())

# Save results
result.save_to_file("output.json", format="json")
```

#### 2. MCP Server
```bash
# Start MCP server with SSE transport
python scripts/start_mcp_server.py --transport sse --port 8000

# Start with GPU acceleration and English language
python scripts/start_mcp_server.py --gpu --lang en
```

#### 3. REST API Server
```python
from src.api import create_app
import uvicorn

# Create FastAPI app
app = create_app(ocr_config={"lang": "en", "use_gpu": True})

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Access API documentation at: `http://localhost:8000/docs`

## API Endpoints

### REST API
- `POST /ocr/image` - Process base64 encoded image
- `POST /ocr/file` - Process uploaded image file  
- `POST /ocr/pdf` - Process PDF document
- `GET /ocr/languages` - Get supported languages
- `POST /ocr/language` - Change OCR language
- `GET /health` - Health check

### MCP Server Tools
- `ocr_image` - Perform OCR on image data
- `ocr_pdf` - Process PDF files
- `set_ocr_language` - Change language settings

## Supported Features

### OCR Models
- **PP-OCRv5**: Latest high-accuracy text recognition
- **PP-OCRv4**: Stable production-ready models
- **PP-StructureV3**: Advanced document structure analysis
- **PP-ChatOCRv4**: AI-powered document understanding

### Languages
- Chinese (Simplified/Traditional)
- English  
- Korean, Japanese
- Arabic, Latin, Cyrillic
- Devanagari, Tamil, Telugu, Kannada
- And more...

### Document Types
- Scanned documents
- Photos of text
- PDFs (single/multi-page)
- Forms and tables
- Engineering drawings
- Handwritten text

## Configuration

### OCR Engine Options
```python
ocr_config = {
    "lang": "ch",  # Language code
    "use_gpu": True,  # GPU acceleration
    "use_doc_orientation": True,  # Auto-rotate documents
    "use_doc_unwarping": True,  # Correct document distortion
    "use_textline_orientation": True,  # Fix text orientation
    "ocr_version": "PP-OCRv5",  # Model version
    "det_model": "PP-OCRv5_server_det",  # Detection model
    "rec_model": "PP-OCRv5_server_rec",  # Recognition model
    "enable_structure": True  # Table/layout analysis
}
```

### Environment Variables
```bash
# Optional GPU configuration
export CUDA_VISIBLE_DEVICES=0

# Logging level
export LOG_LEVEL=INFO
```

## Advanced Usage

### Batch Processing
```python
# Process multiple PDFs
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = []

for pdf_file in pdf_files:
    result = ocr.process_pdf(pdf_file)
    results.extend(result)

# Merge results
from src.ocr.utils import merge_ocr_results
combined = merge_ocr_results(results)
```

### Custom Filtering
```python
from src.ocr.utils import filter_by_keywords

# Filter results by keywords
keywords = ["invoice", "total", "amount"]
filtered = filter_by_keywords(result, keywords)
```

### Structured Data Extraction
```python
from src.ocr.utils import extract_structured_data

# Extract key-value pairs and tables
structured = extract_structured_data(result)
print(structured["key_value_pairs"])
```

## Development

### Project Structure
```
DocuLogix/
├── src/
│   ├── ocr/
│   │   ├── engine.py          # Main OCR engine
│   │   ├── models.py          # Data models
│   │   ├── utils.py           # Utility functions
│   │   └── processors.py     # Image processors
│   ├── api.py                 # FastAPI application
│   ├── mcp_server.py          # MCP server implementation
│   └── __init__.py
├── scripts/
│   └── start_mcp_server.py    # MCP server launcher
├── requirements.txt           # Python dependencies
└── README.md
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/
```

## Docker Deployment

### Build Image
```bash
docker build -t doculogix-ocr .
```

### Run Container
```bash
# REST API server
docker run -p 8000:8000 doculogix-ocr

# MCP server
docker run -p 8000:8000 doculogix-ocr python scripts/start_mcp_server.py --host 0.0.0.0
```

## Performance

### Benchmarks
- **PP-OCRv5**: 86.38% average accuracy, 8.46ms GPU inference
- **PP-StructureV3**: Advanced table recognition with layout analysis
- **Multi-language**: Supports 13+ languages with high accuracy

### Optimization Tips
- Use GPU acceleration for faster processing
- Enable high-performance mode for production
- Batch process multiple documents for efficiency
- Use appropriate model sizes (mobile vs server)

## Integration Examples

### With LangChain
```python
from langchain.document_loaders import BaseLoader
from src.ocr import PaddleOCREngine

class DocuLogixLoader(BaseLoader):
    def __init__(self, ocr_config=None):
        self.ocr = PaddleOCREngine(**(ocr_config or {}))
    
    def load(self, file_path):
        result = self.ocr.process_image(file_path)
        return [{"text": result.get_full_text(), "metadata": result.metadata}]
```

### With Claude/ChatGPT
```python
# Use MCP server for AI integration
# The MCP server provides tools that AI assistants can call
# to perform OCR operations on documents
```

## Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# If PaddlePaddle installation fails
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple/

# For GPU issues
pip install paddlepaddle-gpu==3.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

**Memory Issues:**
- Reduce batch size for large documents
- Use mobile models instead of server models
- Process images at lower resolution

**Accuracy Issues:**
- Ensure correct language setting
- Enable document orientation correction
- Use higher resolution images
- Enable structure analysis for complex documents

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Advanced OCR toolkit
- [PaddlePaddle](https://www.paddlepaddle.org.cn/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - AI integration standard

## Support

- 📧 Email: support@doculogix.com
- 💬 Issues: [GitHub Issues](https://github.com/zbest1000/DocuLogix/issues)
- 📖 Documentation: [Wiki](https://github.com/zbest1000/DocuLogix/wiki)
- 🎥 Tutorials: [YouTube Channel](https://youtube.com/doculogix)

---

**DocuLogix** - Transforming engineering document management with AI-powered OCR
