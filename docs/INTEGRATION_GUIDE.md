# PaddleOCR Integration Guide

This guide provides comprehensive instructions for integrating PaddleOCR 3.x into DocuLogix, including setup, configuration, and usage examples.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Configuration](#basic-configuration)
4. [OCR Engine Usage](#ocr-engine-usage)
5. [MCP Server Setup](#mcp-server-setup)
6. [REST API Integration](#rest-api-integration)
7. [Advanced Features](#advanced-features)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## Overview

DocuLogix integrates PaddleOCR 3.x to provide state-of-the-art OCR capabilities including:

- **PP-OCRv5**: Latest text recognition with 13% accuracy improvement
- **PP-StructureV3**: Advanced document structure analysis
- **PP-ChatOCRv4**: AI-powered document understanding
- **Multi-language Support**: 13+ languages with high accuracy
- **MCP Server**: Model Context Protocol for AI assistant integration

## Installation

### System Requirements

- Python 3.8+ (3.9+ recommended)
- 4GB+ RAM (8GB+ for GPU)
- Optional: NVIDIA GPU with CUDA 11.8+ or 12.6+

### Step 1: Install Dependencies

```bash
# Clone repository
git clone https://github.com/zbest1000/DocuLogix.git
cd DocuLogix

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### Step 2: Install PaddlePaddle

#### CPU Version
```bash
pip install paddlepaddle>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

#### GPU Version (CUDA 11.8)
```bash
pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

#### GPU Version (CUDA 12.6)  
```bash
pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

### Step 3: Verify Installation

```python
import paddle
print(f"PaddlePaddle version: {paddle.__version__}")
print(f"GPU available: {paddle.is_compiled_with_cuda()}")

from paddleocr import PaddleOCR
print("PaddleOCR imported successfully")
```

## Basic Configuration

### OCR Engine Configuration

```python
from src.ocr import PaddleOCREngine

# Basic configuration
ocr_config = {
    "lang": "ch",  # Language: ch, en, korean, japan, etc.
    "use_gpu": True,  # Enable GPU acceleration
    "use_doc_orientation": True,  # Auto-rotate documents
    "use_doc_unwarping": True,  # Correct document distortion
    "use_textline_orientation": True,  # Fix text orientation
    "ocr_version": "PP-OCRv5",  # PP-OCRv5, PP-OCRv4, PP-OCRv3
    "det_model": "PP-OCRv5_server_det",  # Detection model
    "rec_model": "PP-OCRv5_server_rec",  # Recognition model
    "enable_structure": True  # Enable table/layout analysis
}

# Initialize engine
engine = PaddleOCREngine(**ocr_config)
```

### Language Support

| Language | Code | Description |
|----------|------|-------------|
| Chinese (Simplified) | `ch` | Default Chinese model |
| English | `en` | English text recognition |
| Korean | `korean` | Korean text support |
| Japanese | `japan` | Japanese character recognition |
| Chinese (Traditional) | `chinese_cht` | Traditional Chinese |
| Arabic | `arabic` | Arabic script |
| Latin | `latin` | Latin-based languages |
| Cyrillic | `cyrillic` | Cyrillic script |
| Devanagari | `devanagari` | Hindi and related languages |
| Tamil | `ta` | Tamil script |
| Telugu | `te` | Telugu script |
| Kannada | `ka` | Kannada script |
| East Slavic | `eslav` | Russian, Ukrainian, etc. |

## OCR Engine Usage

### Basic Image Processing

```python
from src.ocr import PaddleOCREngine

# Initialize engine
ocr = PaddleOCREngine(lang="en", use_gpu=True)

# Process single image
result = ocr.process_image("document.jpg")

# Get extracted text
print("Full text:")
print(result.get_full_text())

# Get high-confidence text only
print("\nHigh confidence text:")
print(result.get_high_confidence_text(threshold=0.8))

# Access individual text regions
for region in result.text_regions:
    print(f"Text: {region.text}")
    print(f"Confidence: {region.confidence:.2f}")
    print(f"Bounding box: {region.bbox}")
    print("---")
```

### PDF Processing

```python
# Process entire PDF
pdf_results = ocr.process_pdf("document.pdf")

# Process specific pages
pdf_results = ocr.process_pdf(
    "document.pdf",
    start_page=0,  # First page (0-indexed)
    end_page=5,    # Process first 5 pages
    dpi=200        # Image resolution
)

# Combine results from all pages
from src.ocr.utils import merge_ocr_results
combined_result = merge_ocr_results(pdf_results)
```

### Table Extraction

```python
# Process document with table analysis
result = ocr.process_image(
    "table_document.jpg",
    extract_tables=True,
    extract_layout=True
)

# Access table data
for i, table in enumerate(result.tables):
    print(f"Table {i+1}:")
    print(f"Rows: {table.num_rows}, Cols: {table.num_cols}")
    
    # Get table as dictionary
    table_dict = table.to_dict()
    print(table_dict)
    
    # Get table as CSV rows
    csv_rows = table.to_csv_rows()
    for row in csv_rows:
        print(row)
```

### Batch Processing

```python
import os
from pathlib import Path

# Process multiple images
image_folder = Path("documents/")
results = []

for image_file in image_folder.glob("*.jpg"):
    try:
        result = ocr.process_image(str(image_file))
        results.append(result)
        print(f"Processed: {image_file.name}")
    except Exception as e:
        print(f"Error processing {image_file.name}: {e}")

# Merge all results
combined = merge_ocr_results(results)
```

## MCP Server Setup

The Model Context Protocol (MCP) server enables AI assistants to perform OCR operations.

### Starting the MCP Server

```bash
# Basic SSE server
python scripts/start_mcp_server.py

# With custom configuration
python scripts/start_mcp_server.py \
  --transport sse \
  --host 0.0.0.0 \
  --port 8000 \
  --lang en \
  --gpu \
  --ocr-version PP-OCRv5

# STDIO transport for direct integration
python scripts/start_mcp_server.py --transport stdio
```

### MCP Server Configuration

```python
from src.mcp_server import MCPOCRServer

# Create server with custom config
server = MCPOCRServer(
    name="doculogix-ocr-server",
    version="1.0.0",
    ocr_config={
        "lang": "en",
        "use_gpu": True,
        "enable_structure": True
    }
)

# Run server
await server.run(transport_type="sse", host="localhost", port=8000)
```

### MCP Client Integration

For AI assistants, the MCP server provides these tools:

- `ocr_image`: Process base64 encoded images
- `ocr_pdf`: Process PDF documents  
- `set_ocr_language`: Change OCR language

Example client usage:
```json
{
  "method": "tools/call",
  "params": {
    "name": "ocr_image",
    "arguments": {
      "image": "data:image/jpeg;base64,/9j/4AAQ...",
      "language": "en",
      "extract_tables": true,
      "confidence_threshold": 0.7
    }
  }
}
```

## REST API Integration

### Starting the API Server

```python
from src.api import create_app
import uvicorn

# Create FastAPI app
app = create_app(ocr_config={
    "lang": "en",
    "use_gpu": True,
    "enable_structure": True
})

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

### API Endpoints

#### 1. Process Image (Base64)

```bash
curl -X POST "http://localhost:8000/ocr/image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "data:image/jpeg;base64,/9j/4AAQ...",
    "language": "en",
    "extract_tables": true,
    "confidence_threshold": 0.5
  }'
```

#### 2. Upload Image File

```bash
curl -X POST "http://localhost:8000/ocr/file" \
  -F "file=@document.jpg" \
  -F "language=en" \
  -F "extract_tables=true"
```

#### 3. Process PDF

```bash
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_data": "JVBERi0xLjQK...",
    "start_page": 0,
    "end_page": 5,
    "language": "en"
  }'
```

#### 4. Change Language

```bash
curl -X POST "http://localhost:8000/ocr/language" \
  -H "Content-Type: application/json" \
  -d '{"language": "ch"}'
```

#### 5. Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

### Python Client Example

```python
import requests
import base64

# Read and encode image
with open("document.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post(
    "http://localhost:8000/ocr/image",
    json={
        "image_data": f"data:image/jpeg;base64,{image_data}",
        "language": "en",
        "extract_tables": True,
        "confidence_threshold": 0.7
    }
)

result = response.json()
if result["success"]:
    print("Extracted text:", result["data"]["full_text"])
    print("Tables found:", len(result["data"]["tables"]))
else:
    print("Error:", result["error"])
```

## Advanced Features

### Custom Result Processing

```python
from src.ocr.utils import (
    filter_by_keywords,
    extract_structured_data,
    validate_ocr_result
)

# Filter results by keywords
keywords = ["invoice", "total", "amount", "date"]
filtered_result = filter_by_keywords(result, keywords, case_sensitive=False)

# Extract structured data
structured = extract_structured_data(result)
print("Key-value pairs:", structured["key_value_pairs"])
print("Table data:", structured["table_data"])

# Validate results
issues = validate_ocr_result(result)
if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"- {issue}")
```

### Result Export Formats

```python
from src.ocr.utils import format_results, save_results

# Format in different formats
json_output = format_results(result, "json")
text_output = format_results(result, "text")
markdown_output = format_results(result, "markdown")
csv_output = format_results(result, "csv")

# Save results with multiple formats
save_results(
    result,
    output_path="output/document_results.json",
    format_type="json",
    include_images=True
)
```

### Language Switching

```python
# Change language during runtime
ocr.set_language("ch")  # Switch to Chinese
result_ch = ocr.process_image("chinese_document.jpg")

ocr.set_language("en")  # Switch back to English
result_en = ocr.process_image("english_document.jpg")
```

## Performance Optimization

### GPU Acceleration

```python
# Enable GPU with optimized settings
ocr_config = {
    "use_gpu": True,
    "enable_hpi": True,  # High-performance inference
    "use_tensorrt": True,  # TensorRT acceleration
    "precision": "fp16"  # Half precision for speed
}

ocr = PaddleOCREngine(**ocr_config)
```

### Model Selection

```python
# Server models (higher accuracy, slower)
server_config = {
    "det_model": "PP-OCRv5_server_det",
    "rec_model": "PP-OCRv5_server_rec"
}

# Mobile models (faster, smaller)
mobile_config = {
    "det_model": "PP-OCRv5_mobile_det", 
    "rec_model": "PP-OCRv5_mobile_rec"
}
```

### Batch Processing Optimization

```python
# Process multiple images efficiently
def batch_process_images(image_paths, batch_size=8):
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        
        # Process batch
        batch_results = []
        for image_path in batch:
            result = ocr.process_image(image_path)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Optional: Clear GPU cache
        import gc
        gc.collect()
    
    return results
```

### Memory Management

```python
# Configure memory limits
import os
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.8"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

# For large documents, process in chunks
def process_large_pdf(pdf_path, pages_per_chunk=10):
    import fitz
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    
    all_results = []
    
    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)
        
        chunk_results = ocr.process_pdf(
            pdf_path,
            start_page=start_page,
            end_page=end_page
        )
        
        all_results.extend(chunk_results)
        
        # Clear memory
        import gc
        gc.collect()
    
    return all_results
```

## Troubleshooting

### Common Installation Issues

#### 1. PaddlePaddle Installation Fails

```bash
# Try Chinese mirror
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Or use conda
conda install paddlepaddle-gpu -c paddle

# For specific CUDA version
pip install paddlepaddle-gpu==3.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

#### 2. CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Install compatible PaddlePaddle version
# CUDA 11.8
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.6
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

#### 3. Import Errors

```python
# Test imports individually
try:
    import paddle
    print("✓ PaddlePaddle imported")
except ImportError as e:
    print(f"✗ PaddlePaddle import failed: {e}")

try:
    from paddleocr import PaddleOCR
    print("✓ PaddleOCR imported")
except ImportError as e:
    print(f"✗ PaddleOCR import failed: {e}")
```

### Runtime Issues

#### 1. Out of Memory

```python
# Reduce image resolution
result = ocr.process_image(
    "large_image.jpg",
    text_det_limit_side_len=736,  # Reduce from default 960
    text_det_limit_type="min"
)

# Use mobile models
ocr = PaddleOCREngine(
    det_model="PP-OCRv5_mobile_det",
    rec_model="PP-OCRv5_mobile_rec"
)
```

#### 2. Low Accuracy

```python
# Enable all preprocessing
ocr = PaddleOCREngine(
    use_doc_orientation=True,    # Fix document rotation
    use_doc_unwarping=True,      # Correct distortion
    use_textline_orientation=True # Fix text orientation
)

# Adjust detection thresholds
result = ocr.process_image(
    "document.jpg",
    text_det_thresh=0.3,      # Lower threshold for more detection
    text_det_box_thresh=0.5,  # Lower box threshold
    confidence_threshold=0.3   # Lower confidence threshold
)
```

#### 3. Slow Performance

```bash
# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Use high-performance mode
python -c "
from src.ocr import PaddleOCREngine
ocr = PaddleOCREngine(
    use_gpu=True,
    enable_hpi=True,
    use_tensorrt=True
)
"
```

### Debugging

#### Enable Detailed Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set PaddleOCR logging
logging.getLogger('ppocr').setLevel(logging.DEBUG)
```

#### Performance Profiling

```python
import time

# Measure processing time
start_time = time.time()
result = ocr.process_image("document.jpg")
processing_time = time.time() - start_time

print(f"Processing time: {processing_time:.2f}s")
print(f"Text regions: {len(result.text_regions)}")
print(f"Average confidence: {result.get_average_confidence():.2f}")
```

#### Memory Usage Monitoring

```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Monitor memory during processing
monitor_memory()
result = ocr.process_image("large_document.jpg")
monitor_memory()
```

## Support and Resources

### Documentation
- [PaddleOCR Official Docs](https://paddlepaddle.github.io/PaddleOCR/)
- [PaddlePaddle Documentation](https://www.paddlepaddle.org.cn/documentation)
- [Model Context Protocol](https://modelcontextprotocol.io/)

### Community
- [PaddleOCR GitHub Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)
- [PaddlePaddle Community](https://www.paddlepaddle.org.cn/community)

### Performance Benchmarks
- PP-OCRv5: 86.38% average accuracy, 8.46ms GPU inference
- PP-StructureV3: Advanced table recognition with layout analysis
- Supports 13+ languages with high accuracy

For additional support, please refer to the main [README.md](../README.md) or open an issue on GitHub.