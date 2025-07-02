"""
DocuLogix - Engineering document/submittal tracker with PaddleOCR integration

This package provides comprehensive OCR capabilities for engineering documents
using PaddleOCR 3.x with MCP server support.
"""

__version__ = "1.0.0"
__author__ = "DocuLogix Team"
__description__ = "Engineering document tracker with advanced OCR capabilities"

from .ocr import PaddleOCREngine, OCRResult
from .api import create_app
from .mcp_server import MCPOCRServer

__all__ = [
    "PaddleOCREngine",
    "OCRResult", 
    "create_app",
    "MCPOCRServer"
]