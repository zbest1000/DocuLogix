"""
OCR module for DocuLogix using PaddleOCR 3.x

This module provides comprehensive OCR capabilities including:
- Document text recognition
- Table structure recognition  
- Layout analysis
- Multi-language support
"""

from .engine import PaddleOCREngine
from .models import OCRResult, TextRegion, TableResult
from .processors import DocumentProcessor, ImagePreprocessor
from .utils import format_results, save_results

__all__ = [
    "PaddleOCREngine",
    "OCRResult",
    "TextRegion", 
    "TableResult",
    "DocumentProcessor",
    "ImagePreprocessor",
    "format_results",
    "save_results"
]