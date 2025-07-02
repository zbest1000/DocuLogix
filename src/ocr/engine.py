"""
PaddleOCR Engine for DocuLogix

This module provides the main OCR engine using PaddleOCR 3.x with support for:
- PP-OCRv5 for text recognition
- PP-StructureV3 for document structure analysis
- PP-ChatOCRv4 for intelligent document understanding
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

try:
    from paddleocr import PaddleOCR, PPStructureV3
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. Install with: pip install paddleocr")

from .models import OCRResult, TextRegion, TableResult


class PaddleOCREngine:
    """
    Main OCR engine using PaddleOCR 3.x for document processing
    
    Features:
    - Multi-language text recognition
    - Table structure recognition
    - Document layout analysis
    - High-performance inference
    """
    
    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = True,
        use_doc_orientation: bool = True,
        use_doc_unwarping: bool = True,
        use_textline_orientation: bool = True,
        ocr_version: str = "PP-OCRv5",
        det_model: str = "PP-OCRv5_server_det",
        rec_model: str = "PP-OCRv5_server_rec",
        enable_structure: bool = True,
        **kwargs
    ):
        """
        Initialize PaddleOCR Engine
        
        Args:
            lang: Language for OCR ('ch', 'en', 'korean', etc.)
            use_gpu: Whether to use GPU acceleration
            use_doc_orientation: Enable document orientation classification
            use_doc_unwarping: Enable document unwarping
            use_textline_orientation: Enable text line orientation classification
            ocr_version: PP-OCR version ('PP-OCRv5', 'PP-OCRv4', 'PP-OCRv3')
            det_model: Text detection model name
            rec_model: Text recognition model name
            enable_structure: Enable structure analysis (tables, layout)
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not installed. Please install with: pip install paddleocr")
        
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_structure = enable_structure
        
        # Device configuration
        device = "gpu" if use_gpu else "cpu"
        
        # Initialize main OCR engine
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=use_doc_orientation,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            ocr_version=ocr_version,
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            device=device,
            **kwargs
        )
        
        # Initialize structure analysis if enabled
        if enable_structure:
            try:
                self.structure = PPStructureV3(
                    use_doc_orientation_classify=use_doc_orientation,
                    use_doc_unwarping=use_doc_unwarping,
                    device=device
                )
            except Exception as e:
                logging.warning(f"Failed to initialize PP-StructureV3: {e}")
                self.structure = None
        else:
            self.structure = None
            
        self.logger = logging.getLogger(__name__)
    
    def process_image(
        self, 
        image_path: Union[str, Path, np.ndarray, Image.Image],
        extract_tables: bool = True,
        extract_layout: bool = True,
        confidence_threshold: float = 0.5
    ) -> OCRResult:
        """
        Process an image for OCR and structure analysis
        
        Args:
            image_path: Path to image file, numpy array, or PIL Image
            extract_tables: Whether to extract table structures
            extract_layout: Whether to perform layout analysis
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            OCRResult object containing all extracted information
        """
        try:
            # Handle different input types
            if isinstance(image_path, (str, Path)):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = str(image_path)
            elif isinstance(image_path, np.ndarray):
                image = image_path
            elif isinstance(image_path, Image.Image):
                image = np.array(image_path)
            else:
                raise ValueError("Unsupported image input type")
            
            # Perform OCR
            ocr_results = self.ocr.predict(image)
            
            # Extract text regions
            text_regions = []
            for page_result in ocr_results:
                if hasattr(page_result, 'res') and 'rec_texts' in page_result.res:
                    texts = page_result.res['rec_texts']
                    scores = page_result.res['rec_scores']
                    boxes = page_result.res['rec_boxes']
                    
                    for text, score, box in zip(texts, scores, boxes):
                        if score >= confidence_threshold:
                            text_regions.append(TextRegion(
                                text=text,
                                confidence=float(score),
                                bbox=box.tolist(),
                                page_number=0
                            ))
            
            # Perform structure analysis if enabled
            tables = []
            layout_info = {}
            
            if self.structure and (extract_tables or extract_layout):
                try:
                    structure_results = self.structure.predict(image)
                    
                    for page_result in structure_results:
                        if hasattr(page_result, 'res'):
                            # Extract layout information
                            if 'layout_det_res' in page_result.res:
                                layout_info = self._parse_layout_results(page_result.res['layout_det_res'])
                            
                            # Extract table information
                            if extract_tables and 'overall_ocr_res' in page_result.res:
                                tables.extend(self._extract_tables(page_result.res))
                                
                except Exception as e:
                    self.logger.warning(f"Structure analysis failed: {e}")
            
            return OCRResult(
                text_regions=text_regions,
                tables=tables,
                layout_info=layout_info,
                confidence_scores=[region.confidence for region in text_regions],
                processing_time=0.0,  # TODO: Add timing
                metadata={
                    'lang': self.lang,
                    'ocr_version': 'PP-OCRv5',
                    'structure_enabled': self.structure is not None
                }
            )
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        start_page: int = 0,
        end_page: Optional[int] = None,
        dpi: int = 200
    ) -> List[OCRResult]:
        """
        Process a PDF file for OCR
        
        Args:
            pdf_path: Path to PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (None for all pages)
            dpi: DPI for PDF to image conversion
            
        Returns:
            List of OCRResult objects, one per page
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
        
        results = []
        doc = fitz.open(pdf_path)
        
        end_page = end_page or doc.page_count
        
        for page_num in range(start_page, min(end_page, doc.page_count)):
            page = doc[page_num]
            
            # Convert page to image
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the page
            result = self.process_image(image)
            result.metadata['page_number'] = page_num
            results.append(result)
        
        doc.close()
        return results
    
    def _parse_layout_results(self, layout_res: Dict[str, Any]) -> Dict[str, Any]:
        """Parse layout detection results"""
        layout_info = {
            'regions': [],
            'page_structure': {}
        }
        
        if 'boxes' in layout_res:
            for box_info in layout_res['boxes']:
                region = {
                    'type': box_info.get('label', 'unknown'),
                    'confidence': box_info.get('score', 0.0),
                    'bbox': box_info.get('coordinate', []),
                    'class_id': box_info.get('cls_id', -1)
                }
                layout_info['regions'].append(region)
        
        return layout_info
    
    def _extract_tables(self, structure_res: Dict[str, Any]) -> List[TableResult]:
        """Extract table information from structure results"""
        tables = []
        
        # This would need to be implemented based on PP-StructureV3 output format
        # For now, return empty list
        return tables
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [
            'ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka',
            'latin', 'arabic', 'cyrillic', 'devanagari', 'eslav'
        ]
    
    def set_language(self, lang: str):
        """Change OCR language"""
        if lang not in self.get_supported_languages():
            raise ValueError(f"Unsupported language: {lang}")
        
        self.lang = lang
        # Reinitialize OCR with new language
        device = "gpu" if self.use_gpu else "cpu"
        self.ocr = PaddleOCR(lang=lang, device=device)