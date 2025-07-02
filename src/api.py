"""
FastAPI Web API for PaddleOCR Integration

This module provides REST API endpoints for OCR functionality,
allowing web applications to access PaddleOCR capabilities.
"""

import logging
import asyncio
import base64
import io
import tempfile
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

from PIL import Image
import numpy as np

from .ocr.engine import PaddleOCREngine
from .ocr.models import OCRResult


# Pydantic models for API requests/responses
class OCRRequest(BaseModel):
    """Request model for OCR processing"""
    image_data: str = Field(..., description="Base64 encoded image data")
    language: Optional[str] = Field("ch", description="OCR language")
    extract_tables: bool = Field(True, description="Extract table structures")
    extract_layout: bool = Field(True, description="Perform layout analysis")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")


class PDFOCRRequest(BaseModel):
    """Request model for PDF OCR processing"""
    pdf_data: str = Field(..., description="Base64 encoded PDF data")
    start_page: int = Field(0, ge=0, description="Starting page number (0-indexed)")
    end_page: Optional[int] = Field(None, description="Ending page number")
    language: Optional[str] = Field("ch", description="OCR language")
    extract_tables: bool = Field(True, description="Extract table structures")
    extract_layout: bool = Field(True, description="Perform layout analysis")


class LanguageRequest(BaseModel):
    """Request model for language change"""
    language: str = Field(..., description="New OCR language")


class OCRResponse(BaseModel):
    """Response model for OCR results"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    ocr_engine_status: str
    supported_languages: List[str]


def create_app(ocr_config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Create and configure FastAPI application
    
    Args:
        ocr_config: Configuration for PaddleOCR engine
        
    Returns:
        Configured FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is not installed. Please install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="DocuLogix PaddleOCR API",
        description="REST API for PaddleOCR integration with document processing capabilities",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize OCR engine
    ocr_config = ocr_config or {}
    ocr_engine = PaddleOCREngine(**ocr_config)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    def get_ocr_engine():
        """Dependency injection for OCR engine"""
        return ocr_engine
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(engine: PaddleOCREngine = Depends(get_ocr_engine)):
        """Health check endpoint"""
        try:
            return HealthResponse(
                status="healthy",
                version="1.0.0",
                ocr_engine_status="ready",
                supported_languages=engine.get_supported_languages()
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Service unhealthy")
    
    @app.post("/ocr/image", response_model=OCRResponse)
    async def process_image_ocr(
        request: OCRRequest,
        engine: PaddleOCREngine = Depends(get_ocr_engine)
    ):
        """
        Process an image for OCR
        
        Accepts base64 encoded image data and returns extracted text and structure.
        """
        try:
            # Change language if specified
            if request.language and request.language != engine.lang:
                engine.set_language(request.language)
            
            # Decode image
            image = _decode_base64_image(request.image_data)
            
            # Process image
            result = engine.process_image(
                image,
                extract_tables=request.extract_tables,
                extract_layout=request.extract_layout,
                confidence_threshold=request.confidence_threshold
            )
            
            # Format response
            response_data = _format_ocr_result(result)
            
            return OCRResponse(
                success=True,
                message="OCR processing completed successfully",
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return OCRResponse(
                success=False,
                message="OCR processing failed",
                error=str(e)
            )
    
    @app.post("/ocr/file")
    async def process_file_upload(
        file: UploadFile = File(...),
        language: str = "ch",
        extract_tables: bool = True,
        extract_layout: bool = True,
        confidence_threshold: float = 0.5,
        engine: PaddleOCREngine = Depends(get_ocr_engine)
    ):
        """
        Process uploaded image file for OCR
        
        Accepts multipart file upload and returns extracted text and structure.
        """
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Change language if specified
            if language and language != engine.lang:
                engine.set_language(language)
            
            # Read file data
            file_data = await file.read()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(file_data))
            image_array = np.array(image)
            
            # Process image
            result = engine.process_image(
                image_array,
                extract_tables=extract_tables,
                extract_layout=extract_layout,
                confidence_threshold=confidence_threshold
            )
            
            # Format response
            response_data = _format_ocr_result(result)
            
            return OCRResponse(
                success=True,
                message="File OCR processing completed successfully",
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"File OCR failed: {e}")
            return OCRResponse(
                success=False,
                message="File OCR processing failed",
                error=str(e)
            )
    
    @app.post("/ocr/pdf", response_model=OCRResponse)
    async def process_pdf_ocr(
        request: PDFOCRRequest,
        background_tasks: BackgroundTasks,
        engine: PaddleOCREngine = Depends(get_ocr_engine)
    ):
        """
        Process a PDF file for OCR
        
        Accepts base64 encoded PDF data and returns extracted text from all pages.
        """
        try:
            # Change language if specified
            if request.language and request.language != engine.lang:
                engine.set_language(request.language)
            
            # Decode PDF
            pdf_bytes = base64.b64decode(request.pdf_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Process PDF
                results = engine.process_pdf(
                    tmp_path,
                    start_page=request.start_page,
                    end_page=request.end_page
                )
                
                # Format results
                response_data = {
                    "total_pages": len(results),
                    "pages": [_format_ocr_result(result) for result in results]
                }
                
                # Schedule cleanup
                background_tasks.add_task(_cleanup_temp_file, tmp_path)
                
                return OCRResponse(
                    success=True,
                    message="PDF OCR processing completed successfully",
                    data=response_data
                )
                
            except Exception:
                # Immediate cleanup on error
                _cleanup_temp_file(tmp_path)
                raise
            
        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return OCRResponse(
                success=False,
                message="PDF OCR processing failed",
                error=str(e)
            )
    
    @app.post("/ocr/language", response_model=OCRResponse)
    async def set_ocr_language(
        request: LanguageRequest,
        engine: PaddleOCREngine = Depends(get_ocr_engine)
    ):
        """
        Change the OCR language
        
        Updates the OCR engine to use a different language model.
        """
        try:
            engine.set_language(request.language)
            
            return OCRResponse(
                success=True,
                message=f"OCR language changed to: {request.language}",
                data={
                    "current_language": engine.lang,
                    "supported_languages": engine.get_supported_languages()
                }
            )
            
        except Exception as e:
            logger.error(f"Language change failed: {e}")
            return OCRResponse(
                success=False,
                message="Language change failed",
                error=str(e)
            )
    
    @app.get("/ocr/languages")
    async def get_supported_languages(engine: PaddleOCREngine = Depends(get_ocr_engine)):
        """Get list of supported OCR languages"""
        return {
            "supported_languages": engine.get_supported_languages(),
            "current_language": engine.lang
        }
    
    @app.get("/ocr/models")
    async def get_model_info(engine: PaddleOCREngine = Depends(get_ocr_engine)):
        """Get information about available OCR models"""
        return {
            "detection_models": [
                "PP-OCRv5_server_det",
                "PP-OCRv5_mobile_det",
                "PP-OCRv4_server_det", 
                "PP-OCRv4_mobile_det"
            ],
            "recognition_models": [
                "PP-OCRv5_server_rec",
                "PP-OCRv5_mobile_rec",
                "PP-OCRv4_server_rec",
                "PP-OCRv4_mobile_rec"
            ],
            "structure_models": [
                "PP-StructureV3"
            ],
            "current_config": {
                "language": engine.lang,
                "gpu_enabled": engine.use_gpu,
                "structure_enabled": engine.enable_structure
            }
        }
    
    return app


def _decode_base64_image(base64_data: str) -> np.ndarray:
    """Decode base64 image data"""
    # Remove data URL prefix if present
    if base64_data.startswith("data:image/"):
        base64_data = base64_data.split(",")[1]
    
    # Decode base64
    image_bytes = base64.b64decode(base64_data)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to numpy array
    return np.array(image)


def _format_ocr_result(result: OCRResult) -> Dict[str, Any]:
    """Format OCR result for API response"""
    return {
        "text_regions": [
            {
                "text": region.text,
                "confidence": region.confidence,
                "bbox": region.bbox,
                "page_number": region.page_number,
                "language": region.language,
                "text_type": region.text_type
            }
            for region in result.text_regions
        ],
        "full_text": result.get_full_text(),
        "high_confidence_text": result.get_high_confidence_text(),
        "tables": [
            {
                "cells": len(table.cells),
                "rows": table.num_rows,
                "cols": table.num_cols,
                "confidence": table.confidence,
                "page_number": table.page_number,
                "table_type": table.table_type,
                "data": table.to_dict(),
                "csv_rows": table.to_csv_rows()
            }
            for table in result.tables
        ],
        "layout_info": result.layout_info,
        "summary": {
            "total_text_regions": len(result.text_regions),
            "total_tables": len(result.tables),
            "average_confidence": result.get_average_confidence(),
            "processing_time": result.processing_time,
            "full_text_length": len(result.get_full_text())
        },
        "metadata": result.metadata,
        "timestamp": result.timestamp.isoformat()
    }


def _cleanup_temp_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logging.warning(f"Failed to cleanup temp file {file_path}: {e}")


# Entry point for running the API server
if __name__ == "__main__":
    import uvicorn
    
    # Create app with default config
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )