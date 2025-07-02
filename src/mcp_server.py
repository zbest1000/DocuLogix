"""
MCP Server for PaddleOCR Integration

This module implements the Model Context Protocol (MCP) server for DocuLogix,
providing OCR capabilities through a standardized interface.

Based on PaddleOCR 3.x MCP server implementation:
https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html
"""

import asyncio
import json
import logging
import base64
import io
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import tempfile
import os

try:
    from mcp.server import Server
    from mcp.types import (
        Resource, Tool, TextContent, ImageContent, EmbeddedResource,
        CallToolResult, ListResourcesResult, ListToolsResult, ReadResourceResult
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP not available. Install with: pip install mcp")

from PIL import Image
import numpy as np

from .ocr.engine import PaddleOCREngine
from .ocr.models import OCRResult


class MCPOCRServer:
    """
    MCP Server implementation for PaddleOCR
    
    Provides OCR capabilities through the Model Context Protocol,
    enabling AI assistants to perform document analysis.
    """
    
    def __init__(
        self,
        name: str = "paddleocr-server",
        version: str = "1.0.0",
        ocr_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP OCR Server
        
        Args:
            name: Server name
            version: Server version
            ocr_config: Configuration for PaddleOCR engine
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP is not installed. Please install with: pip install mcp")
        
        self.name = name
        self.version = version
        self.server = Server(name)
        
        # Initialize OCR engine
        ocr_config = ocr_config or {}
        self.ocr_engine = PaddleOCREngine(**ocr_config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register MCP handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register MCP protocol handlers"""
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available OCR resources"""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="ocr://supported-languages",
                        name="Supported OCR Languages",
                        description="List of languages supported by PaddleOCR",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="ocr://models",
                        name="Available OCR Models", 
                        description="Information about available OCR models",
                        mimeType="application/json"
                    )
                ]
            )
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read OCR resource information"""
            if uri == "ocr://supported-languages":
                languages = self.ocr_engine.get_supported_languages()
                content = json.dumps({
                    "supported_languages": languages,
                    "current_language": self.ocr_engine.lang
                }, indent=2)
                
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=content
                        )
                    ]
                )
            
            elif uri == "ocr://models":
                model_info = {
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
                        "language": self.ocr_engine.lang,
                        "gpu_enabled": self.ocr_engine.use_gpu,
                        "structure_enabled": self.ocr_engine.enable_structure
                    }
                }
                
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text", 
                            text=json.dumps(model_info, indent=2)
                        )
                    ]
                )
            
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available OCR tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="ocr_image",
                        description="Perform OCR on an image file or base64 encoded image",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image": {
                                    "type": "string",
                                    "description": "Base64 encoded image or file path"
                                },
                                "language": {
                                    "type": "string", 
                                    "description": "OCR language (optional)",
                                    "enum": self.ocr_engine.get_supported_languages()
                                },
                                "extract_tables": {
                                    "type": "boolean",
                                    "description": "Extract table structures",
                                    "default": True
                                },
                                "extract_layout": {
                                    "type": "boolean", 
                                    "description": "Perform layout analysis",
                                    "default": True
                                },
                                "confidence_threshold": {
                                    "type": "number",
                                    "description": "Minimum confidence threshold",
                                    "default": 0.5,
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                }
                            },
                            "required": ["image"]
                        }
                    ),
                    Tool(
                        name="ocr_pdf",
                        description="Perform OCR on a PDF file",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "pdf_data": {
                                    "type": "string",
                                    "description": "Base64 encoded PDF file"
                                },
                                "start_page": {
                                    "type": "integer",
                                    "description": "Starting page number (0-indexed)",
                                    "default": 0
                                },
                                "end_page": {
                                    "type": "integer",
                                    "description": "Ending page number (optional)"
                                },
                                "language": {
                                    "type": "string",
                                    "description": "OCR language (optional)",
                                    "enum": self.ocr_engine.get_supported_languages()
                                }
                            },
                            "required": ["pdf_data"]
                        }
                    ),
                    Tool(
                        name="set_ocr_language",
                        description="Change the OCR language",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "language": {
                                    "type": "string",
                                    "description": "New OCR language",
                                    "enum": self.ocr_engine.get_supported_languages()
                                }
                            },
                            "required": ["language"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls"""
            try:
                if name == "ocr_image":
                    return await self._handle_ocr_image(arguments)
                elif name == "ocr_pdf":
                    return await self._handle_ocr_pdf(arguments)
                elif name == "set_ocr_language":
                    return await self._handle_set_language(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                self.logger.error(f"Tool call failed: {e}")
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error: {str(e)}"
                        )
                    ],
                    isError=True
                )
    
    async def _handle_ocr_image(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle OCR image processing"""
        image_data = arguments["image"]
        language = arguments.get("language")
        extract_tables = arguments.get("extract_tables", True)
        extract_layout = arguments.get("extract_layout", True)
        confidence_threshold = arguments.get("confidence_threshold", 0.5)
        
        # Change language if specified
        if language and language != self.ocr_engine.lang:
            self.ocr_engine.set_language(language)
        
        # Process image
        if image_data.startswith("data:image/") or len(image_data) > 500:
            # Base64 encoded image
            image = self._decode_base64_image(image_data)
        else:
            # File path
            image = image_data
        
        # Perform OCR
        result = self.ocr_engine.process_image(
            image,
            extract_tables=extract_tables,
            extract_layout=extract_layout,
            confidence_threshold=confidence_threshold
        )
        
        # Format results
        response = self._format_ocr_result(result)
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, ensure_ascii=False)
                )
            ]
        )
    
    async def _handle_ocr_pdf(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle PDF OCR processing"""
        pdf_data = arguments["pdf_data"]
        start_page = arguments.get("start_page", 0)
        end_page = arguments.get("end_page")
        language = arguments.get("language")
        
        # Change language if specified
        if language and language != self.ocr_engine.lang:
            self.ocr_engine.set_language(language)
        
        # Decode PDF
        pdf_bytes = base64.b64decode(pdf_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Process PDF
            results = self.ocr_engine.process_pdf(
                tmp_path,
                start_page=start_page,
                end_page=end_page
            )
            
            # Format results
            response = {
                "total_pages": len(results),
                "pages": [self._format_ocr_result(result) for result in results]
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, ensure_ascii=False)
                )
            ]
        )
    
    async def _handle_set_language(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle language change"""
        language = arguments["language"]
        
        try:
            self.ocr_engine.set_language(language)
            response = {
                "status": "success",
                "message": f"OCR language changed to: {language}",
                "current_language": self.ocr_engine.lang
            }
        except Exception as e:
            response = {
                "status": "error", 
                "message": f"Failed to change language: {str(e)}",
                "current_language": self.ocr_engine.lang
            }
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(response, indent=2)
                )
            ]
        )
    
    def _decode_base64_image(self, base64_data: str) -> np.ndarray:
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
    
    def _format_ocr_result(self, result: OCRResult) -> Dict[str, Any]:
        """Format OCR result for MCP response"""
        return {
            "text_regions": [
                {
                    "text": region.text,
                    "confidence": region.confidence,
                    "bbox": region.bbox,
                    "page_number": region.page_number
                }
                for region in result.text_regions
            ],
            "full_text": result.get_full_text(),
            "tables": [
                {
                    "cells": len(table.cells),
                    "rows": table.num_rows,
                    "cols": table.num_cols,
                    "confidence": table.confidence,
                    "data": table.to_dict()
                }
                for table in result.tables
            ],
            "layout_info": result.layout_info,
            "summary": {
                "total_text_regions": len(result.text_regions),
                "total_tables": len(result.tables), 
                "average_confidence": result.get_average_confidence(),
                "processing_time": result.processing_time
            },
            "metadata": result.metadata
        }
    
    async def run(self, transport_type: str = "sse", host: str = "localhost", port: int = 8000):
        """Run the MCP server"""
        if transport_type == "sse":
            # Server-Sent Events transport
            from mcp.server.sse import SseServerTransport
            
            async with SseServerTransport(f"http://{host}:{port}/sse") as transport:
                await self.server.run(transport)
        
        elif transport_type == "stdio":
            # Standard I/O transport
            from mcp.server.stdio import StdioServerTransport
            
            async with StdioServerTransport() as transport:
                await self.server.run(transport)
        
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")


async def main():
    """Main entry point for MCP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PaddleOCR MCP Server")
    parser.add_argument("--transport", choices=["sse", "stdio"], default="sse",
                       help="Transport type")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--lang", default="ch", help="OCR language")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run server
    ocr_config = {
        "lang": args.lang,
        "use_gpu": args.gpu
    }
    
    server = MCPOCRServer(ocr_config=ocr_config)
    await server.run(
        transport_type=args.transport,
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":
    asyncio.run(main())