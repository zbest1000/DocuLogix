#!/usr/bin/env python3
"""
Startup script for PaddleOCR MCP Server

This script provides an easy way to start the MCP server with different configurations.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.mcp_server import MCPOCRServer
    import asyncio
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mcp_server.log')
        ]
    )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PaddleOCR MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start SSE server on default port
  python scripts/start_mcp_server.py
  
  # Start with STDIO transport
  python scripts/start_mcp_server.py --transport stdio
  
  # Start with GPU acceleration and English language
  python scripts/start_mcp_server.py --gpu --lang en
  
  # Start on custom host and port
  python scripts/start_mcp_server.py --host 0.0.0.0 --port 9000
        """
    )
    
    # Transport options
    parser.add_argument(
        "--transport", 
        choices=["sse", "stdio"], 
        default="sse",
        help="Transport type (default: sse)"
    )
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host address for SSE transport (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port number for SSE transport (default: 8000)"
    )
    
    # OCR configuration
    parser.add_argument(
        "--lang", 
        default="ch",
        choices=[
            'ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka',
            'latin', 'arabic', 'cyrillic', 'devanagari', 'eslav'
        ],
        help="OCR language (default: ch)"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Use GPU acceleration"
    )
    parser.add_argument(
        "--ocr-version",
        default="PP-OCRv5",
        choices=["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"],
        help="PaddleOCR version (default: PP-OCRv5)"
    )
    parser.add_argument(
        "--det-model",
        default="PP-OCRv5_server_det",
        help="Text detection model (default: PP-OCRv5_server_det)"
    )
    parser.add_argument(
        "--rec-model", 
        default="PP-OCRv5_server_rec",
        help="Text recognition model (default: PP-OCRv5_server_rec)"
    )
    parser.add_argument(
        "--disable-structure",
        action="store_true",
        help="Disable structure analysis (tables, layout)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Build OCR configuration
    ocr_config = {
        "lang": args.lang,
        "use_gpu": args.gpu,
        "ocr_version": args.ocr_version,
        "det_model": args.det_model,
        "rec_model": args.rec_model,
        "enable_structure": not args.disable_structure
    }
    
    logger.info(f"Starting PaddleOCR MCP Server with config: {ocr_config}")
    logger.info(f"Transport: {args.transport}")
    
    if args.transport == "sse":
        logger.info(f"Server will be available at: http://{args.host}:{args.port}/sse")
    
    try:
        # Create and start server
        server = MCPOCRServer(
            name="paddleocr-mcp-server",
            version="1.0.0",
            ocr_config=ocr_config
        )
        
        await server.run(
            transport_type=args.transport,
            host=args.host,
            port=args.port
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)