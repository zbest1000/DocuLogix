"""
Utility functions for OCR processing

This module provides helper functions for result formatting,
file handling, and common OCR operations.
"""

import json
import csv
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from .models import OCRResult, TextRegion, TableResult


def format_results(result: OCRResult, format_type: str = "json") -> str:
    """
    Format OCR results in different output formats
    
    Args:
        result: OCR result object
        format_type: Output format ('json', 'text', 'markdown', 'csv')
        
    Returns:
        Formatted string
    """
    if format_type.lower() == "json":
        return result.to_json()
    
    elif format_type.lower() == "text":
        return result.get_full_text()
    
    elif format_type.lower() == "markdown":
        return _format_markdown(result)
    
    elif format_type.lower() == "csv":
        return _format_csv(result)
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def save_results(
    result: OCRResult, 
    output_path: Union[str, Path],
    format_type: str = "json",
    include_images: bool = False
) -> None:
    """
    Save OCR results to file
    
    Args:
        result: OCR result object
        output_path: Output file path
        format_type: Output format
        include_images: Whether to save visualization images
    """
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    formatted_content = format_results(result, format_type)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    # Save additional formats
    base_path = output_path.with_suffix('')
    
    # Always save JSON for machine readability
    if format_type != "json":
        json_path = base_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(result.to_json())
    
    # Save plain text for human readability
    if format_type != "text":
        txt_path = base_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.get_full_text())
    
    # Save tables as CSV if present
    if result.tables:
        for i, table in enumerate(result.tables):
            csv_path = base_path.with_suffix(f'_table_{i}.csv')
            _save_table_csv(table, csv_path)
    
    logging.info(f"Results saved to {output_path}")


def merge_ocr_results(results: List[OCRResult]) -> OCRResult:
    """
    Merge multiple OCR results into a single result
    
    Args:
        results: List of OCR results to merge
        
    Returns:
        Combined OCR result
    """
    if not results:
        return OCRResult()
    
    if len(results) == 1:
        return results[0]
    
    # Combine all text regions
    all_regions = []
    page_offset = 0
    
    for result in results:
        for region in result.text_regions:
            # Update page numbers
            new_region = TextRegion(
                text=region.text,
                confidence=region.confidence,
                bbox=region.bbox,
                page_number=region.page_number + page_offset,
                language=region.language,
                text_type=region.text_type,
                rotation_angle=region.rotation_angle
            )
            all_regions.append(new_region)
        page_offset += 1
    
    # Combine tables
    all_tables = []
    page_offset = 0
    
    for result in results:
        for table in result.tables:
            # Update page numbers for tables
            table.page_number += page_offset
            all_tables.append(table)
        page_offset += 1
    
    # Combine metadata
    combined_metadata = {
        "source_documents": len(results),
        "total_pages": sum(len(r.text_regions) for r in results),
        "processing_details": [r.metadata for r in results]
    }
    
    # Add original metadata from first result
    if results[0].metadata:
        combined_metadata.update(results[0].metadata)
    
    return OCRResult(
        text_regions=all_regions,
        tables=all_tables,
        layout_info={},  # Layout info is not merged
        confidence_scores=[r.confidence for r in all_regions],
        processing_time=sum(r.processing_time for r in results),
        metadata=combined_metadata
    )


def filter_by_keywords(
    result: OCRResult, 
    keywords: List[str],
    case_sensitive: bool = False
) -> OCRResult:
    """
    Filter OCR results to only include text regions containing keywords
    
    Args:
        result: OCR result object
        keywords: List of keywords to search for
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        Filtered OCR result
    """
    filtered_regions = []
    
    for region in result.text_regions:
        text = region.text if case_sensitive else region.text.lower()
        search_keywords = keywords if case_sensitive else [k.lower() for k in keywords]
        
        if any(keyword in text for keyword in search_keywords):
            filtered_regions.append(region)
    
    return OCRResult(
        text_regions=filtered_regions,
        tables=result.tables,  # Keep all tables
        layout_info=result.layout_info,
        confidence_scores=[r.confidence for r in filtered_regions],
        processing_time=result.processing_time,
        metadata={
            **result.metadata,
            "filtered_by_keywords": keywords,
            "original_region_count": len(result.text_regions),
            "filtered_region_count": len(filtered_regions)
        }
    )


def extract_structured_data(result: OCRResult) -> Dict[str, Any]:
    """
    Extract structured data from OCR results
    
    Args:
        result: OCR result object
        
    Returns:
        Dictionary with structured data
    """
    structured_data = {
        "document_text": result.get_full_text(),
        "high_confidence_text": result.get_high_confidence_text(),
        "text_regions": len(result.text_regions),
        "tables": len(result.tables),
        "average_confidence": result.get_average_confidence()
    }
    
    # Extract key-value pairs (simple heuristic)
    key_value_pairs = _extract_key_value_pairs(result.text_regions)
    if key_value_pairs:
        structured_data["key_value_pairs"] = key_value_pairs
    
    # Extract table data
    if result.tables:
        structured_data["table_data"] = [
            {
                "table_id": i,
                "rows": table.num_rows,
                "cols": table.num_cols,
                "data": table.to_dict(),
                "csv_data": table.to_csv_rows()
            }
            for i, table in enumerate(result.tables)
        ]
    
    # Extract layout information
    if result.layout_info:
        structured_data["layout"] = result.layout_info
    
    return structured_data


def validate_ocr_result(result: OCRResult) -> List[str]:
    """
    Validate OCR result for common issues
    
    Args:
        result: OCR result object
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    # Check for empty results
    if not result.text_regions:
        issues.append("No text regions detected")
    
    # Check confidence scores
    low_confidence_count = sum(
        1 for region in result.text_regions 
        if region.confidence < 0.5
    )
    
    if low_confidence_count > 0:
        issues.append(f"{low_confidence_count} text regions have low confidence (<0.5)")
    
    # Check for very short text regions (possible noise)
    short_text_count = sum(
        1 for region in result.text_regions 
        if len(region.text.strip()) < 2
    )
    
    if short_text_count > len(result.text_regions) * 0.3:
        issues.append("High number of very short text regions detected (possible noise)")
    
    # Check bounding boxes
    invalid_bbox_count = sum(
        1 for region in result.text_regions 
        if len(region.bbox) < 4
    )
    
    if invalid_bbox_count > 0:
        issues.append(f"{invalid_bbox_count} text regions have invalid bounding boxes")
    
    return issues


def _format_markdown(result: OCRResult) -> str:
    """Format OCR result as Markdown"""
    markdown = f"# OCR Results\n\n"
    markdown += f"**Processing Time:** {result.processing_time:.2f}s\n"
    markdown += f"**Average Confidence:** {result.get_average_confidence():.2f}\n"
    markdown += f"**Text Regions:** {len(result.text_regions)}\n"
    markdown += f"**Tables:** {len(result.tables)}\n\n"
    
    # Add extracted text
    markdown += "## Extracted Text\n\n"
    markdown += result.get_full_text()
    markdown += "\n\n"
    
    # Add tables
    if result.tables:
        markdown += "## Tables\n\n"
        for i, table in enumerate(result.tables):
            markdown += f"### Table {i+1}\n\n"
            markdown += _table_to_markdown(table)
            markdown += "\n\n"
    
    # Add metadata
    if result.metadata:
        markdown += "## Metadata\n\n"
        markdown += f"```json\n{json.dumps(result.metadata, indent=2)}\n```\n"
    
    return markdown


def _format_csv(result: OCRResult) -> str:
    """Format OCR result as CSV"""
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Text', 'Confidence', 'Page', 'BBox_X1', 'BBox_Y1', 'BBox_X2', 'BBox_Y2'])
    
    # Write text regions
    for region in result.text_regions:
        bbox = region.bbox + [0] * (4 - len(region.bbox))  # Pad to 4 elements
        writer.writerow([
            region.text,
            region.confidence,
            region.page_number,
            bbox[0], bbox[1], bbox[2], bbox[3]
        ])
    
    return output.getvalue()


def _save_table_csv(table: TableResult, filepath: Path) -> None:
    """Save table as CSV file"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        csv_rows = table.to_csv_rows()
        writer.writerows(csv_rows)


def _table_to_markdown(table: TableResult) -> str:
    """Convert table to Markdown format"""
    if not table.cells:
        return "*Empty table*"
    
    csv_rows = table.to_csv_rows()
    if not csv_rows:
        return "*No table data*"
    
    markdown = ""
    
    # Header row
    if csv_rows:
        header = csv_rows[0]
        markdown += "| " + " | ".join(header) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
        
        # Data rows
        for row in csv_rows[1:]:
            markdown += "| " + " | ".join(row) + " |\n"
    
    return markdown


def _extract_key_value_pairs(text_regions: List[TextRegion]) -> Dict[str, str]:
    """Extract key-value pairs from text regions using simple heuristics"""
    key_value_pairs = {}
    
    for region in text_regions:
        text = region.text.strip()
        
        # Look for patterns like "Key: Value" or "Key = Value"
        if ':' in text:
            parts = text.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    key_value_pairs[key] = value
        elif '=' in text:
            parts = text.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    key_value_pairs[key] = value
    
    return key_value_pairs