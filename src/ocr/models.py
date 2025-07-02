"""
Data models for OCR results

This module defines the data structures used to represent OCR results,
including text regions, tables, and overall document analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TextRegion:
    """Represents a detected text region in a document"""
    text: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] or polygon points
    page_number: int = 0
    language: Optional[str] = None
    text_type: str = "general"  # general, handwritten, printed, etc.
    rotation_angle: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if len(self.bbox) < 4:
            raise ValueError("Bounding box must have at least 4 coordinates")


@dataclass 
class TableCell:
    """Represents a cell in a detected table"""
    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    confidence: float = 0.0
    bbox: List[float] = field(default_factory=list)


@dataclass
class TableResult:
    """Represents a detected table structure"""
    cells: List[TableCell]
    num_rows: int
    num_cols: int
    bbox: List[float]
    confidence: float
    page_number: int = 0
    table_type: str = "standard"  # standard, financial, form, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary format"""
        table_dict = {}
        for cell in self.cells:
            if cell.row not in table_dict:
                table_dict[cell.row] = {}
            table_dict[cell.row][cell.col] = {
                'text': cell.text,
                'rowspan': cell.rowspan,
                'colspan': cell.colspan,
                'confidence': cell.confidence
            }
        return table_dict
    
    def to_csv_rows(self) -> List[List[str]]:
        """Convert table to CSV-compatible row format"""
        rows = []
        for row_idx in range(self.num_rows):
            row = [''] * self.num_cols
            for cell in self.cells:
                if cell.row == row_idx:
                    row[cell.col] = cell.text
            rows.append(row)
        return rows


@dataclass
class LayoutRegion:
    """Represents a layout region (paragraph, title, figure, etc.)"""
    region_type: str  # paragraph, title, figure, table, etc.
    bbox: List[float]
    confidence: float
    text_content: Optional[str] = None
    region_id: Optional[str] = None


@dataclass
class OCRResult:
    """
    Complete OCR result for a document or page
    
    Contains all extracted information including text regions, tables,
    layout analysis, and metadata.
    """
    text_regions: List[TextRegion] = field(default_factory=list)
    tables: List[TableResult] = field(default_factory=list)
    layout_info: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: List[float] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_full_text(self, separator: str = "\n") -> str:
        """Get all recognized text concatenated"""
        return separator.join([region.text for region in self.text_regions])
    
    def get_high_confidence_text(self, threshold: float = 0.8) -> str:
        """Get text from regions with confidence above threshold"""
        high_conf_regions = [
            region for region in self.text_regions 
            if region.confidence >= threshold
        ]
        return "\n".join([region.text for region in high_conf_regions])
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence across all text regions"""
        if not self.text_regions:
            return 0.0
        return sum(region.confidence for region in self.text_regions) / len(self.text_regions)
    
    def filter_by_confidence(self, threshold: float = 0.5) -> "OCRResult":
        """Return a new OCRResult with only high-confidence regions"""
        filtered_regions = [
            region for region in self.text_regions 
            if region.confidence >= threshold
        ]
        
        return OCRResult(
            text_regions=filtered_regions,
            tables=self.tables,
            layout_info=self.layout_info,
            confidence_scores=[r.confidence for r in filtered_regions],
            processing_time=self.processing_time,
            metadata=self.metadata,
            timestamp=self.timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OCR result to dictionary"""
        return {
            'text_regions': [
                {
                    'text': region.text,
                    'confidence': region.confidence,
                    'bbox': region.bbox,
                    'page_number': region.page_number,
                    'language': region.language,
                    'text_type': region.text_type,
                    'rotation_angle': region.rotation_angle
                }
                for region in self.text_regions
            ],
            'tables': [
                {
                    'cells': [
                        {
                            'text': cell.text,
                            'row': cell.row,
                            'col': cell.col,
                            'rowspan': cell.rowspan,
                            'colspan': cell.colspan,
                            'confidence': cell.confidence,
                            'bbox': cell.bbox
                        }
                        for cell in table.cells
                    ],
                    'num_rows': table.num_rows,
                    'num_cols': table.num_cols,
                    'bbox': table.bbox,
                    'confidence': table.confidence,
                    'page_number': table.page_number,
                    'table_type': table.table_type
                }
                for table in self.tables
            ],
            'layout_info': self.layout_info,
            'confidence_scores': self.confidence_scores,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'summary': {
                'total_text_regions': len(self.text_regions),
                'total_tables': len(self.tables),
                'average_confidence': self.get_average_confidence(),
                'full_text_length': len(self.get_full_text())
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert OCR result to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_to_file(self, filepath: str, format: str = "json"):
        """Save OCR result to file"""
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        elif format.lower() == "txt":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.get_full_text())
        else:
            raise ValueError(f"Unsupported format: {format}")


@dataclass
class BatchOCRResult:
    """Results from processing multiple documents"""
    results: List[OCRResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    def get_combined_text(self) -> str:
        """Get all text from all documents combined"""
        all_text = []
        for result in self.results:
            all_text.append(result.get_full_text())
        return "\n\n--- NEW DOCUMENT ---\n\n".join(all_text)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the batch"""
        total_regions = sum(len(result.text_regions) for result in self.results)
        total_tables = sum(len(result.tables) for result in self.results)
        avg_confidence = sum(result.get_average_confidence() for result in self.results) / len(self.results) if self.results else 0
        
        return {
            'total_documents': len(self.results),
            'total_text_regions': total_regions,
            'total_tables': total_tables,
            'average_confidence': avg_confidence,
            'total_processing_time': self.total_processing_time,
            'success_rate': self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0
        }