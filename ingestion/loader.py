"""Document loader using Docling for PDF/DOCX/HTML → markdown conversion."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_file(file_path: str, use_gpu: bool = False) -> str:
    """Load document from file using Docling.

    Supports: PDF, DOCX, PPTX, XLSX, HTML, TXT
    Returns markdown text representation.

    For TXT files, uses simple read (no Docling needed).
    For other formats, uses Docling DocumentConverter.

    Args:
        file_path: Path to the document file.
        use_gpu: If True, use GPU acceleration for Docling PDF pipeline.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # TXT files - simple read (no Docling needed)
    if path.suffix.lower() == ".txt":
        logger.info("Loading TXT file: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # Other formats - use Docling (lazy import)
    from docling.document_converter import DocumentConverter

    converter_kwargs: dict = {}

    if use_gpu and path.suffix.lower() == ".pdf":
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            accel = AcceleratorOptions(device=AcceleratorDevice.AUTO)
            pdf_opts = PdfPipelineOptions(accelerator_options=accel)
            converter_kwargs["format_options"] = {
                "pdf": PdfFormatOption(pipeline_options=pdf_opts),
            }
            logger.info("GPU acceleration enabled for PDF")
        except ImportError:
            logger.warning("GPU acceleration imports failed, falling back to CPU")

    logger.info("Loading document via Docling: %s", file_path)
    converter = DocumentConverter(**converter_kwargs)
    result = converter.convert(file_path)

    # Export to markdown
    markdown_text = result.document.export_to_markdown()
    logger.info("Loaded %d characters from %s", len(markdown_text), file_path)

    return markdown_text


def load_text(text: str) -> str:
    """Load raw text as-is (pass-through).

    Used when text is already extracted/provided directly.
    """
    return text
