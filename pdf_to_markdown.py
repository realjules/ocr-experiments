#!/usr/bin/env python3
"""
PDF to Markdown converter using DeepSeek-OCR via Ollama

Converts each page of a PDF to an image, processes with deepseek-ocr,
and combines the results into a single markdown file.

Requirements:
    pip install pdf2image pillow

System requirements:
    - poppler-utils (Linux: apt install poppler-utils, Mac: brew install poppler, Windows: see README)
    - Ollama with deepseek-ocr model installed
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image


def check_ollama():
    """Check if Ollama is installed and deepseek-ocr model is available"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        if "deepseek-ocr" not in result.stdout:
            print("Error: deepseek-ocr model not found in Ollama")
            print("Please run: ollama pull deepseek-ocr")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: Ollama is not installed")
        print("Please install Ollama from: https://ollama.com")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: Could not run Ollama")
        sys.exit(1)


def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF pages to images

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (higher = better quality but slower)

    Returns:
        List of PIL Image objects
    """
    print(f"Converting PDF to images (DPI: {dpi})...")
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"✓ Converted {len(images)} pages")
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        print("\nMake sure poppler is installed:")
        print("  Linux: sudo apt install poppler-utils")
        print("  Mac: brew install poppler")
        print("  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
        sys.exit(1)


def ocr_image(image_path, prompt="<image>\nConvert the document to markdown."):
    """
    Process image through deepseek-ocr using Ollama

    Args:
        image_path: Path to image file
        prompt: Prompt for OCR processing

    Returns:
        Extracted text as string
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-ocr", prompt, str(image_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per page
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"[Error: Processing timeout for {image_path.name}]"
    except Exception as e:
        return f"[Error processing {image_path.name}: {e}]"


def process_pdf_to_markdown(
    pdf_path,
    output_path=None,
    dpi=300,
    prompt="<image>\nConvert the document to markdown.",
    page_separator="\n\n---\n\n"
):
    """
    Main function to convert PDF to markdown

    Args:
        pdf_path: Path to input PDF
        output_path: Path for output markdown file (defaults to input_name.md)
        dpi: Resolution for PDF to image conversion
        prompt: Custom prompt for OCR
        page_separator: Separator between pages in output
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Set default output path
    if output_path is None:
        output_path = pdf_path.with_suffix('.md')
    else:
        output_path = Path(output_path)

    print(f"Input PDF: {pdf_path}")
    print(f"Output Markdown: {output_path}\n")

    # Check dependencies
    check_ollama()

    # Create temporary directory for images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Convert PDF to images
        images = pdf_to_images(pdf_path, dpi=dpi)

        # Save images temporarily and process each page
        markdown_pages = []

        for i, image in enumerate(images, 1):
            # Save image
            image_path = temp_path / f"page_{i:03d}.png"
            image.save(image_path, "PNG")

            print(f"Processing page {i}/{len(images)}...")

            # OCR the image
            markdown_text = ocr_image(image_path, prompt)
            markdown_pages.append(markdown_text)

            print(f"✓ Page {i} processed ({len(markdown_text)} characters)")

        # Combine all pages
        print("\nCombining pages...")
        full_markdown = page_separator.join(markdown_pages)

        # Save to file
        output_path.write_text(full_markdown, encoding='utf-8')

        print(f"\n✓ Done! Markdown saved to: {output_path}")
        print(f"  Total pages: {len(images)}")
        print(f"  Total characters: {len(full_markdown)}")

# HELLOshA@55


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using DeepSeek-OCR via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pdf_to_markdown.py document.pdf

  # Custom output path
  python pdf_to_markdown.py document.pdf -o output.md

  # Higher quality (slower)
  python pdf_to_markdown.py document.pdf --dpi 600

  # Custom prompt
  python pdf_to_markdown.py document.pdf --prompt "Extract all text and preserve formatting"

  # No page separators
  python pdf_to_markdown.py document.pdf --no-separator
        """
    )

    parser.add_argument(
        "pdf_path",
        help="Path to input PDF file"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output markdown file path (default: same name as PDF with .md extension)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300, higher = better quality but slower)"
    )

    parser.add_argument(
        "--prompt",
        default="<image>\nConvert the document to markdown.",
        help="Custom prompt for OCR processing"
    )

    parser.add_argument(
        "--separator",
        default="\n\n---\n\n",
        help="Page separator in output (default: \\n\\n---\\n\\n)"
    )

    parser.add_argument(
        "--no-separator",
        action="store_true",
        help="Don't add separators between pages"
    )

    args = parser.parse_args()

    # Handle no-separator flag
    separator = "" if args.no_separator else args.separator

    process_pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_path=args.output,
        dpi=args.dpi,
        prompt=args.prompt,
        page_separator=separator
    )


if __name__ == "__main__":
    main()
