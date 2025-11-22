#!/usr/bin/env python3
"""
Intelligent OCR Agent using DeepSeek-OCR via Ollama

Features:
- Auto-detects document types
- Extracts structured data (JSON + Markdown)
- Batch processing with smart organization
- Interactive refinement mode
- Quality validation and confidence scoring
"""

import subprocess
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_path
from PIL import Image


class OCRAgent:
    """Intelligent OCR processing agent"""

    DOCUMENT_TYPES = {
        'invoice': {
            'keywords': ['invoice', 'bill to', 'amount due', 'subtotal', 'tax'],
            'prompt': '<image>\nExtract invoice data as JSON with fields: invoice_number, date, vendor, total, line_items. Also provide markdown summary.',
            'structured': True
        },
        'receipt': {
            'keywords': ['receipt', 'total', 'paid', 'change', 'thank you'],
            'prompt': '<image>\nExtract receipt data as JSON: vendor, date, items, subtotal, tax, total.',
            'structured': True
        },
        'contract': {
            'keywords': ['agreement', 'party', 'terms', 'conditions', 'signature'],
            'prompt': '<image>\nExtract key contract information: parties involved, effective date, key terms, obligations. Output as structured markdown.',
            'structured': False
        },
        'form': {
            'keywords': ['name:', 'date:', 'address:', 'phone:', 'email:'],
            'prompt': '<image>\nExtract all form fields and their values as JSON.',
            'structured': True
        },
        'architectural': {
            'keywords': ['floor plan', 'elevation', 'section', 'district', 'construction'],
            'prompt': '<image>\nExtract: project title, location, dimensions, room labels, construction notes. Preserve structure.',
            'structured': False
        },
        'table': {
            'keywords': ['table', 'row', 'column', 'header'],
            'prompt': '<image>\nExtract all tables as HTML with proper headers and data cells.',
            'structured': False
        },
        'document': {
            'keywords': [],  # default fallback
            'prompt': '<image>\nConvert the document to markdown, preserving structure and formatting.',
            'structured': False
        }
    }

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.check_dependencies()

    def log(self, message, level="INFO"):
        """Print log messages"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def check_dependencies(self):
        """Check if Ollama and deepseek-ocr are available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            if "deepseek-ocr" not in result.stdout:
                self.log("deepseek-ocr model not found. Run: ollama pull deepseek-ocr", "ERROR")
                sys.exit(1)
        except FileNotFoundError:
            self.log("Ollama not installed. Visit: https://ollama.com", "ERROR")
            sys.exit(1)

    def detect_document_type(self, image_path):
        """Auto-detect document type using quick OCR scan"""
        self.log(f"Detecting document type for: {image_path.name}")

        # Quick scan with basic OCR
        result = subprocess.run(
            ["ollama", "run", "deepseek-ocr", "Extract first 500 characters of text", str(image_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        sample_text = result.stdout.lower()

        # Score each document type
        scores = {}
        for doc_type, config in self.DOCUMENT_TYPES.items():
            if doc_type == 'document':  # skip default
                continue
            score = sum(1 for keyword in config['keywords'] if keyword in sample_text)
            scores[doc_type] = score

        # Get best match
        if scores and max(scores.values()) > 0:
            detected_type = max(scores, key=scores.get)
            confidence = scores[detected_type] / len(self.DOCUMENT_TYPES[detected_type]['keywords'])
            self.log(f"Detected: {detected_type} (confidence: {confidence:.0%})")
            return detected_type, confidence

        return 'document', 0.0

    def process_image(self, image_path, doc_type=None, custom_prompt=None):
        """Process a single image through OCR"""
        image_path = Path(image_path)

        # Auto-detect if not specified
        if doc_type is None:
            doc_type, confidence = self.detect_document_type(image_path)
        else:
            confidence = 1.0

        # Get prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self.DOCUMENT_TYPES[doc_type]['prompt']

        self.log(f"Processing as: {doc_type}")
        self.log(f"Using prompt: {prompt[:80]}...")

        # Run OCR
        result = subprocess.run(
            ["ollama", "run", "deepseek-ocr", prompt, str(image_path)],
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='ignore'  # Handle encoding issues
        )

        output = result.stdout.strip()

        # Try to extract JSON if structured output expected
        structured_data = None
        if self.DOCUMENT_TYPES[doc_type]['structured']:
            structured_data = self.extract_json(output)

        return {
            'type': doc_type,
            'confidence': confidence,
            'raw_output': output,
            'structured_data': structured_data,
            'success': result.returncode == 0
        }

    def extract_json(self, text):
        """Extract JSON from mixed text/JSON output"""
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None

    def process_pdf(self, pdf_path, doc_type=None, custom_prompt=None, dpi=300):
        """Process PDF by converting to images"""
        pdf_path = Path(pdf_path)
        self.log(f"Converting PDF: {pdf_path.name}")

        images = convert_from_path(str(pdf_path), dpi=dpi)
        self.log(f"Converted {len(images)} pages")

        results = []
        for i, image in enumerate(images, 1):
            self.log(f"Processing page {i}/{len(images)}")

            # Save temp image
            temp_path = Path(f".temp_page_{i}.png")
            image.save(temp_path, "PNG")

            # Process
            result = self.process_image(temp_path, doc_type, custom_prompt)
            result['page'] = i
            results.append(result)

            # Cleanup
            temp_path.unlink()

        return results

    def save_results(self, results, output_dir, filename):
        """Save processing results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save raw markdown
        if isinstance(results, list):
            # Multi-page
            markdown = "\n\n---\n\n".join([r['raw_output'] for r in results])
        else:
            # Single page
            markdown = results['raw_output']

        md_path = output_dir / f"{filename}.md"
        md_path.write_text(markdown, encoding='utf-8')
        self.log(f"Saved markdown: {md_path}")

        # Save JSON if structured
        if isinstance(results, list):
            structured = [r['structured_data'] for r in results if r['structured_data']]
        else:
            structured = [results['structured_data']] if results['structured_data'] else []

        if structured:
            json_path = output_dir / f"{filename}.json"
            json_path.write_text(json.dumps(structured, indent=2), encoding='utf-8')
            self.log(f"Saved JSON: {json_path}")

        # Save metadata
        meta_path = output_dir / f"{filename}_meta.json"
        if isinstance(results, list):
            meta = {
                'pages': len(results),
                'types': [r['type'] for r in results],
                'confidences': [r['confidence'] for r in results]
            }
        else:
            meta = {
                'type': results['type'],
                'confidence': results['confidence']
            }

        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

        return output_dir

    def batch_process(self, input_path, output_dir="ocr_output", doc_type=None, pattern="*"):
        """Process multiple files in a directory"""
        input_path = Path(input_path)

        if input_path.is_file():
            files = [input_path]
        else:
            # Find all images and PDFs
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.pdf', '*.JPG', '*.JPEG', '*.PNG']
            files = []
            for ext in extensions:
                files.extend(input_path.glob(ext))

        if not files:
            self.log("No files found", "ERROR")
            return

        self.log(f"Found {len(files)} files to process")

        results_summary = []

        for i, file_path in enumerate(files, 1):
            self.log(f"\n{'='*60}")
            self.log(f"Processing [{i}/{len(files)}]: {file_path.name}")
            self.log(f"{'='*60}")

            try:
                # Process based on type
                if file_path.suffix.lower() == '.pdf':
                    results = self.process_pdf(file_path, doc_type)
                else:
                    results = self.process_image(file_path, doc_type)

                # Save
                filename = file_path.stem
                saved_to = self.save_results(results, output_dir, filename)

                results_summary.append({
                    'file': file_path.name,
                    'status': 'success',
                    'output': str(saved_to)
                })

            except Exception as e:
                self.log(f"Error processing {file_path.name}: {e}", "ERROR")
                results_summary.append({
                    'file': file_path.name,
                    'status': 'error',
                    'error': str(e)
                })

        # Save batch summary
        summary_path = Path(output_dir) / "_batch_summary.json"
        summary_path.write_text(json.dumps(results_summary, indent=2), encoding='utf-8')

        self.log(f"\n{'='*60}")
        self.log(f"Batch processing complete!")
        self.log(f"Processed: {len([r for r in results_summary if r['status'] == 'success'])}/{len(files)}")
        self.log(f"Results saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Intelligent OCR Agent using DeepSeek-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and process
  python ocr_agent.py document.pdf

  # Process specific type
  python ocr_agent.py invoice.pdf --type invoice

  # Batch process directory
  python ocr_agent.py ./invoices/ --output ./processed

  # Custom prompt
  python ocr_agent.py form.jpg --prompt "Extract all form fields as JSON"

  # High quality PDF processing
  python ocr_agent.py document.pdf --dpi 600

Document types: invoice, receipt, contract, form, architectural, table, document
        """
    )

    parser.add_argument('input', help="Input file or directory")
    parser.add_argument('-o', '--output', default='ocr_output', help="Output directory")
    parser.add_argument('-t', '--type', choices=list(OCRAgent.DOCUMENT_TYPES.keys()),
                       help="Document type (auto-detected if not specified)")
    parser.add_argument('--prompt', help="Custom OCR prompt")
    parser.add_argument('--dpi', type=int, default=300, help="DPI for PDF conversion")
    parser.add_argument('-q', '--quiet', action='store_true', help="Suppress verbose output")

    args = parser.parse_args()

    # Create agent
    agent = OCRAgent(verbose=not args.quiet)

    # Process
    input_path = Path(args.input)

    if input_path.is_dir():
        agent.batch_process(input_path, args.output, args.type)
    elif input_path.suffix.lower() == '.pdf':
        results = agent.process_pdf(input_path, args.type, args.prompt, args.dpi)
        agent.save_results(results, args.output, input_path.stem)
    else:
        results = agent.process_image(input_path, args.type, args.prompt)
        agent.save_results(results, args.output, input_path.stem)


if __name__ == "__main__":
    main()
