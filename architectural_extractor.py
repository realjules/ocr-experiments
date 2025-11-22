#!/usr/bin/env python3
"""
Architectural Dimension Extractor for Government Tax Assessment

Extracts dimensions, scale, and room information from floor plans
Outputs structured JSON for downstream tax calculation tools
"""

import subprocess
import json
import sys
import re
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image


class ArchitecturalExtractor:
    """Extract architectural dimensions and measurements from floor plans"""

    # Ultra-simple prompt to avoid Ollama batching issues
    EXTRACTION_PROMPT = """<image>
Extract: scale, width, length, unit, all dimensions as JSON."""

    # Detailed prompt (use with --prompt flag if simple one works)
    EXTRACTION_PROMPT_DETAILED = """<image>

Extract from this floor plan as JSON:
1. Scale (e.g., "1:65")
2. Overall building width and length (largest dimensions)
3. Unit of measurement (cm, mm, or m) - check labels, legend, or infer from dimension values
4. All visible dimension numbers

Output JSON:
{
  "scale": "1:65",
  "width": 840,
  "length": 670,
  "unit": "cm",
  "unit_confidence": "high",
  "all_dimensions": [840, 670, 370, 510, 253, 240, 150, 90, 20]
}

unit_confidence: "high" (explicit), "medium" (inferred), "low" (guessed)"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.check_ollama()

    def log(self, message, level="INFO"):
        if self.verbose:
            print(f"[{level}] {message}")

    def check_ollama(self):
        """Verify Ollama and deepseek-ocr are available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            if "deepseek-ocr" not in result.stdout:
                print("ERROR: deepseek-ocr model not found")
                print("Run: ollama pull deepseek-ocr")
                sys.exit(1)
        except FileNotFoundError:
            print("ERROR: Ollama not installed")
            print("Install from: https://ollama.com")
            sys.exit(1)

    def extract_from_image(self, image_path, custom_prompt=None):
        """Extract dimensions from a floor plan image"""
        image_path = Path(image_path).absolute()  # Use absolute path
        self.log(f"Processing: {image_path}")

        prompt = custom_prompt or self.EXTRACTION_PROMPT

        # Run OCR
        self.log("Running DeepSeek-OCR (this may take 2-5 minutes)...")
        self.log(f"Command: ollama run deepseek-ocr [prompt] {image_path}")

        try:
            result = subprocess.run(
                ["ollama", "run", "deepseek-ocr", prompt, str(image_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',  # Force UTF-8
                errors='replace',  # Replace invalid characters instead of failing
                timeout=600,  # 10 minutes
            )
        except subprocess.TimeoutExpired:
            self.log("ERROR: Command timed out after 10 minutes", "ERROR")
            return {
                'success': False,
                'data': None,
                'raw_output': '',
                'error': 'Timeout'
            }

        # Debug output
        self.log(f"Return code: {result.returncode}")
        self.log(f"Stdout length: {len(result.stdout or '')} characters")
        self.log(f"Stderr length: {len(result.stderr or '')} characters")

        # Save debug files
        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""

        Path("debug_stdout.txt").write_text(stdout_text, encoding='utf-8')
        Path("debug_stderr.txt").write_text(stderr_text, encoding='utf-8')

        if stderr_text:
            self.log(f"STDERR preview: {stderr_text[:200]}", "WARNING")
            # Show full stderr if it's not too long
            if len(stderr_text) < 2000:
                self.log(f"Full STDERR:\n{stderr_text}", "DEBUG")

        if result.returncode != 0:
            self.log(f"Command failed with return code {result.returncode}", "ERROR")
            self.log(f"Check debug_stderr.txt for full error details", "ERROR")
            # Don't return yet - maybe stdout has useful info despite the error
            # Continue to process stdout if available

        raw_output = result.stdout.strip()

        if not raw_output:
            self.log("Empty stdout from OCR", "WARNING")
            if result.returncode != 0:
                return {
                    'success': False,
                    'data': None,
                    'raw_output': '',
                    'error': stderr_text or 'Unknown error'
                }
            # If return code is 0 but empty output, that's weird but continue
            self.log("Return code was 0 but output is empty - this is unusual", "WARNING")

        self.log(f"Raw output preview (first 200 chars): {raw_output[:200]}")

        # Extract JSON from output
        extracted_json = self.extract_json(raw_output)

        if extracted_json:
            self.log("✓ Successfully extracted JSON")
        else:
            self.log("✗ Failed to extract valid JSON from output", "WARNING")

        return {
            'success': extracted_json is not None,
            'data': extracted_json,
            'raw_output': raw_output
        }

    def extract_json(self, text):
        """Extract and validate JSON from OCR output"""
        # Try to find JSON block (between ```json and ``` or just { })
        patterns = [
            r'```json\s*(\{.*?\})\s*```',  # Code block
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Nested JSON
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    # Validate structure
                    if self.validate_structure(data):
                        return data
                except json.JSONDecodeError:
                    continue

        # If no valid JSON found, try to parse the entire text
        try:
            data = json.loads(text)
            if self.validate_structure(data):
                return data
        except json.JSONDecodeError:
            pass

        return None

    def validate_structure(self, data):
        """Validate that JSON has required fields"""
        required_fields = ['width', 'length', 'unit']
        return all(field in data for field in required_fields)

    def extract_from_pdf(self, pdf_path, custom_prompt=None, dpi=300):
        """Extract from PDF floor plan"""
        pdf_path = Path(pdf_path)
        self.log(f"Converting PDF: {pdf_path.name}")

        # Convert first page only (floor plans are usually single page)
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)

        if not images:
            return {'success': False, 'error': 'Failed to convert PDF'}

        self.log(f"Converted to image")

        # Save temporary image
        temp_path = Path(".temp_floorplan.png")
        images[0].save(temp_path, "PNG")

        # Extract
        result = self.extract_from_image(temp_path, custom_prompt)

        # Cleanup
        temp_path.unlink()

        return result

    def save_results(self, result, output_path):
        """Save extraction results"""
        output_path = Path(output_path)

        if result['success']:
            # Save structured JSON
            json_path = output_path.with_suffix('.json')
            json_path.write_text(json.dumps(result['data'], indent=2), encoding='utf-8')
            self.log(f"✓ Saved JSON: {json_path}")

            # Save raw output for debugging
            raw_path = output_path.with_suffix('.raw.txt')
            raw_path.write_text(result['raw_output'], encoding='utf-8')
            self.log(f"✓ Saved raw output: {raw_path}")

            # Print summary
            self.print_summary(result['data'])

            return json_path
        else:
            self.log("✗ Extraction failed - no valid JSON found", "ERROR")
            # Save raw output for debugging
            raw_path = output_path.with_suffix('.raw.txt')
            raw_path.write_text(result['raw_output'], encoding='utf-8')
            self.log(f"Raw output saved to: {raw_path}")

            # Save error info if available
            if 'error' in result and result['error']:
                error_path = output_path.with_suffix('.error.txt')
                error_path.write_text(result['error'], encoding='utf-8')
                self.log(f"Error details saved to: {error_path}")

            return None

    def print_summary(self, data):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"Scale: {data.get('scale', 'N/A')}")

        unit = data.get('unit', 'unknown')
        width = data.get('width', 'N/A')
        length = data.get('length', 'N/A')

        print(f"Width: {width} {unit}")
        print(f"Length: {length} {unit}")
        print(f"Unit confidence: {data.get('unit_confidence', 'N/A')}")

        # Calculate area if both dimensions present
        if 'width' in data and 'length' in data:
            # Convert to m² based on unit
            if unit == 'cm':
                area_m2 = (data['width'] * data['length']) / 10000
            elif unit == 'mm':
                area_m2 = (data['width'] * data['length']) / 1000000
            elif unit == 'm':
                area_m2 = data['width'] * data['length']
            else:
                area_m2 = None

            if area_m2:
                print(f"Calculated Area: {area_m2:.2f} m²")

        print(f"\nAll dimensions found: {data.get('all_dimensions', [])}")
        print("="*60 + "\n")

    def process(self, input_path, output_path=None, custom_prompt=None, dpi=300):
        """Main processing function"""
        input_path = Path(input_path)

        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            sys.exit(1)

        # Set default output path
        if output_path is None:
            output_path = input_path.stem + "_dimensions.json"

        # Process based on file type
        if input_path.suffix.lower() == '.pdf':
            result = self.extract_from_pdf(input_path, custom_prompt, dpi)
        else:
            result = self.extract_from_image(input_path, custom_prompt)

        # Save results
        saved_path = self.save_results(result, output_path)

        return result, saved_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract architectural dimensions for tax assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from PDF floor plan
  python architectural_extractor.py floorplan.pdf

  # Extract from image with custom output
  python architectural_extractor.py floorplan.jpg -o dimensions.json

  # High resolution extraction
  python architectural_extractor.py floorplan.pdf --dpi 600

  # Custom extraction prompt
  python architectural_extractor.py floorplan.pdf --prompt "Extract dimensions..."

Output JSON format:
  {
    "scale": "1:65",
    "width": 840,
    "length": 670,
    "unit": "cm",
    "unit_confidence": "high",
    "all_dimensions": [840, 670, 370, 510, ...]
  }

This JSON can then be passed to Claude with tax_calculator tool for area calculations.
        """
    )

    parser.add_argument('input', help="Floor plan image or PDF")
    parser.add_argument('-o', '--output', help="Output JSON path (default: input_name_dimensions.json)")
    parser.add_argument('--prompt', help="Custom extraction prompt")
    parser.add_argument('--dpi', type=int, default=300, help="DPI for PDF conversion (default: 300)")
    parser.add_argument('-q', '--quiet', action='store_true', help="Suppress verbose output")

    args = parser.parse_args()

    # Create extractor
    extractor = ArchitecturalExtractor(verbose=not args.quiet)

    # Process
    result, saved_path = extractor.process(
        args.input,
        args.output,
        args.prompt,
        args.dpi
    )

    # Exit code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
