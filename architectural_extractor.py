#!/usr/bin/env python3
"""
Architectural Dimension Extractor for Government Tax Assessment

Extracts dimensions, scale, and room information from floor plans.
Uses a three-tier extraction pipeline:
  1. LiteParse (instant, no model) - primary
  2. DeepSeek-OCR text-only fallback (seconds)
  3. DeepSeek-OCR image fallback (minutes, last resort)

Outputs structured JSON for downstream tax calculation tools.
"""

import subprocess
import json
import sys
import re
import csv
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

try:
    from liteparse import LiteParse
    HAS_LITEPARSE = True
except ImportError:
    HAS_LITEPARSE = False


class ArchitecturalExtractor:
    """Extract architectural dimensions and measurements from floor plans"""

    # Focused prompt for image-based OCR (last resort)
    EXTRACTION_PROMPT = "<image>\nWhat are the two overall building dimensions (width x length) on this floor plan? Reply ONLY: W=___ L=___ UNIT=___ SCALE=___"

    # Text-only prompt for DeepSeek fallback (fast)
    TEXT_FALLBACK_PROMPT = (
        "This is extracted text from an architectural floor plan. "
        "The candidate dimension numbers are: {candidates}. "
        "Which two numbers are the OVERALL BUILDING width and length "
        "(the outer envelope, not room dimensions)? "
        "Reply ONLY: W=___ L=___ UNIT=___"
    )

    def __init__(self, verbose=True, ollama_path=None):
        self.verbose = verbose
        self.ollama_path = ollama_path or "ollama"
        self._ollama_checked = False
        if HAS_LITEPARSE:
            self._liteparse = LiteParse()
        else:
            self._liteparse = None

    def log(self, message, level="INFO"):
        if self.verbose:
            print(f"[{level}] {message}")

    def check_ollama(self):
        """Verify Ollama and deepseek-ocr are available (lazy, checked once)"""
        if self._ollama_checked:
            return True
        try:
            result = subprocess.run(
                [self.ollama_path, "list"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            self._ollama_checked = True
            if "deepseek-ocr" not in result.stdout:
                self.log("deepseek-ocr model not found. Run: ollama pull deepseek-ocr", "WARNING")
                return False
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.log("Ollama not available", "WARNING")
            return False

    # ── Tier 1: LiteParse (instant, no model) ──

    def extract_with_liteparse(self, pdf_path):
        """Extract text from a born-digital PDF using LiteParse. No model needed."""
        if not self._liteparse:
            return None

        self.log("Tier 1: Trying LiteParse (no model)...")
        try:
            result = self._liteparse.parse(str(pdf_path))
            if not result or not hasattr(result, 'pages') or not result.pages:
                self.log("LiteParse returned no pages", "WARNING")
                return None

            text = ""
            for page in result.pages:
                text += page.text if hasattr(page, 'text') else str(page)

            text = text.strip()
            if len(text) < 20:
                self.log(f"LiteParse text too short ({len(text)} chars)", "WARNING")
                return None

            self.log(f"LiteParse extracted {len(text)} chars")
            return text
        except Exception as e:
            self.log(f"LiteParse failed: {e}", "WARNING")
            return None

    @staticmethod
    def filter_candidate_dimensions(text):
        """Filter extracted text to isolate dimension numbers from noise.

        Removes:
        - Phone numbers (+250..., 9+ digit sequences)
        - Dates (4-digit years 2000-2099)
        - Construction spec numbers ("150mm thick", "200mm deep")
        - UPI/reference numbers (slash-separated digit sequences)
        - Very small numbers (< 10) and very large (> 10000)
        """
        cleaned = text

        # Remove phone numbers: +250788815711, 0788244592, +250788449070
        cleaned = re.sub(r'\+?\d{9,}', ' ', cleaned)

        # Remove UPI/reference numbers like 1/03/05/03/10947
        cleaned = re.sub(r'\d+/\d+/\d+[\d/]*', ' ', cleaned)

        # Remove dates (4-digit years)
        cleaned = re.sub(r'\b(20[0-9]{2}|19[0-9]{2})\b', ' ', cleaned)

        # Remove construction spec numbers: "150mm thick", "200mm deep", "3-ply"
        cleaned = re.sub(r'\b(\d+)\s*mm\s+(thick|deep|wide|high)\b', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b\d+-ply\b', ' ', cleaned, flags=re.IGNORECASE)

        # Remove version numbers like 0.27.100.100
        cleaned = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', ' ', cleaned)

        # Extract all remaining numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', cleaned)
        candidates = []
        for n in numbers:
            val = float(n)
            if val >= 10 and val <= 10000:
                candidates.append(val)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

    @staticmethod
    def extract_metadata(text):
        """Extract scale and unit from floor plan text."""
        # Clean text before searching: remove UPI/reference numbers that look like scales
        cleaned = re.sub(r'\d+/\d+/\d+[\d/]*', ' ', text)

        # Find scale: 1:65, 1:100, SCALE: 1/80
        # Prefer colon notation (1:65) over slash (1/80) since slash conflicts with dates/refs
        scale = None
        # Try colon first (most reliable)
        scale_match = re.search(r'(?:scale\s*[:=]?\s*)?1\s*:\s*(\d+)', cleaned, re.IGNORECASE)
        if not scale_match:
            # Try slash only if preceded by "scale" keyword
            scale_match = re.search(r'scale\s*[:=]?\s*1\s*/\s*(\d+)', cleaned, re.IGNORECASE)
        if not scale_match:
            # Try standalone 1/NN where NN is a plausible scale (50-200)
            scale_match = re.search(r'\b1\s*/\s*(\d{2,3})\b', cleaned)
            if scale_match and not (50 <= int(scale_match.group(1)) <= 200):
                scale_match = None
        if scale_match:
            scale = f"1:{scale_match.group(1)}"

        # Find unit from construction notes
        unit = None
        unit_confidence = "low"

        # Explicit: "dimensions are in cm/meters/mm"
        unit_match = re.search(r'dimensions?\s+(?:are\s+)?in\s+(cm|meters?|m|mm|centimeters?|millimeters?)', text, re.IGNORECASE)
        if unit_match:
            raw = unit_match.group(1).lower()
            if raw.startswith('cm') or raw.startswith('centimeter'):
                unit, unit_confidence = "cm", "high"
            elif raw == 'mm' or raw.startswith('millimeter'):
                unit, unit_confidence = "mm", "high"
            elif raw in ('m', 'meters', 'meter'):
                unit, unit_confidence = "m", "high"

        # Also check for "(in cm)" pattern
        if not unit:
            paren_match = re.search(r'\(in\s+(cm|mm|m)\)', text, re.IGNORECASE)
            if paren_match:
                unit = paren_match.group(1).lower()
                unit_confidence = "high"

        return scale, unit, unit_confidence

    @staticmethod
    def filter_by_unit(candidates, unit):
        """Filter candidates to plausible dimension ranges based on unit.

        Meters: 1-50 (residential buildings are 3m-50m per side)
        Centimeters: 50-2000 (same range in cm)
        Millimeters: 500-20000 (same range in mm)
        Unknown: keep all
        """
        if not unit:
            return candidates

        if unit == "m":
            return [c for c in candidates if 1 <= c <= 50]
        elif unit == "cm":
            return [c for c in candidates if 50 <= c <= 2000]
        elif unit == "mm":
            return [c for c in candidates if 500 <= c <= 20000]
        return candidates

    def assess_confidence(self, candidates, scale, unit):
        """Score how confident we are that the top-2 candidates are building dimensions.

        Returns (score, width, length) where score 0-10.
        Score >= 6 means high confidence, use LiteParse result directly.
        """
        # Apply unit-aware filtering if unit is known
        filtered = self.filter_by_unit(candidates, unit) if unit else candidates

        # Fall back to unfiltered if filtering removed everything
        if len(filtered) < 2:
            filtered = candidates

        if len(filtered) < 2:
            return 0, None, None

        sorted_dims = sorted(filtered, reverse=True)
        width = sorted_dims[0]
        length = sorted_dims[1]

        score = 0

        # Scale found
        if scale:
            score += 2

        # Unit found explicitly
        if unit:
            score += 3

        # Top-2 are clearly larger than the rest (gap > 30%)
        if len(sorted_dims) >= 3:
            third = sorted_dims[2]
            if third > 0 and length / third > 1.3:
                score += 2
        else:
            score += 2  # only 2 candidates, no ambiguity

        # Top-2 are plausible building dimensions (not equal, ratio < 5:1)
        if width > 0 and length > 0:
            ratio = width / length
            if 0.2 < ratio < 5.0:
                score += 1

        # Sanity check area
        area_m2 = self.calculate_area_m2(width, length, unit or "cm")
        if area_m2 and 20 < area_m2 < 5000:
            score += 2

        return score, width, length

    # ── Tier 2: DeepSeek text-only (fast fallback) ──

    def deepseek_text_fallback(self, text, candidates):
        """Send extracted text + candidate list to DeepSeek for disambiguation.
        Text-only prompt, no image processing. Should complete in seconds."""
        if not self.check_ollama():
            return None

        prompt = self.TEXT_FALLBACK_PROMPT.format(
            candidates=', '.join(str(int(c) if c == int(c) else c) for c in candidates[:20])
        )
        # Include relevant text context (truncated)
        full_prompt = f"{prompt}\n\nFloor plan text:\n{text[:2000]}"

        self.log("Tier 2: DeepSeek text-only fallback...")
        try:
            result = subprocess.run(
                [self.ollama_path, "run", "deepseek-ocr", full_prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=60
            )
            output = result.stdout.strip()
            if not output:
                return None

            self.log(f"DeepSeek text response: {output[:200]}")
            return self._parse_wl_response(output)
        except subprocess.TimeoutExpired:
            self.log("DeepSeek text-only timed out after 60s", "WARNING")
            return None
        except Exception as e:
            self.log(f"DeepSeek text-only failed: {e}", "WARNING")
            return None

    def _parse_wl_response(self, text):
        """Parse W=___ L=___ UNIT=___ from DeepSeek response."""
        w_match = re.search(r'W\s*=\s*([\d.]+)', text)
        l_match = re.search(r'L\s*=\s*([\d.]+)', text)
        u_match = re.search(r'UNIT\s*=\s*(\w+)', text, re.IGNORECASE)

        if w_match and l_match:
            return {
                'width': float(w_match.group(1)),
                'length': float(l_match.group(1)),
                'unit': u_match.group(1).lower() if u_match else None
            }
        return None

    # ── Tier 3: DeepSeek image (last resort) ──

    def extract_from_image(self, image_path, custom_prompt=None):
        """Tier 3: Extract dimensions from a floor plan image via DeepSeek-OCR."""
        if not self.check_ollama():
            return {
                'success': False, 'data': None, 'raw_output': '',
                'error': 'Ollama not available'
            }

        image_path = Path(image_path).absolute()
        self.log(f"Tier 3: DeepSeek image OCR on {image_path.name}...")

        prompt = custom_prompt or self.EXTRACTION_PROMPT

        try:
            result = subprocess.run(
                [self.ollama_path, "run", "deepseek-ocr", prompt, str(image_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            self.log("DeepSeek image OCR timed out after 120s", "ERROR")
            return {
                'success': False, 'data': None, 'raw_output': '',
                'error': 'Timeout'
            }

        raw_output = (result.stdout or "").strip()
        self.log(f"DeepSeek image returned {len(raw_output)} chars")

        if not raw_output:
            return {
                'success': False, 'data': None, 'raw_output': '',
                'error': result.stderr or 'Empty output'
            }

        # Try structured parse first
        wl = self._parse_wl_response(raw_output)
        if wl:
            extracted = self.parse_dimensions_from_text(raw_output)
            if extracted:
                extracted['width'] = wl['width']
                extracted['length'] = wl['length']
                if wl.get('unit'):
                    extracted['unit'] = wl['unit']
            else:
                extracted = wl
            return {'success': True, 'data': extracted, 'raw_output': raw_output}

        # Fall back to text parsing
        extracted = self.parse_dimensions_from_text(raw_output)
        return {
            'success': extracted is not None,
            'data': extracted,
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

    def parse_grounding_format(self, text):
        """Parse grounding format output with text and bounding boxes"""
        import re

        # Pattern to match <|ref|>TEXT<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
        pattern = r'<\|ref\|>(.*?)<\/\|ref\|><\|det\|>\[\[([\d\s,]+)\]\]<\/\|det\|>'
        matches = re.findall(pattern, text)

        grounded_items = []
        for text_content, coords_str in matches:
            try:
                # Parse coordinates
                coords = [int(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    grounded_items.append({
                        'text': text_content.strip(),
                        'bbox': [x1, y1, x2, y2],
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2
                    })
            except (ValueError, IndexError):
                # Skip malformed entries
                continue

        return grounded_items

    def parse_dimensions_from_text(self, text):
        """Parse dimensions from plain text using smart filtering."""

        # First, try to parse grounding format
        grounded_items = self.parse_grounding_format(text)

        if grounded_items:
            self.log(f"Extracted {len(grounded_items)} grounded text items")
            all_text = ' '.join([item['text'] for item in grounded_items])
        else:
            all_text = text

        # Use smart filtering to get candidate dimensions
        candidates = self.filter_candidate_dimensions(all_text)

        if len(candidates) < 2:
            return None

        # Extract metadata
        scale, unit, unit_confidence = self.extract_metadata(all_text)

        # If no explicit unit found, infer from values
        if not unit:
            sorted_dims = sorted(candidates, reverse=True)
            if sorted_dims[0] > 1000:
                unit, unit_confidence = "mm", "medium"
            elif sorted_dims[0] < 50:
                unit, unit_confidence = "m", "medium"
            else:
                unit, unit_confidence = "cm", "low"

        # Get overall dimensions (two largest)
        sorted_dims = sorted(candidates, reverse=True)
        width = sorted_dims[0]
        length = sorted_dims[1]

        result = {
            'scale': scale or 'unknown',
            'width': width,
            'length': length,
            'unit': unit,
            'unit_confidence': unit_confidence,
            'all_dimensions': sorted_dims[:20]
        }

        # Add grounded items if available
        if grounded_items:
            result['grounded_items'] = grounded_items
            result['total_items'] = len(grounded_items)

        return result

    def validate_structure(self, data):
        """Validate that JSON has required fields"""
        required_fields = ['width', 'length', 'unit']
        return all(field in data for field in required_fields)

    def extract_from_pdf(self, pdf_path, custom_prompt=None, dpi=300):
        """Extract dimensions from PDF using three-tier pipeline.

        Tier 1: LiteParse (instant, no model needed)
        Tier 2: DeepSeek-OCR text-only (seconds, uses LiteParse text)
        Tier 3: DeepSeek-OCR image (minutes, last resort)
        """
        pdf_path = Path(pdf_path)
        self.log(f"Processing PDF: {pdf_path.name}")

        # ── Tier 1: LiteParse ──
        lp_text = self.extract_with_liteparse(pdf_path)

        if lp_text:
            candidates = self.filter_candidate_dimensions(lp_text)
            scale, unit, unit_confidence = self.extract_metadata(lp_text)
            confidence, width, length = self.assess_confidence(candidates, scale, unit)

            self.log(f"  Candidates: {candidates[:10]}")
            self.log(f"  Scale: {scale}, Unit: {unit} ({unit_confidence})")
            self.log(f"  Confidence: {confidence}/10, W={width}, L={length}")

            if confidence >= 6 and width and length:
                self.log("Tier 1 SUCCESS: high confidence from LiteParse alone")
                # If no explicit unit, infer
                if not unit:
                    if width > 1000:
                        unit, unit_confidence = "mm", "medium"
                    elif width < 50:
                        unit, unit_confidence = "m", "medium"
                    else:
                        unit, unit_confidence = "cm", "low"

                data = {
                    'scale': scale or 'unknown',
                    'width': width,
                    'length': length,
                    'unit': unit,
                    'unit_confidence': unit_confidence,
                    'all_dimensions': sorted(candidates, reverse=True)[:20],
                    'extraction_tier': 1,
                    'extraction_method': 'liteparse'
                }
                return {'success': True, 'data': data, 'raw_output': lp_text}

            # ── Tier 2: DeepSeek text-only ──
            if candidates:
                wl = self.deepseek_text_fallback(lp_text, candidates)
                if wl:
                    self.log("Tier 2 SUCCESS: DeepSeek text-only resolved dimensions")
                    data = {
                        'scale': scale or 'unknown',
                        'width': wl['width'],
                        'length': wl['length'],
                        'unit': wl.get('unit') or unit or 'cm',
                        'unit_confidence': 'medium' if wl.get('unit') else unit_confidence,
                        'all_dimensions': sorted(candidates, reverse=True)[:20],
                        'extraction_tier': 2,
                        'extraction_method': 'deepseek_text'
                    }
                    return {'success': True, 'data': data, 'raw_output': lp_text}

        # ── Tier 3: DeepSeek image (last resort) ──
        self.log("Falling back to Tier 3: DeepSeek image OCR...")

        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
        if not images:
            return {'success': False, 'data': None, 'raw_output': '', 'error': 'Failed to convert PDF'}

        temp_path = Path(".temp_floorplan.png")
        images[0].save(temp_path, "PNG")

        result = self.extract_from_image(temp_path, custom_prompt)

        temp_path.unlink(missing_ok=True)

        if result.get('success') and result.get('data'):
            result['data']['extraction_tier'] = 3
            result['data']['extraction_method'] = 'deepseek_image'

        return result

    def save_results(self, result, output_path):
        """Save extraction results in multiple formats"""
        output_path = Path(output_path)
        base_name = output_path.stem

        if result['success'] and result['data']:
            data = result['data']

            # Save grounded data if available
            if 'grounded_items' in data and data['grounded_items']:
                grounded_items = data['grounded_items']

                # 1. Save grounded JSON
                grounded_json_path = output_path.parent / f"{base_name}_grounded.json"
                grounded_json_path.write_text(json.dumps(grounded_items, indent=2), encoding='utf-8')
                self.log(f"✓ Saved grounded JSON: {grounded_json_path}")

                # 2. Save CSV
                csv_path = output_path.parent / f"{base_name}_grounded.csv"
                self.save_as_csv(grounded_items, csv_path)
                self.log(f"✓ Saved CSV: {csv_path}")

                # 3. Save plain text (just text, no coordinates)
                text_path = output_path.parent / f"{base_name}_full_text.txt"
                text_content = '\n'.join([item['text'] for item in grounded_items])
                text_path.write_text(text_content, encoding='utf-8')
                self.log(f"✓ Saved plain text: {text_path}")
            else:
                # No grounded data, save raw output as plain text
                text_path = output_path.parent / f"{base_name}_full_text.txt"
                text_path.write_text(result['raw_output'], encoding='utf-8')
                self.log(f"✓ Saved extracted text: {text_path}")

            # 4. Save dimension parsing attempt
            dimensions_only = {k: v for k, v in data.items() if k not in ['grounded_items']}
            json_path = output_path.with_suffix('.json')
            json_path.write_text(json.dumps(dimensions_only, indent=2), encoding='utf-8')
            self.log(f"✓ Saved dimensions JSON: {json_path}")

            # 5. Save raw output for debugging
            raw_path = output_path.with_suffix('.raw.txt')
            raw_path.write_text(result['raw_output'], encoding='utf-8')
            self.log(f"✓ Saved raw output: {raw_path}")

            # Print summary
            self.print_summary(result['data'])

            return json_path
        else:
            self.log("✗ Extraction failed - no valid data found", "ERROR")
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

    def save_as_csv(self, grounded_items, csv_path):
        """Save grounded items as CSV"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['text', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'center_x', 'center_y'])
            # Data
            for item in grounded_items:
                writer.writerow([
                    item['text'],
                    item['bbox'][0],
                    item['bbox'][1],
                    item['bbox'][2],
                    item['bbox'][3],
                    item['width'],
                    item['height'],
                    item['center_x'],
                    item['center_y']
                ])

    @staticmethod
    def calculate_area_m2(width, length, unit):
        """Calculate area in m2 from width, length, and unit."""
        if not width or not length:
            return None
        if unit == 'cm':
            return (width * length) / 10000
        elif unit == 'mm':
            return (width * length) / 1000000
        elif unit == 'm':
            return width * length
        return None

    def print_summary(self, data):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)

        # Show extraction method
        tier = data.get('extraction_tier', '?')
        method = data.get('extraction_method', 'unknown')
        print(f"Extraction: Tier {tier} ({method})")

        # Show grounded items info if available
        if 'grounded_items' in data and data['grounded_items']:
            total = data.get('total_items', len(data['grounded_items']))
            print(f"Extracted {total} text items with locations")

            print("\nSample text items (first 10):")
            for i, item in enumerate(data['grounded_items'][:10], 1):
                text = item['text'][:50]
                print(f"  {i}. {text}")

            if total > 10:
                print(f"  ... and {total - 10} more items")

            print("\n" + "-"*60)

        unit = data.get('unit', 'unknown')
        width = data.get('width', 'N/A')
        length = data.get('length', 'N/A')

        print(f"\nScale: {data.get('scale', 'N/A')}")
        print(f"Width: {width} {unit}")
        print(f"Length: {length} {unit}")
        print(f"Unit confidence: {data.get('unit_confidence', 'N/A')}")

        if 'width' in data and 'length' in data:
            area_m2 = self.calculate_area_m2(data['width'], data['length'], unit)
            if area_m2:
                print(f"Calculated Area: {area_m2:.2f} m2")
                # Sanity check
                if area_m2 < 20 or area_m2 > 5000:
                    print(f"  WARNING: Area {area_m2:.0f} m2 is outside typical range (20-5000)")

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
