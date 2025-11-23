# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a collection of OCR tools using DeepSeek-OCR via Ollama for extracting text and structured data from documents, with a specialized focus on architectural floor plans for government tax assessment.

**Core Architecture:**
- **Ollama Integration**: All tools communicate with the deepseek-ocr model through subprocess calls to `ollama run deepseek-ocr`
- **Two-Stage Processing**: PDF → Image conversion (via pdf2image/poppler) → OCR extraction
- **Prompt Engineering**: Different prompts yield different results - JSON requests can trigger Ollama batching errors, while natural language prompts work reliably

## Prerequisites

**Must be installed before running any scripts:**
1. Ollama: https://ollama.com
2. DeepSeek-OCR model: `ollama pull deepseek-ocr`
3. Python dependencies: `pip install -r requirements.txt`
4. Poppler (for PDF conversion):
   - Linux: `sudo apt install poppler-utils`
   - Mac: `brew install poppler`
   - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/

## Main Scripts

### 1. `architectural_extractor.py` - Government Tax Assessment Tool
**Purpose**: Extract building dimensions (width, length, scale, unit) from architectural floor plans as JSON.

**Usage:**
```bash
python architectural_extractor.py floorplan.pdf
python architectural_extractor.py floorplan.pdf -o output.json
python architectural_extractor.py floorplan.pdf --dpi 600
```

**Output JSON Structure:**
```json
{
  "scale": "1:65",
  "width": 840,
  "length": 670,
  "unit": "cm",
  "unit_confidence": "high",
  "all_dimensions": [840, 670, 370, ...]
}
```

**Critical Implementation Details:**
- Uses natural language prompt (NOT JSON request) to avoid Ollama batching errors
- Falls back to regex parsing if model doesn't return JSON
- Auto-detects units (cm, mm, m) from text or infers from dimension values
- Saves debug files (`debug_stdout.txt`, `debug_stderr.txt`) for troubleshooting
- **Encoding**: Must use `encoding='utf-8', errors='replace'` on Windows to handle special characters
- **Timeout**: 300 seconds (5 minutes) - matches successful configurations

### 2. `pdf_to_markdown.py` - PDF to Markdown Converter
**Purpose**: Convert multi-page PDFs to markdown text.

**Usage:**
```bash
python pdf_to_markdown.py document.pdf
python pdf_to_markdown.py document.pdf -o output.md --dpi 600
python pdf_to_markdown.py document.pdf --no-separator
```

**Key Details:**
- Uses prompt: `"<image>\nConvert the document to markdown."`
- This simple prompt pattern is proven to work reliably
- Processes each page separately, then combines with `---` separators

### 3. `ocr_agent.py` - Intelligent Document Processor
**Purpose**: Auto-detect document types and extract structured data with confidence scoring.

**Supported Types:** invoice, receipt, contract, form, architectural, table, document

**Usage:**
```bash
python ocr_agent.py document.pdf --type invoice
python ocr_agent.py ./invoices/ --output ./processed
```

**Architecture:**
- Uses keyword matching for document type detection
- Type-specific prompts for better extraction
- Outputs both JSON (if structured) and metadata with confidence scores

## Critical Technical Constraints

### Ollama Subprocess Communication
**Working Pattern:**
```python
result = subprocess.run(
    ["ollama", "run", "deepseek-ocr", prompt, str(image_path)],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',  # Critical for Windows
    timeout=300
)
```

**Common Errors & Solutions:**

1. **UnicodeDecodeError (cp1252)**: Windows Python defaults to cp1252. MUST use `encoding='utf-8', errors='replace'`

2. **Ollama Batching Error**: `"SameBatch may not be specified within numKeep"`
   - Cause: Complex prompts or JSON requests trigger internal Ollama/vLLM errors
   - Solution: Use simple natural language prompts, parse output with regex

3. **Empty stdout**: Check stderr - errors go there, not stdout

4. **Timeout (5min+)**: Prompt is too complex or model is stuck. Simplify prompt.

### PDF to Image Conversion
- Always convert to absolute paths: `Path(image_path).absolute()`
- Save to temporary files (`.temp_floorplan.png`) and cleanup after processing
- DPI 300 is default (balance quality/speed), use 600 for small text

### Prompt Engineering Rules
**Do:**
- Keep prompts short and natural: `"<image>\nExtract dimensions from this floor plan."`
- Use proven patterns from `pdf_to_markdown.py`

**Don't:**
- Ask for JSON directly in complex formats
- Use lengthy instructions with multiple numbered steps
- Include extensive example outputs in the prompt

## Integration with External Systems

**Architectural Extractor → Claude + tax_calculator tool:**
```python
from architectural_extractor import ArchitecturalExtractor

extractor = ArchitecturalExtractor()
result, _ = extractor.process("floorplan.pdf")

if result['success']:
    dimensions = result['data']
    # Pass dimensions JSON to Claude with tax_calculator tool
```

**Expected workflow:**
1. Extract dimensions as JSON (width, length, unit)
2. Claude uses tax_calculator tool with these dimensions
3. Calculate built-up area and tax assessment

## Debugging

**All scripts save debug files:**
- `debug_stdout.txt` - OCR model output
- `debug_stderr.txt` - Ollama errors
- `[filename]_dimensions.raw.txt` - Raw text output
- `[filename]_dimensions.error.txt` - Error details

**Enable verbose logging:** All scripts have `verbose=True` by default, shows:
- Command being run
- Return codes
- Output lengths
- Processing steps

## Unit Detection Logic

Units can vary (no standard). Detection order:
1. **Explicit**: Search for "cm", "mm", "m" in text → `confidence: "high"`
2. **Inferred from scale + values**:
   - 100-1000 + scale 1:50-1:100 → likely cm
   - 1000-10000 + scale 1:50-1:100 → likely mm
   - 1-20 with decimals → likely m
   → `confidence: "medium"`
3. **Guessed from values alone** → `confidence: "low"`

## Limitations

**Spatial Understanding:**
- DeepSeek-OCR extracts TEXT but NOT spatial layout
- Floor plan drawings lose geometric relationships
- Dimensions are captured, but not which wall they measure
- Room positions/shapes are lost

**Not suitable for:**
- Reconstructing floor plan layouts
- Calculating individual room areas (use overall dimensions only)
- Understanding visual diagrams/flowcharts

**Suitable for:**
- Extracting overall building dimensions
- Reading dimension numbers and scale
- Text-heavy documents (invoices, contracts, reports)
