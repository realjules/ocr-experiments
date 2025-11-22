# Ollama DeepSeek-OCR Scripts

Simple scripts for using Ollama's deepseek-ocr model for batch image processing and PDF conversion.

## Prerequisites

1. Install Ollama: https://ollama.com
2. Pull the model:
   ```bash
   ollama pull deepseek-ocr
   ```

## PDF to Markdown Converter

Convert entire PDFs to markdown with `pdf_to_markdown.py`:

### Installation

```bash
pip install -r requirements.txt
```

**System Requirements:**
- **Linux**: `sudo apt install poppler-utils`
- **Mac**: `brew install poppler`
- **Windows**: Download poppler from https://github.com/oschwartz10612/poppler-windows/releases/

### Usage

```bash
# Basic usage - converts document.pdf to document.md
python pdf_to_markdown.py document.pdf

# Custom output path
python pdf_to_markdown.py document.pdf -o output.md

# Higher quality (slower, better for small text)
python pdf_to_markdown.py document.pdf --dpi 600

# Custom prompt for specific extraction
python pdf_to_markdown.py document.pdf --prompt "Extract all text and preserve formatting"

# No page separators (continuous markdown)
python pdf_to_markdown.py document.pdf --no-separator
```

**Features:**
- Converts each PDF page to an image
- Processes through deepseek-ocr with customizable prompts
- Combines all pages into a single markdown file
- Configurable DPI for quality/speed tradeoff
- Progress tracking for multi-page documents

## Interactive Usage

For single images, just use Ollama directly:

```bash
ollama run deepseek-ocr "Extract text from this image" path/to/image.jpg
```

## Batch Processing Scripts

### Bash Script (Linux/Mac/WSL)

Save as `batch_ocr.sh`:

```bash
#!/bin/bash

# Process all images in a directory
INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-./ocr_results}"
PROMPT="${3:-Extract all text from this image}"

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    [ -e "$img" ] || continue

    filename=$(basename "$img")
    name="${filename%.*}"

    echo "Processing: $filename"

    ollama run deepseek-ocr "$PROMPT" "$img" > "$OUTPUT_DIR/${name}.txt"

    echo "✓ Saved to: $OUTPUT_DIR/${name}.txt"
done

echo "Done! Results in: $OUTPUT_DIR"
```

Usage:
```bash
chmod +x batch_ocr.sh
./batch_ocr.sh ./images ./results "Extract text from this document"
```

### PowerShell Script (Windows)

Save as `batch_ocr.ps1`:

```powershell
param(
    [string]$InputDir = ".",
    [string]$OutputDir = ".\ocr_results",
    [string]$Prompt = "Extract all text from this image"
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$extensions = @("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

foreach ($ext in $extensions) {
    Get-ChildItem -Path $InputDir -Filter $ext | ForEach-Object {
        $filename = $_.Name
        $basename = $_.BaseName
        $outputFile = Join-Path $OutputDir "$basename.txt"

        Write-Host "Processing: $filename"

        ollama run deepseek-ocr "$Prompt" $_.FullName | Out-File -FilePath $outputFile

        Write-Host "✓ Saved to: $outputFile"
    }
}

Write-Host "Done! Results in: $OutputDir"
```

Usage:
```powershell
.\batch_ocr.ps1 -InputDir .\images -OutputDir .\results -Prompt "Extract text from this document"
```

## Different OCR Tasks

### Extract Tables
```bash
ollama run deepseek-ocr "Extract all tables as HTML" table.jpg > table.html
```

### Extract Handwriting
```bash
ollama run deepseek-ocr "Extract all handwritten text" notes.jpg > notes.txt
```

### Extract Math Equations
```bash
ollama run deepseek-ocr "Extract all mathematical equations in LaTeX format" equation.jpg > equations.tex
```

### Convert Document to Markdown
```bash
ollama run deepseek-ocr "Convert the document to markdown" document.jpg > document.md
```

### Extract Charts
```bash
ollama run deepseek-ocr "Parse all charts and tables. Extract data as HTML tables" chart.png > chart.html
```

## Python Script (Optional)

If you prefer Python, save as `batch_ocr.py`:

```python
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def process_images(input_dir=".", output_dir="ocr_results", prompt="Extract all text from this image"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    images = [f for f in input_path.iterdir() if f.suffix in extensions]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images to process\n")

    for img in images:
        print(f"Processing: {img.name}")
        output_file = output_path / f"{img.stem}.txt"

        result = subprocess.run(
            ["ollama", "run", "deepseek-ocr", prompt, str(img)],
            capture_output=True,
            text=True
        )

        output_file.write_text(result.stdout)
        print(f"✓ Saved to: {output_file}\n")

    print(f"Done! Results in: {output_dir}")

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ocr_results"
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Extract all text from this image"

    process_images(input_dir, output_dir, prompt)
```

Usage:
```bash
chmod +x batch_ocr.py
./batch_ocr.py ./images ./results "Extract text from this document"
```

## Tips

1. **Model loading**: First request takes longer (model loads into memory)
2. **Concurrent processing**: For faster batch processing, run multiple ollama instances in parallel:
   ```bash
   for img in images/*.jpg; do
       ollama run deepseek-ocr "Extract text" "$img" > "results/$(basename "$img" .jpg).txt" &
   done
   wait
   ```
3. **Custom prompts**: Tailor prompts to your specific needs for better results
4. **Image quality**: Higher resolution images = better accuracy (but slower)

## Example Workflows

### Receipt Processing
```bash
for receipt in receipts/*.jpg; do
    ollama run deepseek-ocr "Extract: vendor name, date, total amount, items" "$receipt" > "parsed/$(basename "$receipt" .jpg).txt"
done
```

### Invoice Data Extraction
```bash
ollama run deepseek-ocr "Extract as JSON: {invoice_number, date, vendor, total, line_items}" invoice.pdf
```

### Form Processing
```bash
ollama run deepseek-ocr "Extract all filled form fields and their values" form.jpg
```

## Resources

- Ollama: https://ollama.com
- DeepSeek-OCR Model: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- DeepSeek-OCR Paper: https://arxiv.org/abs/2501.12558
