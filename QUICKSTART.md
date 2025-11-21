# Quick Start Guide

Get DeepSeek-OCR running on your laptop in 5 minutes.

## Prerequisites

- Python 3.8+ installed
- 10GB free disk space
- (Optional but recommended) NVIDIA GPU with CUDA support

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**CPU-only users** (no NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Authenticate with Hugging Face

```bash
pip install huggingface-hub
huggingface-cli login
```

Get your token from: https://huggingface.co/settings/tokens

## Quick Test

### Test 1: Web Interface (Easiest)

```bash
python ocr_gradio_app.py
```

Open http://127.0.0.1:7860 in your browser, upload an image, and click "Process".

### Test 2: Command Line

```bash
python ocr_basic.py your_image.jpg
```

Results will be saved to `output_YYYYMMDD_HHMMSS/result.txt`

### Test 3: See All Modes

```bash
python ocr_basic.py --list-modes
```

## Common First-Time Issues

**"No module named 'transformers'"**
→ Run: `pip install -r requirements.txt`

**"Token is required"**
→ Run: `huggingface-cli login` and paste your token

**"CUDA out of memory"**
→ Use CPU mode: `python ocr_basic.py image.jpg --device cpu`

**"Model download stuck"**
→ Model is 6GB, first download takes time. Be patient.

## Testing Your Use Case

1. **Prepare** 5-10 sample images from your actual use case
2. **Create** a test folder: `mkdir test_images`
3. **Copy** your images to `test_images/`
4. **Run batch processing**:
   ```bash
   python batch_process.py test_images --mode document
   ```
5. **Review** results in `results/` folder

## What Mode to Use?

| Your Use Case | Mode | Example |
|---------------|------|---------|
| General documents, PDFs | `document` | Contracts, reports |
| Charts, graphs, plots | `chart` | Business charts, data viz |
| Tables, spreadsheets | `table` | Financial statements |
| Handwritten notes | `handwriting` | Meeting notes, forms |
| Math/equations | `equation` | Textbooks, papers |
| Chemical structures | `chemistry` | Lab reports |
| Multiple languages | `multilingual` | International docs |
| Memes, social images | `plain` | Social media |

## Next Steps

Read the full [README.md](README.md) for:
- Advanced usage
- Batch processing
- Performance optimization
- Troubleshooting

## Example Commands

```bash
# Process a single document
python ocr_basic.py invoice.pdf --mode table

# Process with custom prompt
python ocr_basic.py chart.png --prompt "<image>\nExtract the chart title and all data points."

# Batch process a folder
python batch_process.py ./my_documents --mode document --output ./results

# Launch web interface
python ocr_gradio_app.py
```

## Performance Expectations

| Hardware | Speed per Image | Notes |
|----------|----------------|-------|
| RTX 4090 | 3-5 seconds | Excellent |
| RTX 3060 | 8-12 seconds | Good |
| GTX 1660 | 15-25 seconds | Acceptable |
| CPU only | 1-5 minutes | Very slow |

---

**That's it!** You're ready to evaluate DeepSeek-OCR on your use case.

For questions or issues, check the full README.md or open an issue.
