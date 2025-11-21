# DeepSeek-OCR Local Testing Setup

This repository contains scripts to run DeepSeek-OCR locally on your laptop for evaluating its capabilities on your specific use case.

## What is DeepSeek-OCR?

DeepSeek-OCR is a 3B parameter open vision-language model designed for document understanding. It can:

- Extract text from documents, handwritten notes, and images
- Parse tables and charts into structured formats (HTML)
- Extract mathematical equations as LaTeX
- Recognize chemical formulas and structures
- Handle multilingual documents
- Process memes and social media images

## System Requirements

### Minimum Requirements
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB free space for model weights
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8 or higher

### Recommended for Good Performance
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or higher
- **RAM**: 16GB+

### Performance Notes
- **With GPU**: 5-15 seconds per image
- **Without GPU (CPU only)**: 1-5 minutes per image (VERY slow, but works)

## Installation

### Step 1: Clone or Download This Repository

If you haven't already, save all files to a directory on your laptop.

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for CPU-only users**: If you don't have a CUDA-capable GPU, install CPU-only PyTorch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4: Hugging Face Authentication (Required)

DeepSeek-OCR is hosted on Hugging Face and requires authentication:

1. Create a free account at https://huggingface.co
2. Generate an access token:
   - Go to Settings → Access Tokens
   - Click "New token"
   - Select "Read" permission
   - Copy the token

3. Login via CLI:
```bash
pip install huggingface-hub
huggingface-cli login
```
Paste your token when prompted.

## Usage

### Option 1: Command-Line Script (Recommended for Testing)

The basic script is perfect for quick tests and batch processing.

#### Basic Usage

```bash
python ocr_basic.py <image_path>
```

Example:
```bash
python ocr_basic.py my_document.jpg
```

#### List Available Modes

```bash
python ocr_basic.py --list-modes
```

Available modes:
- `document` - Extract full document as Markdown
- `chart` - Extract charts/tables as HTML
- `chemistry` - Extract chemical structures
- `handwriting` - OCR handwritten text
- `equation` - Extract equations as LaTeX
- `table` - Extract tables as HTML
- `multilingual` - Extract multi-language content
- `plain` - Fast plain text OCR

#### Use Specific Mode

```bash
python ocr_basic.py invoice.pdf --mode table
python ocr_basic.py notes.jpg --mode handwriting
python ocr_basic.py chart.png --mode chart
```

#### Custom Prompt

```bash
python ocr_basic.py image.jpg --prompt "<image>\nExtract only the title and author name."
```

#### Specify Output Directory

```bash
python ocr_basic.py document.jpg --output ./results/doc1
```

#### Force CPU Mode

```bash
python ocr_basic.py image.jpg --device cpu
```

### Option 2: Gradio Web Interface (Recommended for Interactive Testing)

The Gradio app provides a user-friendly web interface.

#### Launch the App

```bash
python ocr_gradio_app.py
```

Then open your browser to: **http://127.0.0.1:7860**

#### Using the Web Interface

1. Upload an image using the file picker
2. Select an OCR mode from the dropdown
3. (Optional) Adjust advanced settings
4. Click "Process Image"
5. View extracted text and bounding boxes

The model will load automatically on first use (this takes 30-60 seconds).

## Testing Your Use Case

### Step 1: Prepare Test Images

Gather 5-10 representative images that reflect your actual use case:
- Scanned documents
- Photos of receipts or invoices
- Screenshots of charts
- Handwritten notes
- Technical diagrams
- Multilingual documents

### Step 2: Run Tests

Create a test directory:

```bash
mkdir test_images
# Copy your test images here
```

Process each image:

```bash
python ocr_basic.py test_images/image1.jpg --mode document
python ocr_basic.py test_images/image2.jpg --mode table
python ocr_basic.py test_images/image3.jpg --mode chart
```

Or use the Gradio app for visual comparison.

### Step 3: Evaluate Results

For each test, check:

1. **Accuracy**: Is the extracted text correct?
2. **Structure**: Is formatting preserved (tables, lists, etc.)?
3. **Completeness**: Is all text detected?
4. **Speed**: Is processing time acceptable?
5. **Output Format**: Is the output format usable for your workflow?

Results are saved in timestamped folders:
- `output_YYYYMMDD_HHMMSS/result.txt` - Extracted text
- `output_YYYYMMDD_HHMMSS/result_with_boxes.jpg` - Visual overlay

### Step 4: Compare with Alternatives

If evaluating multiple OCR solutions, test the same images with:
- Tesseract OCR
- PaddleOCR
- Cloud services (Google Vision API, AWS Textract)

Compare on: accuracy, speed, ease of use, and cost.

## Common Issues and Solutions

### Issue: "CUDA out of memory"

**Solution**: Reduce image size or use CPU mode:
```bash
python ocr_basic.py image.jpg --device cpu
```

### Issue: "Model download is very slow"

**Solution**: The model is ~6GB. On slow connections, this can take 10-30 minutes on first run. Be patient. Subsequent runs will be fast.

### Issue: "Division by zero error"

**Solution**: This is a known harmless bug in the model's compression stats. The script handles it automatically. Results are still valid.

### Issue: "Extracted text is empty"

**Possible causes**:
- Image quality too low
- Text is too small in the image
- Wrong mode selected

**Try**:
- Increase image resolution
- Use a different mode
- Adjust base_size/image_size in advanced settings

### Issue: "Running on CPU is extremely slow"

**Solution**: This is expected. CPU inference can take 1-5 minutes per image. Consider:
- Using a cloud GPU service (Google Colab, Paperspace)
- Batch processing overnight
- Upgrading to a GPU-equipped machine

## Output Files Explained

Each run creates a directory with these files:

- **input.png** - Preprocessed input image
- **result.txt** or **result.mmd** - Extracted text (Markdown or plain)
- **result_with_boxes.jpg** - Input image with bounding boxes overlaid
- **result_boxes.json** - Detected text regions in JSON format (if available)

## Tips for Best Results

1. **Image Quality**: Higher resolution = better accuracy (but slower)
2. **Mode Selection**: Choose the mode that matches your document type
3. **Preprocessing**: Rotate/crop images before processing if needed
4. **Batch Processing**: Write a simple loop script for multiple files
5. **GPU Usage**: Always use GPU if available (20-50x faster)

## Example Batch Processing Script

Create `batch_process.py`:

```python
import os
import glob
from ocr_basic import load_model, process_image

# Load model once
model, tok = load_model()

# Process all images in a folder
for img_path in glob.glob("test_images/*.jpg"):
    print(f"\nProcessing: {img_path}")
    output_dir = f"results/{os.path.basename(img_path)}_output"
    process_image(model, tok, img_path, mode="document", output_dir=output_dir)
    print(f"Saved to: {output_dir}")
```

Run:
```bash
python batch_process.py
```

## Next Steps

After evaluating DeepSeek-OCR on your use case:

1. **Document findings**: Track accuracy, speed, and edge cases
2. **Compare alternatives**: Test other OCR solutions
3. **Prototype integration**: If results are good, integrate into your pipeline
4. **Optimize**: Fine-tune parameters or add preprocessing steps

## Resources

- **Model**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Paper**: https://arxiv.org/abs/2501.12558
- **GitHub**: https://github.com/deepseek-ai/DeepSeek-OCR

## License

This code is provided as-is for evaluation purposes. Check the DeepSeek-OCR model license on Hugging Face for commercial use restrictions.

## Support

For issues with:
- **This code**: Open an issue or modify the scripts as needed
- **The model**: Check the official DeepSeek-OCR repository
- **Hugging Face**: Visit https://huggingface.co/docs

---

**Happy testing!** If you find this useful for your use case, consider starring the original DeepSeek-OCR repository.
