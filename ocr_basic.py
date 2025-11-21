#!/usr/bin/env python3
"""
Basic DeepSeek-OCR script for local testing
Usage: python ocr_basic.py <image_path> [--mode <mode>] [--output <output_dir>]
"""

import os
import argparse
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from datetime import datetime

# Available OCR modes
MODES = {
    "document": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "desc": "Extract full document as Markdown",
        "base_size": 1024,
        "image_size": 640
    },
    "chart": {
        "prompt": "<image>\nParse all charts and tables. Extract data as HTML tables.",
        "desc": "Extract charts/tables as HTML",
        "base_size": 1024,
        "image_size": 640
    },
    "chemistry": {
        "prompt": "<image>\nExtract all chemical formulas and SMILES.",
        "desc": "Extract chemical structures",
        "base_size": 1024,
        "image_size": 768
    },
    "handwriting": {
        "prompt": "<image>\n<|grounding|>Extract all handwritten text.",
        "desc": "OCR handwritten text",
        "base_size": 1024,
        "image_size": 640
    },
    "equation": {
        "prompt": "<image>\nExtract all mathematical equations in LaTeX format.",
        "desc": "Extract equations as LaTeX",
        "base_size": 1024,
        "image_size": 640
    },
    "table": {
        "prompt": "<image>\nExtract all tables as HTML.",
        "desc": "Extract tables as HTML",
        "base_size": 1024,
        "image_size": 640
    },
    "multilingual": {
        "prompt": "<image>\n<|grounding|>Extract text. Preserve all languages and structure.",
        "desc": "Extract multi-language content",
        "base_size": 1024,
        "image_size": 640
    },
    "plain": {
        "prompt": "<image>\nFree OCR.",
        "desc": "Fast plain text OCR",
        "base_size": 768,
        "image_size": 512
    }
}


def load_model(model_id="deepseek-ai/DeepSeek-OCR", device="auto"):
    """Load DeepSeek-OCR model and tokenizer"""
    print(f"Loading model: {model_id}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load model with appropriate settings for device
    if device == "cuda":
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager"
        ).to(dtype=torch.bfloat16, device="cuda").eval()
    else:
        # CPU mode - use float32
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager"
        ).to(dtype=torch.float32, device="cpu").eval()
        print("WARNING: Running on CPU will be very slow. GPU recommended.")

    return model, tok


def process_image(model, tokenizer, image_path, mode="document", output_dir=None, custom_prompt=None):
    """Process an image with DeepSeek-OCR"""

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")

    # Resize if too large
    if max(img.size) > 2000:
        scale = 2000 / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"Resized image from original to {new_size}")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Save processed image
    processed_img_path = os.path.join(output_dir, "input.png")
    img.save(processed_img_path)

    # Get prompt
    if custom_prompt:
        prompt = custom_prompt
    elif mode in MODES:
        prompt = MODES[mode]["prompt"]
        base_size = MODES[mode]["base_size"]
        image_size = MODES[mode]["image_size"]
    else:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(MODES.keys())}")

    print(f"\nProcessing with mode: {mode}")
    print(f"Prompt: {prompt[:50]}...")

    # Run inference
    print("Running OCR inference...")
    try:
        with torch.inference_mode():
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=processed_img_path,
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=False,
                save_results=True,
                test_compress=True
            )
    except ZeroDivisionError:
        print("Warning: Division by zero in compression ratio (can be ignored)")

    # Read results
    result_file = os.path.join(output_dir, "result.mmd")
    if not os.path.exists(result_file):
        result_file = os.path.join(output_dir, "result.txt")

    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
    else:
        extracted_text = "[No text extracted]"

    # Print results
    print("\n" + "="*80)
    print("EXTRACTED TEXT:")
    print("="*80)
    print(extracted_text)
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Extracted text: {result_file}")

    boxed_img = os.path.join(output_dir, "result_with_boxes.jpg")
    if os.path.exists(boxed_img):
        print(f"  - Image with boxes: {boxed_img}")

    return extracted_text, output_dir


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Local Testing")
    parser.add_argument("image", help="Path to image file to process")
    parser.add_argument("--mode", "-m", default="document",
                       choices=list(MODES.keys()),
                       help="OCR mode to use")
    parser.add_argument("--output", "-o", help="Output directory (default: auto-generated)")
    parser.add_argument("--prompt", "-p", help="Custom prompt (overrides mode)")
    parser.add_argument("--device", "-d", default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--list-modes", action="store_true",
                       help="List available modes and exit")

    args = parser.parse_args()

    # List modes if requested
    if args.list_modes:
        print("\nAvailable OCR modes:")
        for mode_name, mode_info in MODES.items():
            print(f"  {mode_name:15s} - {mode_info['desc']}")
        return

    # Check image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    # Load model
    model, tokenizer = load_model(device=args.device)

    # Process image
    process_image(
        model,
        tokenizer,
        args.image,
        mode=args.mode,
        output_dir=args.output,
        custom_prompt=args.prompt
    )


if __name__ == "__main__":
    main()
