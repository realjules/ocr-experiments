#!/usr/bin/env python3
"""
DeepSeek-OCR Gradio Web Interface for Local Testing
Usage: python ocr_gradio_app.py
"""

import os
import time
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
import random
import string

# OCR Modes Configuration
DEMO_MODES = {
    "Document → Markdown": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "desc": "Extracts full document text as Markdown, preserving structure (headings, tables, lists, etc.).",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    },
    "Chart Deep Parsing": {
        "prompt": "<image>\nParse all charts and tables. Extract data as HTML tables.",
        "desc": "Extracts tabular/chart data into HTML tables.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    },
    "Chemical Formula Recognition": {
        "prompt": "<image>\nExtract all chemical formulas and SMILES.",
        "desc": "Extracts chemical structures and formulae.",
        "base_size": 1024, "image_size": 768, "crop_mode": False
    },
    "Handwriting OCR": {
        "prompt": "<image>\n<|grounding|>Extract all handwritten text.",
        "desc": "Extracts handwritten text from images.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    },
    "Equation Extraction": {
        "prompt": "<image>\nExtract all mathematical equations in LaTeX format.",
        "desc": "Extracts equations as LaTeX strings.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    },
    "Table Extraction": {
        "prompt": "<image>\nExtract all tables as HTML.",
        "desc": "Extracts tables into structured HTML.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    },
    "Plain Text Extraction": {
        "prompt": "<image>\nFree OCR.",
        "desc": "Fast, plain text OCR of the image.",
        "base_size": 768, "image_size": 512, "crop_mode": False
    },
    "Multilingual Document": {
        "prompt": "<image>\n<|grounding|>Extract text. Preserve all languages and structure.",
        "desc": "Extracts multi-language content, preserves structure.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    },
    "Meme Text Extraction": {
        "prompt": "<image>\nExtract all text from this image.",
        "desc": "Extracts text from memes and social media images.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    },
    "Custom Prompt": {
        "prompt": "",
        "desc": "Provide your own custom prompt for flexible OCR or parsing.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    }
}

# Global model and tokenizer
model = None
tok = None


def load_model_once():
    """Load model once and cache it globally"""
    global model, tok

    if model is not None:
        return model, tok

    model_id = "deepseek-ai/DeepSeek-OCR"
    print(f"Loading {model_id}...")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager"
        ).to(dtype=torch.bfloat16, device="cuda").eval()
    else:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager"
        ).to(dtype=torch.float32, device="cpu").eval()
        print("⚠️  Running on CPU - this will be VERY slow!")

    print("✓ Model loaded successfully")
    return model, tok


def new_run_dir(base="./runs"):
    """Create unique output directory for each run"""
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    path = os.path.join(base, f"run_{ts}_{rid}")
    os.makedirs(path)
    return path


def gr_ocr(image, mode, custom_prompt, base_size, image_size, crop_mode):
    """Main OCR processing function for Gradio"""

    if image is None:
        return "[No image provided]", "⚠️ Please upload an image first", None

    # Ensure model is loaded
    try:
        mdl, tokenizer = load_model_once()
    except Exception as e:
        return f"[Error loading model: {e}]", f"❌ Model loading failed: {e}", None

    # Preprocess image
    img = image.convert("RGB")
    original_size = img.size

    if max(img.size) > 2000:
        s = 2000 / max(img.size)
        img = img.resize((int(img.width*s), int(img.height*s)), Image.LANCZOS)

    # Create output directory
    run_dir = new_run_dir()
    img_path_proc = os.path.join(run_dir, "input.png")
    img.save(img_path_proc, optimize=True)

    # Get prompt
    if mode == "Custom Prompt" and custom_prompt.strip():
        prompt = custom_prompt.strip()
    else:
        prompt = DEMO_MODES[mode]["prompt"]

    # Run inference
    t0 = time.time()
    try:
        with torch.inference_mode():
            _ = mdl.infer(
                tokenizer,
                prompt=prompt,
                image_file=img_path_proc,
                output_path=run_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=True
            )
    except ZeroDivisionError:
        print("⚠️  [Patched] Division by zero in compression ratio (valid_img_tokens==0). Ignored.")
    except Exception as e:
        return f"[Error during inference: {e}]", f"❌ Inference failed: {e}", None

    dt = time.time() - t0

    # Read results
    result_file = os.path.join(run_dir, "result.mmd")
    if not os.path.exists(result_file):
        result_file = os.path.join(run_dir, "result.txt")

    result = "[No text extracted]"
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            result = f.read().strip() or "[No text extracted]"

    # Load boxed image
    boxed_path = os.path.join(run_dir, "result_with_boxes.jpg")
    boxed_img = Image.open(boxed_path) if os.path.exists(boxed_path) else None

    # Stats
    stats = f"""
✓ **Processing complete in {dt:.1f}s**
📐 Original image: {original_size[0]}×{original_size[1]} px
📁 Output directory: `{run_dir}`
💾 Files saved: input.png, {os.path.basename(result_file)}, result_with_boxes.jpg
"""

    return result, stats, boxed_img


def update_mode(selected):
    """Update UI when mode changes"""
    d = DEMO_MODES[selected]
    return (
        d["desc"],
        gr.update(visible=selected=="Custom Prompt"),
        d["base_size"],
        d["image_size"],
        d["crop_mode"]
    )


def create_ui():
    """Create Gradio interface"""

    with gr.Blocks(theme=gr.themes.Soft(), title="DeepSeek-OCR Local") as demo:
        gr.Markdown("""
        <div style="text-align:center;">
            <h1>🔍 DeepSeek-OCR Local Testing</h1>
            <p>Upload an image to extract text, tables, charts, equations, and more</p>
        </div>
        """)

        with gr.Row():
            # Left Column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                image_input = gr.Image(type="pil", label="Upload Document/Image", height=350)

                gr.Markdown("#### 🎯 Select OCR Mode")
                mode = gr.Radio(
                    choices=list(DEMO_MODES.keys()),
                    value="Document → Markdown",
                    label="Mode"
                )
                desc = gr.Markdown(DEMO_MODES["Document → Markdown"]["desc"])

                custom_prompt = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Enter your custom prompt here...",
                    visible=False
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    base_size = gr.Slider(
                        512, 1280,
                        value=DEMO_MODES["Document → Markdown"]["base_size"],
                        step=64,
                        label="Base Size"
                    )
                    image_size = gr.Slider(
                        512, 1280,
                        value=DEMO_MODES["Document → Markdown"]["image_size"],
                        step=64,
                        label="Image Size"
                    )
                    crop_mode = gr.Checkbox(
                        value=DEMO_MODES["Document → Markdown"]["crop_mode"],
                        label="Dynamic Resolution (Crop Mode)"
                    )

                process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")

            # Right Column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Results")
                ocr_output = gr.Textbox(
                    label="Extracted Content",
                    lines=18,
                    show_copy_button=True,
                    placeholder="Processed content will appear here..."
                )
                status_out = gr.Markdown("_Upload an image and click Process to begin_")
                boxed_output = gr.Image(
                    label="Result with Bounding Boxes",
                    type="pil"
                )

        # Event handlers
        mode.change(
            update_mode,
            inputs=mode,
            outputs=[desc, custom_prompt, base_size, image_size, crop_mode]
        )

        process_btn.click(
            gr_ocr,
            inputs=[image_input, mode, custom_prompt, base_size, image_size, crop_mode],
            outputs=[ocr_output, status_out, boxed_output]
        )

        # Examples section
        gr.Markdown("""
        ### 💡 Tips
        - **Chart Deep Parsing**: Best for extracting data from charts and graphs
        - **Table Extraction**: Converts tables to HTML format
        - **Handwriting OCR**: Works on handwritten notes (quality matters!)
        - **Equation Extraction**: Returns LaTeX formatted math
        - **Multilingual**: Slower but handles mixed languages well
        - **GPU recommended**: CPU inference will be very slow (minutes vs seconds)
        """)

    return demo


def main():
    print("="*60)
    print("DeepSeek-OCR Local Gradio App")
    print("="*60)

    demo = create_ui()

    print("\n🚀 Launching Gradio interface...")
    print("💡 Model will load on first inference (be patient!)\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()
