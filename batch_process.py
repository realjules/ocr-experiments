#!/usr/bin/env python3
"""
Batch processing script for DeepSeek-OCR
Processes all images in a directory
"""

import os
import glob
import argparse
from ocr_basic import load_model, process_image


def batch_process(input_dir, output_base="results", mode="document", pattern="*.{jpg,jpeg,png,pdf}"):
    """Process all images in a directory"""

    # Load model once for all images
    print("Loading model...")
    model, tok = load_model()
    print("Model loaded!\n")

    # Find all images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.pdf", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process\n")

    # Create base output directory
    os.makedirs(output_base, exist_ok=True)

    # Process each image
    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing [{i}/{len(image_files)}]: {os.path.basename(img_path)}")
        print('='*60)

        # Create output directory for this image
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = os.path.join(output_base, img_name)

        try:
            extracted_text, out_dir = process_image(
                model,
                tok,
                img_path,
                mode=mode,
                output_dir=output_dir
            )

            results.append({
                "image": img_path,
                "status": "success",
                "output_dir": out_dir,
                "text_length": len(extracted_text)
            })

        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            results.append({
                "image": img_path,
                "status": "failed",
                "error": str(e)
            })

    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)

    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count

    print(f"\nTotal images: {len(results)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")

    if success_count > 0:
        print(f"\n📁 Results saved to: {output_base}/")

    # Save summary
    summary_file = os.path.join(output_base, "batch_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {fail_count}\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"Individual Results:\n")
        f.write(f"{'='*60}\n\n")

        for r in results:
            f.write(f"File: {os.path.basename(r['image'])}\n")
            f.write(f"Status: {r['status']}\n")
            if r['status'] == 'success':
                f.write(f"Output: {r['output_dir']}\n")
                f.write(f"Text length: {r['text_length']} chars\n")
            else:
                f.write(f"Error: {r.get('error', 'Unknown')}\n")
            f.write("\n")

    print(f"📄 Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch process images with DeepSeek-OCR")
    parser.add_argument("input_dir", help="Directory containing images to process")
    parser.add_argument("--output", "-o", default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--mode", "-m", default="document",
                       choices=["document", "chart", "chemistry", "handwriting",
                               "equation", "table", "multilingual", "plain"],
                       help="OCR mode to use (default: document)")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory")
        return

    batch_process(args.input_dir, args.output, args.mode)


if __name__ == "__main__":
    main()
