#!/usr/bin/env python3
"""
Quick installation test for DeepSeek-OCR
This script verifies that all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")

    packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "PIL": "Pillow (PIL)",
        "gradio": "Gradio",
        "einops": "einops",
        "addict": "addict",
        "easydict": "easydict"
    }

    failed = []

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            failed.append(name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages installed successfully")
        return True


def test_torch():
    """Test PyTorch installation and device availability"""
    print("\nTesting PyTorch...")

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"  ⚠️  CUDA not available - will run on CPU (very slow)")
            print(f"     This is OK for testing, but GPU is highly recommended")

        return True
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
        return False


def test_huggingface_auth():
    """Test Hugging Face authentication"""
    print("\nTesting Hugging Face authentication...")

    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()

        if token:
            print("  ✓ Hugging Face token found")
            print("  ✓ You can access gated models")
        else:
            print("  ⚠️  No Hugging Face token found")
            print("     Run: huggingface-cli login")
            print("     Get token from: https://huggingface.co/settings/tokens")
            return False

        return True
    except ImportError:
        print("  ⚠️  huggingface-hub not installed")
        print("     Run: pip install huggingface-hub")
        print("     Then: huggingface-cli login")
        return False
    except Exception as e:
        print(f"  ⚠️  Error checking auth: {e}")
        return False


def test_model_access():
    """Test if we can access the DeepSeek-OCR model"""
    print("\nTesting model access (this may take a moment)...")

    try:
        from transformers import AutoConfig
        model_id = "deepseek-ai/DeepSeek-OCR"

        # Try to load config (lightweight test)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"  ✓ Can access {model_id}")
        print(f"  ✓ Model config loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Cannot access model: {e}")
        print("\n  Possible issues:")
        print("    1. Not authenticated with Hugging Face (run: huggingface-cli login)")
        print("    2. No internet connection")
        print("    3. Model may require access request on Hugging Face")
        return False


def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")

    try:
        import shutil
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)

        print(f"  Available space: {free_gb:.1f} GB")

        if free_gb < 10:
            print(f"  ⚠️  Low disk space! Model needs ~6GB + space for outputs")
            print(f"     Recommended: 10GB+ free")
            return False
        else:
            print(f"  ✓ Sufficient disk space")
            return True
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True  # Don't fail on this


def main():
    print("="*60)
    print("DeepSeek-OCR Installation Test")
    print("="*60)
    print()

    results = []

    # Run all tests
    results.append(("Package imports", test_imports()))
    results.append(("PyTorch", test_torch()))
    results.append(("Disk space", check_disk_space()))
    results.append(("Hugging Face auth", test_huggingface_auth()))
    results.append(("Model access", test_model_access()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou're ready to use DeepSeek-OCR!")
        print("\nNext steps:")
        print("  1. Run the web interface: python ocr_gradio_app.py")
        print("  2. Or test a single image: python ocr_basic.py <image_path>")
    else:
        print("⚠️  Some tests failed")
        print("\nPlease fix the issues above before proceeding.")
        print("See QUICKSTART.md or README.md for help")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
