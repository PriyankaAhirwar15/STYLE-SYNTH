#!/usr/bin/env python3
"""
STYLE-SYNTH Readiness & Health Verification Suite
Verifies imports, file structure, PyTorch GAN modules, all 8 styles, and blending.
"""
import sys
import os

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
import numpy as np

def test_imports():
    """Test all required imports"""
    print("[*] Testing imports...")
    required_modules = [
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('streamlit', 'Streamlit'),
        ('groq', 'Groq'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests'),
        ('torch', 'PyTorch')
    ]

    all_good = True
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"   [+] {name}")
        except ImportError as e:
            print(f"   [-] {name}: {e}")
            all_good = False

    return all_good

def test_gan_architecture():
    """Test PyTorch GAN generator and discriminator modules"""
    print("[*] Testing PyTorch GAN Architecture...")
    try:
        import torch
        from src.gan.generator import ResNetGenerator, PatchGANDiscriminator

        gen = ResNetGenerator()
        disc = PatchGANDiscriminator()

        dummy_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out_gen = gen(dummy_tensor)
            out_disc = disc(out_gen)
            adv_score = disc.compute_adversarial_score(out_gen)

        summary = gen.get_summary()
        print(f"   [+] ResNet Generator ({summary['total_parameters']:,} params, {summary['residual_blocks']} ResBlocks)")
        print(f"   [+] PatchGAN Discriminator ({sum(p.numel() for p in disc.parameters()):,} params)")
        print(f"   [+] Adversarial scoring functional: {adv_score:.3f}")
        return True
    except Exception as e:
        print(f"   [-] GAN architecture error: {e}")
        return False

def test_all_8_styles():
    """Test all 8 GAN style engines"""
    print("[*] Testing 8 GAN Style Engines...")
    try:
        from src.gan.transform import StyleTransferPipeline
        from config import STYLES

        pipeline = StyleTransferPipeline()
        test_img = np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8)

        all_passed = True
        for style in STYLES:
            res = pipeline.transform(test_img, style, check_quality=False)
            if res["success"]:
                print(f"   [+] {style.upper():12} -> Success ({res['processing_time']:.2f}s)")
            else:
                print(f"   [-] {style.upper():12} -> Failed: {res.get('error')}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"   [-] Style engines test error: {e}")
        return False

def test_latent_blending():
    """Test multi-style latent interpolation"""
    print("[*] Testing Latent Style Blending...")
    try:
        from src.gan.transform import StyleTransferPipeline
        pipeline = StyleTransferPipeline()
        test_img = np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8)

        blend_res = pipeline.blend(test_img, "anime", "cyberpunk", alpha=0.5, check_quality=False)
        if blend_res["success"]:
            print(f"   [+] 50% Anime + 50% Cyberpunk blend generated in {blend_res['processing_time']:.2f}s")
            return True
        else:
            print(f"   [-] Blending failed: {blend_res.get('error')}")
            return False
    except Exception as e:
        print(f"   [-] Blending test error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("[*] Testing file structure...")
    required_files = [
        'config.py',
        'api/main.py',
        'frontend/app.py',
        'src/gan/generator.py',
        'src/gan/neural_styles.py',
        'src/gan/interpolator.py',
        'src/gan/face_detect.py',
        'src/gan/transform.py',
        'src/auditor/llm_checker.py',
        'src/utils/image_utils.py',
        'src/utils/logger.py',
        'requirements.txt',
        'README.md'
    ]

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   [+] {file_path}")
        else:
            print(f"   [-] {file_path} (missing)")
            all_exist = False

    return all_exist

def main():
    print("STYLE-SYNTH 2.0 READINESS & INTEGRITY TEST")
    print("=" * 55)

    tests = [
        ("File Structure", test_file_structure),
        ("Python Imports", test_imports),
        ("GAN Architecture", test_gan_architecture),
        ("All 8 GAN Styles", test_all_8_styles),
        ("Latent Blending", test_latent_blending)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n[*] {test_name}")
        print("-" * 35)
        results[test_name] = test_func()

    print(f"\n{'='*55}")
    print("FINAL RESULTS")
    print(f"{'='*55}")

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "[PASSED]" if success else "[FAILED]"
        print(f"{test_name:25} | {status}")
        if success:
            passed += 1

    print(f"\nOverall Score: {passed}/{total} tests passed")

    if passed == total:
        print("\nCONGRATULATIONS! STYLE-SYNTH 2.0 IS 100% READY WITH GANs!")
        print("Ready for local execution, Docker, and Hugging Face Spaces deployment.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())