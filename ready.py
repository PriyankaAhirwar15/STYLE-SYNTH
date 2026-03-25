#!/usr/bin/env python3
"""
STYLE-SYNTH Final Readiness Test
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")

    required_modules = [
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('streamlit', 'Streamlit'),
        ('groq', 'Groq'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests')
    ]

    all_good = True
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            all_good = False

    return all_good

def test_core_functionality():
    """Test core style transfer functionality"""
    print("🎨 Testing core functionality...")

    try:
        from src.gan.transform import StyleTransferPipeline
        import numpy as np

        # Create test pipeline
        pipeline = StyleTransferPipeline()
        print("   ✅ Pipeline initialized")

        # Create test image
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Test transform
        result = pipeline.transform(test_img, 'sketch', check_quality=False)
        if result['success']:
            print("   ✅ Style transfer working")
            print(f"   📊 Processing time: {result['processing_time']:.2f}s")
            return True
        else:
            print(f"   ❌ Transform failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"   ❌ Core functionality error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("📁 Testing file structure...")

    required_files = [
        'config.py',
        'api/main.py',
        'frontend/app.py',
        'src/gan/transform.py',
        'src/auditor/llm_checker.py',
        'src/utils/image_utils.py',
        'src/utils/logger.py',
        'requirements.txt',
        'README.md',
        '.env'
    ]

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
            all_exist = False

    return all_exist

def main():
    """Run all tests"""
    print("🎭 STYLE-SYNTH FINAL READINESS TEST")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Python Imports", test_imports),
        ("Core Functionality", test_core_functionality)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        results[test_name] = test_func()

    # Summary
    print(f"\n{'='*50}")
    print("📊 FINAL RESULTS")
    print(f"{'='*50}")

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} | {status}")
        if success:
            passed += 1

    print(f"\n🎯 Overall Score: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 CONGRATULATIONS! STYLE-SYNTH IS READY!")
        print("\n🚀 To start your application:")
        print("   python start.py")
        print("   # or on Windows: start.bat")
        print("\n🌐 Then open: http://127.0.0.1:8501")
        print("\n📤 Upload face photos and enjoy AI-powered style transfer!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())