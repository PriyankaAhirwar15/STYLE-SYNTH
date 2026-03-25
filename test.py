#!/usr/bin/env python3
"""
STYLE-SYNTH Quick Test
Simple test anyone can run to verify the system works
"""
import requests
import time
import sys
from pathlib import Path

def test_api_connection(port=8002):
    """Test if API is responding"""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_style_transfer(port=8002):
    """Test actual style transfer with a simple image"""
    try:
        # Create a simple test image (256x256 RGB)
        import numpy as np
        from PIL import Image
        import io

        # Create a gradient test image
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                img[i, j] = [i, j, (i+j)//2]  # Color gradient

        # Convert to JPEG
        pil_img = Image.fromarray(img)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()

        # Test transform endpoint
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        data = {"style": "sketch", "check_quality": "false"}

        response = requests.post(
            f"http://127.0.0.1:{port}/transform",
            files=files,
            data=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("success", False)
        return False

    except Exception as e:
        print(f"Test error: {e}")
        return False

def main():
    """Run quick tests"""
    print("🧪 STYLE-SYNTH Quick Test")
    print("=" * 30)

    port = 8002

    # Test 1: API Connection
    print("1. Testing API connection...")
    if test_api_connection(port):
        print("   ✅ API is responding")
    else:
        print("   ❌ API not responding")
        print("   💡 Make sure to run: python start.py")
        return 1

    # Test 2: Style Transfer
    print("2. Testing style transfer...")
    if test_style_transfer(port):
        print("   ✅ Style transfer working")
    else:
        print("   ❌ Style transfer failed")
        return 1

    # Test 3: Check available styles
    print("3. Testing styles endpoint...")
    try:
        response = requests.get(f"http://127.0.0.1:{port}/styles", timeout=5)
        if response.status_code == 200:
            styles_data = response.json()
            styles = [s["id"] for s in styles_data.get("styles", [])]
            print(f"   ✅ Available styles: {', '.join(styles)}")
        else:
            print("   ❌ Styles endpoint failed")
            return 1
    except:
        print("   ❌ Could not fetch styles")
        return 1

    print("\n🎉 All tests passed! STYLE-SYNTH is working perfectly!")
    print("\n🌐 Open your browser to: http://127.0.0.1:8501")
    print("📤 Upload any face photo and try the different styles!")

    return 0

if __name__ == "__main__":
    sys.exit(main())