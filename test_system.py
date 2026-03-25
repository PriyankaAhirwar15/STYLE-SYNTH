#!/usr/bin/env python3
"""
STYLE-SYNTH Test Script
Tests the complete pipeline with a sample image
"""
import requests
import base64
import cv2
import numpy as np
import time
from pathlib import Path

def create_test_image():
    """Create a simple test face-like image"""
    # Create a 256x256 RGB image
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Add some basic face-like features
    # Background
    img[:, :] = [200, 180, 160]  # Skin tone background

    # Face oval
    cv2.ellipse(img, (128, 140), (80, 100), 0, 0, 360, (220, 190, 170), -1)

    # Eyes
    cv2.circle(img, (110, 120), 8, (255, 255, 255), -1)  # Left eye white
    cv2.circle(img, (146, 120), 8, (255, 255, 255), -1)  # Right eye white
    cv2.circle(img, (110, 120), 4, (0, 0, 0), -1)       # Left pupil
    cv2.circle(img, (146, 120), 4, (0, 0, 0), -1)       # Right pupil

    # Nose
    cv2.ellipse(img, (128, 140), (3, 8), 0, 0, 360, (200, 170, 150), -1)

    # Mouth
    cv2.ellipse(img, (128, 160), (15, 5), 0, 0, 360, (150, 50, 50), -1)

    return img

def test_transform_endpoint(api_url="http://127.0.0.1:8002", style="anime", check_quality=True):
    """Test the transform endpoint"""
    print(f"🧪 Testing transform endpoint with style: {style}")

    # Create test image
    test_img = create_test_image()

    # Convert to JPEG bytes
    success, img_encoded = cv2.imencode('.jpg', test_img)
    if not success:
        print("❌ Failed to encode test image")
        return False

    img_bytes = img_encoded.tobytes()

    # Prepare request
    files = {"file": ("test_face.jpg", img_bytes, "image/jpeg")}
    data = {"style": style, "check_quality": str(check_quality).lower()}

    try:
        start_time = time.time()
        response = requests.post(f"{api_url}/transform", files=files, data=data, timeout=60)
        end_time = time.time()

        print(f"📡 Request completed in {end_time - start_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print("✅ Transform successful!")

            # Print results
            print(f"🎨 Style: {result.get('style')}")
            print(f"⚡ Processing Time: {result.get('processing_time', 0):.2f}s")

            if "quality_report" in result and result["quality_report"]:
                quality = result["quality_report"]
                print(f"⭐ Quality Score: {quality.get('quality_score', 'N/A')}/10")
                print(f"🎯 Style Accuracy: {quality.get('style_accuracy', 'N/A')}")
                print(f"🤖 Assessment: {quality.get('assessment', 'N/A')[:100]}...")

            # Save the result
            if "styled_image" in result:
                styled_b64 = result["styled_image"]
                styled_data = base64.b64decode(styled_b64)
                output_path = f"test_output_{style}_{int(time.time())}.jpg"
                with open(output_path, "wb") as f:
                    f.write(styled_data)
                print(f"💾 Saved result to: {output_path}")

            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎭 STYLE-SYNTH Comprehensive Test Suite")
    print("=" * 50)

    api_url = "http://127.0.0.1:8002"

    # Test all styles
    styles = ["anime", "sketch", "oil_painting", "watercolor", "cartoon"]

    results = {}
    for style in styles:
        print(f"\n{'='*30} Testing {style.upper()} {'='*30}")
        success = test_transform_endpoint(api_url, style, check_quality=True)
        results[style] = success
        time.sleep(1)  # Brief pause between tests

    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")

    successful = sum(results.values())
    total = len(results)

    for style, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{style.title():15} | {status}")

    print(f"\n🎯 Overall: {successful}/{total} tests passed")

    if successful == total:
        print("🎉 All tests passed! STYLE-SYNTH is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the API server and try again.")

if __name__ == "__main__":
    main()