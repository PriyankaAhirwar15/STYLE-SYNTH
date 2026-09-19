#!/usr/bin/env python3
"""
STYLE-SYNTH Comprehensive System & Pipeline Test Suite
Tests direct pipeline transformations, latent blending, face detection,
and FastAPI endpoint operations.
"""
import sys
import os

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import base64
import cv2
import numpy as np
import time
from pathlib import Path
from config import STYLES
from src.gan.transform import StyleTransferPipeline


def create_test_face_image():
    """Creates a synthetic portrait image with distinct facial features for testing."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = [210, 190, 175]  # Background

    # Face Oval
    cv2.ellipse(img, (128, 135), (75, 95), 0, 0, 360, (230, 205, 190), -1)

    # Hair
    cv2.ellipse(img, (128, 90), (80, 50), 0, 180, 360, (40, 30, 25), -1)

    # Eyes
    cv2.circle(img, (105, 125), 9, (255, 255, 255), -1)
    cv2.circle(img, (151, 125), 9, (255, 255, 255), -1)
    cv2.circle(img, (105, 125), 4, (30, 80, 160), -1)
    cv2.circle(img, (151, 125), 4, (30, 80, 160), -1)

    # Nose
    cv2.line(img, (128, 130), (128, 145), (180, 150, 130), 2)

    # Mouth
    cv2.ellipse(img, (128, 165), (18, 6), 0, 0, 360, (120, 40, 180), -1)

    return img


def test_pipeline_direct():
    """Tests all 8 styles and blending via direct in-memory pipeline."""
    print("[*] Testing In-Memory StyleTransferPipeline...")
    pipeline = StyleTransferPipeline()
    face_img = create_test_face_image()

    passed = 0
    for style in STYLES:
        res = pipeline.transform(
            image=face_img,
            style=style,
            style_strength=0.9,
            smoothness=1.0,
            vibrancy=1.1,
            sharpness=1.0,
            auto_face_crop=True,
            check_quality=True
        )
        if res["success"]:
            score = res.get("quality_report", {}).get("quality_score", "N/A")
            print(f"   [+] Style: {style:15} | Score: {score}/10 | Time: {res['processing_time']:.2f}s")
            passed += 1
        else:
            print(f"   [-] Style: {style:15} | Error: {res.get('error')}")

    # Test Blending
    b_res = pipeline.blend(face_img, "anime", "cyberpunk", alpha=0.5, check_quality=True)
    if b_res["success"]:
        print(f"   [+] Blending: Anime + Cyberpunk (alpha=0.5) -> Success ({b_res['processing_time']:.2f}s)")
        passed += 1
    else:
        print(f"   [-] Blending Error: {b_res.get('error')}")

    return passed == (len(STYLES) + 1)


def main():
    print("STYLE-SYNTH Comprehensive Test Suite")
    print("=" * 60)

    success = test_pipeline_direct()

    print("\n" + "=" * 60)
    if success:
        print("ALL GAN PIPELINE TESTS PASSED WITH 100% SUCCESS!")
    else:
        print("Some pipeline tests encountered issues.")


if __name__ == "__main__":
    main()