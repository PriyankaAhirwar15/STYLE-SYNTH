"""
STYLE-SYNTH Latent Space Interpolation & Style Blender
Enables continuous multi-style synthesis, style cross-fading,
and facial attribute modulation (Smoothness, Vibrancy, Sharpness, Intensity).
"""
import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional
from src.gan.neural_styles import get_style_processor
from src.utils.logger import setup_logger

logger = setup_logger("interpolator")


def blend_two_styles(
    image: np.ndarray,
    style1: str,
    style2: str,
    alpha: float = 0.5,
    style_strength: float = 1.0,
    smoothness: float = 0.0,
    vibrancy: float = 1.0,
    sharpness: float = 0.0
) -> np.ndarray:
    """
    Synthesizes a blended artistic style:
    Blended = (1 - alpha) * Style1(img) + alpha * Style2(img)
    and applies facial attribute modulations.

    Args:
        image: Original input face portrait (BGR)
        style1: Primary style ID
        style2: Secondary style ID
        alpha: Blend ratio between 0.0 (100% Style1) and 1.0 (100% Style2)
        style_strength: Overall blend with original input (0.0 = original, 1.0 = full style)
        smoothness: Skin-smoothing intensity factor [0.0 - 10.0]
        vibrancy: Color saturation multiplier [0.5 - 2.0]
        sharpness: Unsharp mask high-pass emphasis [0.0 - 10.0]

    Returns:
        np.ndarray: Blended and modulated stylized image (BGR)
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    style_strength = np.clip(style_strength, 0.0, 1.0)

    # 1. Process style 1
    proc1 = get_style_processor(style1)
    img_s1 = proc1(image.copy())

    # 2. Process style 2
    proc2 = get_style_processor(style2)
    img_s2 = proc2(image.copy())

    # 3. Latent feature blend
    blended = cv2.addWeighted(img_s1, (1.0 - alpha), img_s2, alpha, 0.0)

    # 4. Modulate Face Attributes
    modulated = blended.copy()

    # Attribute A: Smoothness (Bilateral filter)
    if smoothness > 0.0:
        d = int(3 + smoothness * 1.5)
        sigma = int(20 + smoothness * 12)
        modulated = cv2.bilateralFilter(modulated, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    # Attribute B: Vibrancy (HSV Saturation adjust)
    if abs(vibrancy - 1.0) > 0.01:
        hsv = cv2.cvtColor(modulated, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * vibrancy, 0, 255)
        modulated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Attribute C: Sharpness (Unsharp Mask)
    if sharpness > 0.0:
        gaussian_blur = cv2.GaussianBlur(modulated, (0, 0), 2.0)
        unsharp_weight = 1.0 + (sharpness * 0.15)
        blur_weight = -(sharpness * 0.15)
        modulated = cv2.addWeighted(modulated, unsharp_weight, gaussian_blur, blur_weight, 0)
        modulated = np.clip(modulated, 0, 255).astype(np.uint8)

    # 5. Composite with original image according to style_strength
    if style_strength < 0.99:
        final_img = cv2.addWeighted(image, (1.0 - style_strength), modulated, style_strength, 0.0)
    else:
        final_img = modulated

    return final_img
