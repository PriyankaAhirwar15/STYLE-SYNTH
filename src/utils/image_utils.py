"""
STYLE-SYNTH Image Utilities
Provides robust image loading, encoding, conversion, and artistic helpers.
"""
import cv2
import numpy as np
from PIL import Image
import base64
import io
from pathlib import Path
from typing import Union
from src.utils.logger import setup_logger

logger = setup_logger("image_utils")


def load_image(image_path: str) -> np.ndarray:
    """Loads an image from disk in BGR format."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image from: {image_path}")
    return image


def resize_image(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Resizes image to square target size."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def image_to_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """Encodes BGR numpy array into base64 string."""
    _, buffer = cv2.imencode(format, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(base64_str: str) -> np.ndarray:
    """Decodes base64 string into BGR numpy array."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 into valid image array.")
    return img


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts PIL Image (RGB/RGBA) to OpenCV format (BGR)."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Converts OpenCV image (BGR) to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def save_image(image: np.ndarray, output_path: str) -> str:
    """Saves image to disk, ensuring directory existence."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.info(f"Image saved: {output_path}")
    return output_path


def apply_style_effect(image: np.ndarray, style: str) -> np.ndarray:
    """Dispatches to neural style engine."""
    from src.gan.neural_styles import get_style_processor
    processor = get_style_processor(style)
    return processor(image)
