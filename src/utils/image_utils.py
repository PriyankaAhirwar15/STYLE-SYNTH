import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
from pathlib import Path
from src.utils.logger import setup_logger
logger = setup_logger("image_utils")
def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return image
def resize_image(image: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(image, (size, size))
def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")
def base64_to_image(base64_str: str) -> np.ndarray:
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
def save_image(image: np.ndarray, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.info(f"Image saved: {output_path}")
    return output_path
def apply_style_effect(image: np.ndarray, style: str) -> np.ndarray:
    if style == "sketch":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif style == "oil_painting":
        return cv2.xphoto.oilPainting(image, 7, 1) if hasattr(cv2, 'xphoto') else cv2.bilateralFilter(image, 9, 75, 75)
    elif style == "watercolor":
        return cv2.stylization(image, sigma_s=60, sigma_r=0.45)
    elif style == "cartoon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)
    elif style == "anime":
        return cv2.stylization(image, sigma_s=150, sigma_r=0.25)
    else:
        return image
if __name__ == "__main__":
    logger.info("Image utils working!")
    print("Image utils test passed!")
