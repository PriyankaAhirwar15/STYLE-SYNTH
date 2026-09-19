"""
STYLE-SYNTH Face Detection & Smart Alignment
Provides robust face detection and bounding box extraction for optimal GAN portrait centering.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger("face_detect")


def detect_and_crop_face(
    image: np.ndarray,
    target_size: int = 512,
    margin_ratio: float = 0.4
) -> Tuple[np.ndarray, bool]:
    """
    Detects the primary face in the image and crops a square region centered around it
    with proportional margin for portrait GAN processing.

    Returns:
        cropped_image: np.ndarray of shape (target_size, target_size, 3)
        face_found: bool indicating whether an actual face was detected
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input for face detection.")

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load OpenCV Pre-trained Haar Cascade for frontal face
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        logger.info("No face detected with primary cascade, using smart center-square crop.")
        # Fallback to smart center square crop
        crop_dim = min(h, w)
        start_y = (h - crop_dim) // 2
        start_x = (w - crop_dim) // 2
        cropped = image[start_y:start_y + crop_dim, start_x:start_x + crop_dim]
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return resized, False

    # Pick the largest face detected
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    fx, fy, fw, fh = faces[0]

    # Calculate expanded square crop with margin
    cx = fx + fw // 2
    cy = fy + fh // 2
    side = int(max(fw, fh) * (1.0 + margin_ratio))

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    # Adjust x1, y1 if boundary constrained
    if (x2 - x1) < side and x1 > 0:
        x1 = max(0, x2 - side)
    if (y2 - y1) < side and y1 > 0:
        y1 = max(0, y2 - side)

    cropped = image[y1:y2, x1:x2]

    # Ensure square aspect ratio
    ch, cw = cropped.shape[:2]
    if ch != cw:
        cdim = min(ch, cw)
        cropped = cropped[:cdim, :cdim]

    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    logger.info(f"Face detected at ({fx}, {fy}, {fw}, {fh}), cropped to {target_size}x{target_size}")
    return resized, True
