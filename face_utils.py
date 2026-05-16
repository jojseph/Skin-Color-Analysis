"""
face_utils.py
Handles face detection, cropping, and Base64 encoding using OpenCV.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image


TARGET_SIZE = 512
FACE_PADDING = 0.25

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def detect_and_crop_face(pil_image: Image.Image) -> Image.Image | None:
    """Detect the largest face in a PIL image and return a padded crop, or None."""
    img_rgb = np.array(pil_image.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    #faces = _face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces = _face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
    )

    if len(faces) == 0:
        faces = _profile_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
        )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    height, width = img_rgb.shape[:2]
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    crop = img_rgb[y1:y2, x1:x2]
    return Image.fromarray(crop).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)


def encode_image_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to a Base64-encoded JPEG string."""
    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_uploaded_image(uploaded_file) -> tuple[str, Image.Image] | tuple[None, None]:
    """
    Full pipeline: uploaded Streamlit file → (base64_string, cropped_face_image).
    Returns (None, None) if no face is detected.
    """
    return process_pil_image(Image.open(uploaded_file))


def process_pil_image(pil_image: Image.Image) -> tuple[str, Image.Image] | tuple[None, None]:
    """
    Full pipeline: PIL image → (base64_string, cropped_face_image).
    Returns (None, None) if no face is detected.
    """
    face_crop = detect_and_crop_face(pil_image)

    if face_crop is None:
        return None, None

    return encode_image_to_base64(face_crop), face_crop
