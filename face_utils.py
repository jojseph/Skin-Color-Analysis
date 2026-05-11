"""
face_utils.py
Person 1 owns this file.
Handles face detection, cropping, and Base64 encoding using MediaPipe.
"""

import base64
import io
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_SIZE = 512        # Resize crop to this before encoding
FACE_PADDING = 0.25      # Extra margin around the bounding box (fraction of face size)


# ── MediaPipe setup ─────────────────────────────────────────────────────────────
_mp_face = mp.solutions.face_detection
_detector = _mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def detect_and_crop_face(pil_image: Image.Image) -> Image.Image | None:
    """
    Detect the largest face in a PIL image and return a padded crop.
    Returns None if no face is detected.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    results = _detector.process(img_rgb)

    if not results.detections:
        return None  # TODO: Person 1 — handle no-face edge case in app.py

    # Pick the detection with the highest confidence score
    best = max(results.detections, key=lambda d: d.score[0])
    bbox = best.location_data.relative_bounding_box

    h, w = img_rgb.shape[:2]

    # Convert relative bbox to pixel coords with padding
    pad_x = bbox.width  * FACE_PADDING
    pad_y = bbox.height * FACE_PADDING

    x1 = max(0, int((bbox.xmin - pad_x) * w))
    y1 = max(0, int((bbox.ymin - pad_y) * h))
    x2 = min(w, int((bbox.xmin + bbox.width  + pad_x) * w))
    y2 = min(h, int((bbox.ymin + bbox.height + pad_y) * h))

    crop = img_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return crop_pil


def encode_image_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to a Base64-encoded JPEG string."""
    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_uploaded_image(uploaded_file) -> tuple[str, Image.Image] | tuple[None, None]:
    """
    Full pipeline: uploaded Streamlit file → (base64_string, cropped_face_image).
    Returns (None, None) if no face is detected.

    Usage in app.py:
        b64, face_img = process_uploaded_image(uploaded_file)
        if b64 is None:
            st.error("No face detected. Please upload a clear portrait photo.")
    """
    pil_image = Image.open(uploaded_file)
    face_crop = detect_and_crop_face(pil_image)

    if face_crop is None:
        return None, None

    b64 = encode_image_to_base64(face_crop)
    return b64, face_crop
