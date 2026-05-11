# Skin Tone Seasonal Analyzer — Project Brief

> **VLM:** Google Gemini API (free) or LLaVA via Ollama (local)
> **Frontend:** Python + Streamlit
> **Timeline:** 1 day (~12 working hours)
> **Team:** 3 members
> **Outputs:** Season label · Color palette · Outfit suggestions · Makeup tips

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [App Architecture & Pipeline](#3-app-architecture--pipeline)
4. [File Structure](#4-file-structure)
5. [Team Task Split](#5-team-task-split)
6. [1-Day Schedule](#6-1-day-schedule)
7. [Setup Instructions](#7-setup-instructions)
8. [Master VLM Prompt](#8-master-vlm-prompt)
9. [Code Skeletons](#9-code-skeletons)
10. [Streamlit UI Checklist](#10-streamlit-ui-checklist)
11. [Edge Cases to Test](#11-edge-cases-to-test)
12. [GitHub & README Checklist](#12-github--readme-checklist)

---

## 1. Project Overview

This app takes a portrait photo, detects the face, and sends it to a Vision Language Model (VLM) to determine the user's **seasonal colour type** (Spring, Summer, Autumn, Winter). The model then returns a full set of personalised recommendations:

- **Season label** — Spring, Summer, Autumn, or Winter
- **Color palette** — 6 flattering hex colors for that season
- **Outfit colors** — clothing tones that complement the skin tone
- **Makeup tips** — foundation undertone, blush, lips, and eye looks

The VLM does all classification and recommendation work in a single prompt. No model training is required — only prompt engineering and pipeline integration.

> **Why a VLM instead of a traditional CV pipeline?**
> A VLM understands colour context, undertones, and seasonal colour theory in one shot.
> A traditional approach would require separate face detection, skin pixel extraction, colour
> clustering, and a rules-based season classifier — each with its own failure mode.

---

## 2. Technology Stack

### VLM API Options — Pick ONE before Hour 1

| | Option A — Google Gemini | Option B — LLaVA (Ollama) |
|---|---|---|
| **Model** | `gemini-2.5-flash` | `llava:13b` or `llava:7b` |
| **Cost** | Free tier — no credit card needed | Completely free — runs locally |
| **Setup** | API key from [ai.google.dev](https://ai.google.dev) | Install Ollama + `ollama pull llava` |
| **Python SDK** | `pip install google-generativeai` | `pip install ollama` |
| **GPU needed?** | No — runs in the cloud | Yes — min. 8 GB VRAM |
| **Speed** | Fast (~2–5 sec per request) | Slower (~10–30 sec locally) |
| **Output quality** | Excellent for colour reasoning | Good; may need prompt tuning |
| **Recommendation** | ✅ Best for most teams | Use if no API key available |

### Other Dependencies

```
streamlit
mediapipe
opencv-python
Pillow
python-dotenv
google-generativeai   # Gemini only
ollama                # LLaVA only
```

Install all at once:

```bash
pip install streamlit mediapipe opencv-python Pillow python-dotenv google-generativeai
```

---

## 3. App Architecture & Pipeline

```
User uploads photo
       │
       ▼
face_utils.py — MediaPipe face detection + crop (512×512) + Base64 encode
       │
       ▼
analyzer.py — VLM API call (Gemini or LLaVA) with master prompt
       │
       ▼
JSON response parsed → { season, undertone, palette, outfit_colors,
                          avoid_colors, makeup_tips, summary }
       │
       ▼
app.py — Streamlit renders season banner, colour swatches, outfit card, makeup tips
```

| Step | What happens | File |
|------|-------------|------|
| 1 | User uploads portrait photo | `app.py` |
| 2 | MediaPipe detects face bounding box; region cropped + resized to 512×512 | `face_utils.py` |
| 3 | Cropped face Base64-encoded for API | `face_utils.py` |
| 4 | Image + prompt sent to Gemini or LLaVA; JSON returned | `analyzer.py` |
| 5 | JSON parsed; season banner, swatches, cards rendered | `app.py` |

---

## 4. File Structure

```
skin-tone-analyzer/
├── app.py              # Streamlit UI — upload, loading, all result sections
├── analyzer.py         # VLM API call + prompt + JSON parsing
├── face_utils.py       # MediaPipe face crop + Base64 encode
├── requirements.txt    # All Python dependencies
├── .env                # GEMINI_API_KEY (never commit this)
├── .gitignore          # Excludes .env, __pycache__, model files
├── samples/            # 8–12 test portraits covering all 4 seasons
└── README.md           # Setup guide + screenshots + demo link
```

---

## 5. Team Task Split

### Person 1 — CV / Face Detection → `face_utils.py`

- Set up MediaPipe face mesh / face detection
- Crop the face bounding box with a small padding margin
- Resize the crop to 512×512
- Convert the image to a Base64 string
- Handle edge cases: no face detected, multiple faces (use the largest/most centred)

**Deliverable:** `face_utils.py` tested on all images in `samples/`

---

### Person 2 — VLM / Prompt Engineering → `analyzer.py`

- Set up Gemini or LLaVA API with the chosen credentials
- Implement the master prompt (Section 8)
- Parse the JSON response; add regex fallback for malformed output
- Validate results across `samples/` — at least one image per season
- Tune the prompt if any season is consistently misclassified

**Deliverable:** `analyzer.py` returning correct JSON for all 4 seasons

---

### Person 3 — Streamlit UI → `app.py`

- Build the image uploader + loading spinner
- Season label banner styled per season (different colour per season)
- Hex colour swatches rendered as inline HTML blocks
- Outfit suggestions grid
- Makeup tips in an expandable `st.expander` section
- Clear error messages for no-face and API-error states

**Deliverable:** `app.py` with a complete, working end-to-end demo

---

> ⚠️ **Integration checkpoint at Hour 6** — all three members merge and run one
> full test: upload → face crop → API call → display results. Fix blocking issues
> before continuing individual polish work.

---

## 6. 1-Day Schedule

| Hours | Block | Tasks |
|-------|-------|-------|
| 1–2 | Environment setup | Create GitHub repo, virtual environment, install deps. Obtain and test API key (Gemini) OR install Ollama and pull LLaVA. Verify a basic API call works with a test image before splitting into tracks. |
| 3–5 | Core development | All three members work in parallel on their tracks (Section 5). Person 1: face detection. Person 2: VLM prompt + parsing. Person 3: Streamlit skeleton. |
| 6 | Integration checkpoint | Merge branches. Run one full end-to-end test. Fix blocking issues together. |
| 7–9 | Feature completion | Person 1: edge case testing. Person 2: season validation + prompt tuning. Person 3: complete all UI panels + error states. |
| 10–11 | Polish & testing | Run app on 10+ diverse sample images. Fix layout/parsing bugs. Add sidebar instructions. Confirm `.env` is gitignored. |
| 12 | Docs & submission | Write README, take screenshots, record 60–90 sec demo video, push final code, tag release. |

---

## 7. Setup Instructions

### Option A — Google Gemini (Recommended)

1. Go to [https://ai.google.dev](https://ai.google.dev) and sign in with a Google account
2. Click **Get API Key → Create API Key** and copy it
3. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   ```
4. Add `.env` to `.gitignore` immediately
5. Quick test before integrating:
   ```python
   import google.generativeai as genai, os
   from dotenv import load_dotenv
   load_dotenv()
   genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
   model = genai.GenerativeModel("gemini-2.5-flash")
   print(model.generate_content("Say hello.").text)
   ```

### Option B — LLaVA via Ollama (Local / Free)

1. Install Ollama: [https://ollama.com/download](https://ollama.com/download)
2. Pull the model (pick one based on your VRAM):
   ```bash
   ollama pull llava:13b    # Best quality — needs ~8 GB VRAM
   ollama pull llava:7b     # Lighter option — needs ~5 GB VRAM
   ```
3. The Ollama server starts automatically; if not: `ollama serve`
4. Install the Python client: `pip install ollama`
5. Quick test:
   ```python
   import ollama
   r = ollama.chat(model="llava:13b", messages=[{"role": "user", "content": "Say hello."}])
   print(r["message"]["content"])
   ```

---

## 8. Master VLM Prompt

Copy this exactly into `analyzer.py` as `PROMPT`. Only modify it during the validation phase (Hours 7–9) if outputs are consistently wrong.

```python
PROMPT = """
You are an expert colour analyst specialising in seasonal skin tone theory.
Analyse the face in this image and determine the person's seasonal colour type.
Focus on: skin undertone (warm/cool/neutral), skin depth (light/medium/deep),
hair colour contrast, and overall colour harmony.

Return ONLY a valid JSON object with this exact structure — no markdown, no code
fences, no explanation text before or after:

{
  "season": "Spring" | "Summer" | "Autumn" | "Winter",
  "undertone": "warm" | "cool" | "neutral",
  "palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5", "#hex6"],
  "outfit_colors": ["color name 1", "color name 2", "color name 3", "color name 4"],
  "avoid_colors": ["color name 1", "color name 2"],
  "makeup_tips": {
    "foundation": "description of ideal foundation undertone and finish",
    "blush": "description of ideal blush shades",
    "lips": "description of ideal lip colours",
    "eyes": "description of ideal eye makeup look"
  },
  "summary": "2-3 sentence explanation of why this season was chosen"
}
"""
```

---

## 9. Code Skeletons

### `face_utils.py`

```python
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
```

---

### `analyzer.py`

```python
"""
analyzer.py
Person 2 owns this file.
Sends the face image + master prompt to the chosen VLM and returns parsed JSON.
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────────
# Set this to "gemini" or "llava" before running.
VLM_PROVIDER = "gemini"   # <-- CHANGE THIS to "llava" if using Ollama

GEMINI_MODEL = "gemini-2.5-flash"
LLAVA_MODEL  = "llava:13b"   # or "llava:7b" for the lighter version


# ── Master prompt ───────────────────────────────────────────────────────────────
PROMPT = """
You are an expert colour analyst specialising in seasonal skin tone theory.
Analyse the face in this image and determine the person's seasonal colour type.
Focus on: skin undertone (warm/cool/neutral), skin depth (light/medium/deep),
hair colour contrast, and overall colour harmony.

Return ONLY a valid JSON object with this exact structure — no markdown, no code
fences, no explanation text before or after:

{
  "season": "Spring" | "Summer" | "Autumn" | "Winter",
  "undertone": "warm" | "cool" | "neutral",
  "palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5", "#hex6"],
  "outfit_colors": ["color name 1", "color name 2", "color name 3", "color name 4"],
  "avoid_colors": ["color name 1", "color name 2"],
  "makeup_tips": {
    "foundation": "description of ideal foundation undertone and finish",
    "blush": "description of ideal blush shades",
    "lips": "description of ideal lip colours",
    "eyes": "description of ideal eye makeup look"
  },
  "summary": "2-3 sentence explanation of why this season was chosen"
}
"""


# ── JSON parser (with fallback) ────────────────────────────────────────────────
def _parse_json(raw_text: str) -> dict:
    """
    Parse JSON from a VLM response string.
    Falls back to regex extraction if the model adds extra text around the JSON.
    """
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # LLaVA sometimes wraps JSON in markdown fences — strip them
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: extract the first {...} block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse VLM response as JSON.\nRaw output:\n{raw_text}")


# ── Gemini implementation ───────────────────────────────────────────────────────
def _analyze_gemini(base64_image: str) -> dict:
    """Send image to Google Gemini and return parsed result dict."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found. Check your .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    image_part = {
        "mime_type": "image/jpeg",
        "data": base64_image,
    }

    response = model.generate_content([PROMPT, image_part])
    return _parse_json(response.text)


# ── LLaVA / Ollama implementation ──────────────────────────────────────────────
def _analyze_llava(base64_image: str) -> dict:
    """Send image to local LLaVA model via Ollama and return parsed result dict."""
    import ollama

    response = ollama.chat(
        model=LLAVA_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [base64_image],
        }]
    )
    raw_text = response["message"]["content"]
    return _parse_json(raw_text)


# ── Public API ─────────────────────────────────────────────────────────────────
def analyze(base64_image: str) -> dict:
    """
    Main entry point. Call this from app.py.

    Args:
        base64_image: Base64-encoded JPEG string from face_utils.process_uploaded_image()

    Returns:
        dict with keys: season, undertone, palette, outfit_colors,
                        avoid_colors, makeup_tips, summary

    Raises:
        ValueError  — if JSON cannot be parsed from the VLM response
        EnvironmentError — if the API key is missing (Gemini only)
        Exception   — if the API call fails (network error, rate limit, etc.)
    """
    if VLM_PROVIDER == "gemini":
        return _analyze_gemini(base64_image)
    elif VLM_PROVIDER == "llava":
        return _analyze_llava(base64_image)
    else:
        raise ValueError(f"Unknown VLM_PROVIDER: '{VLM_PROVIDER}'. Use 'gemini' or 'llava'.")
```

---

### `app.py`

```python
"""
app.py
Person 3 owns this file.
Streamlit UI — handles upload, loading state, and all result display sections.
"""

import streamlit as st
from PIL import Image

from face_utils import process_uploaded_image
from analyzer import analyze


# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Tone Analyzer",
    page_icon="🎨",
    layout="centered",
)


# ── Season theme config ─────────────────────────────────────────────────────────
SEASON_THEMES = {
    "Spring": {
        "emoji": "🌸",
        "description": "Warm, light, and bright — you have a delicate, fresh quality.",
        "bg_color": "#FFF0F5",
        "text_color": "#A0522D",
    },
    "Summer": {
        "emoji": "☀️",
        "description": "Cool, muted, and soft — you suit powdery, dusty tones.",
        "bg_color": "#F0F4FF",
        "text_color": "#4A5A8A",
    },
    "Autumn": {
        "emoji": "🍂",
        "description": "Warm, rich, and deep — earthy and golden tones are your friend.",
        "bg_color": "#FFF5EC",
        "text_color": "#8B4513",
    },
    "Winter": {
        "emoji": "❄️",
        "description": "Cool, clear, and high-contrast — bold and icy colours suit you best.",
        "bg_color": "#F0F8FF",
        "text_color": "#1A3A6A",
    },
}


# ── Helper: render hex colour swatches ─────────────────────────────────────────
def render_swatches(hex_colors: list[str]):
    """Render a row of colour swatches using inline HTML."""
    swatches_html = "".join([
        f"""<div style="
            display:inline-block;
            width:60px; height:60px;
            background:{hex};
            border-radius:8px;
            margin:4px;
            border:1px solid #ddd;
            " title="{hex}"></div>"""
        for hex in hex_colors
    ])
    labels_html = "".join([
        f'<div style="display:inline-block;width:60px;text-align:center;'
        f'font-size:11px;color:#666;margin:0 4px;">{hex}</div>'
        for hex in hex_colors
    ])
    st.markdown(
        f'<div style="margin:8px 0">{swatches_html}</div>'
        f'<div style="margin:0 0 12px 0">{labels_html}</div>',
        unsafe_allow_html=True,
    )


# ── Helper: render colour chip tags ─────────────────────────────────────────────
def render_color_chips(color_names: list[str], style: str = "normal"):
    """Render colour names as pill-shaped chips. style='avoid' renders in red."""
    bg = "#FFEBEE" if style == "avoid" else "#F0F4F8"
    text = "#B71C1C" if style == "avoid" else "#1A2A3A"
    border = "#FFCDD2" if style == "avoid" else "#CBD5E0"
    chips_html = " ".join([
        f'<span style="background:{bg};color:{text};border:1px solid {border};'
        f'padding:4px 12px;border-radius:20px;font-size:13px;margin:3px;'
        f'display:inline-block;">{c}</span>'
        for c in color_names
    ])
    st.markdown(f'<div style="margin:8px 0">{chips_html}</div>', unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ How to use")
    st.markdown("""
    1. Upload a **clear portrait photo** — face fully visible, good lighting.
    2. Wait a few seconds for the AI analysis.
    3. See your **seasonal colour type** and personalised recommendations.

    **Tips for best results:**
    - Use a photo with a plain or blurred background
    - Remove glasses if possible
    - Avoid heavy filters or extreme lighting
    """)
    st.divider()
    st.caption("Powered by a Vision Language Model (VLM)")


# ── Main UI ─────────────────────────────────────────────────────────────────────
st.title("🎨 Skin Tone Seasonal Analyzer")
st.markdown("Upload a portrait photo to discover your seasonal colour type and get personalised recommendations.")

uploaded_file = st.file_uploader(
    "Choose a portrait photo",
    type=["jpg", "jpeg", "png"],
    help="A clear, well-lit portrait works best."
)

if uploaded_file is not None:

    # Show the uploaded photo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)

    st.divider()

    # ── Run analysis ────────────────────────────────────────────────────────────
    with st.spinner("Detecting face and analysing your skin tone..."):
        try:
            # Step 1: detect face + encode
            base64_img, face_crop = process_uploaded_image(uploaded_file)

            if base64_img is None:
                st.error(
                    "**No face detected.** Please upload a clear portrait photo "
                    "where your face is fully visible and well-lit."
                )
                st.stop()

            # Step 2: VLM analysis
            result = analyze(base64_img)

        except EnvironmentError as e:
            st.error(f"**API key error:** {e}")
            st.stop()
        except ValueError as e:
            st.error(f"**Could not parse the model response.** Try uploading a clearer photo. ({e})")
            st.stop()
        except Exception as e:
            st.error(f"**Something went wrong:** {e}")
            st.stop()

    # ── Results ─────────────────────────────────────────────────────────────────
    season = result.get("season", "Unknown")
    theme  = SEASON_THEMES.get(season, SEASON_THEMES["Spring"])

    # Season banner
    st.markdown(
        f"""<div style="
            background:{theme['bg_color']};
            border-left:5px solid {theme['text_color']};
            padding:16px 20px;
            border-radius:8px;
            margin:12px 0;
        ">
            <h2 style="color:{theme['text_color']};margin:0">
                {theme['emoji']} {season}
            </h2>
            <p style="color:{theme['text_color']};margin:6px 0 0;opacity:0.85">
                {theme['description']}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Undertone tag
    undertone = result.get("undertone", "").capitalize()
    if undertone:
        st.markdown(f"**Undertone:** `{undertone}`")

    # Summary
    summary = result.get("summary", "")
    if summary:
        st.info(summary)

    st.divider()

    # Colour palette
    st.subheader("🎨 Your Colour Palette")
    palette = result.get("palette", [])
    if palette:
        render_swatches(palette)
    else:
        st.write("No palette returned.")

    st.divider()

    # Outfit suggestions + avoid colours side by side
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("👗 Wear These")
        outfit_colors = result.get("outfit_colors", [])
        if outfit_colors:
            render_color_chips(outfit_colors, style="normal")
        else:
            st.write("No outfit colours returned.")

    with col_b:
        st.subheader("🚫 Avoid These")
        avoid_colors = result.get("avoid_colors", [])
        if avoid_colors:
            render_color_chips(avoid_colors, style="avoid")
        else:
            st.write("No avoid colours returned.")

    st.divider()

    # Makeup tips
    st.subheader("💄 Makeup Tips")
    makeup = result.get("makeup_tips", {})

    with st.expander("Foundation", expanded=True):
        st.write(makeup.get("foundation", "—"))
    with st.expander("Blush"):
        st.write(makeup.get("blush", "—"))
    with st.expander("Lips"):
        st.write(makeup.get("lips", "—"))
    with st.expander("Eyes"):
        st.write(makeup.get("eyes", "—"))

    st.divider()
    st.caption("Results are AI-generated and meant as inspiration, not professional colour analysis.")
```

---

### `requirements.txt`

```
streamlit
mediapipe
opencv-python
Pillow
python-dotenv
google-generativeai
ollama
```

---

### `.env.example`

```
# Copy this file to .env and fill in your key.
# NEVER commit .env to Git.
GEMINI_API_KEY=your_gemini_api_key_here
```

---

### `.gitignore`

```
.env
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/
venv/
*.model
*.gguf
```

---

## 10. Streamlit UI Checklist

- [ ] Page title + sidebar how-to instructions
- [ ] Image file uploader (`jpg`, `jpeg`, `png`)
- [ ] Uploaded photo preview
- [ ] Loading spinner wrapping the entire analysis block
- [ ] Season banner with per-season background and text colour
- [ ] Undertone label displayed near the season
- [ ] Summary paragraph in an `st.info()` block
- [ ] 6 hex colour swatches rendered as coloured HTML divs
- [ ] Outfit colour chips (normal style)
- [ ] Avoid colour chips (red style)
- [ ] Makeup tips in `st.expander` sections (Foundation, Blush, Lips, Eyes)
- [ ] Error state: no face detected
- [ ] Error state: API / environment key error
- [ ] Error state: JSON parse failure

---

## 11. Edge Cases to Test

- **No face detected** — blurry photo, extreme angle, object instead of face
- **Multiple faces** — should crop the largest/most centred face only
- **Accessories** — glasses, heavy makeup, strong shadows, face filters
- **Lighting extremes** — overexposed or very dark photos
- **Diverse skin tones** — at least one sample image per season
- **LLaVA only** — model wraps JSON in markdown fences → regex fallback must handle this
- **API rate limit (Gemini free tier)** — add a small `time.sleep(1)` between rapid test calls during dev

---

## 12. GitHub & README Checklist

- [ ] Repository is public
- [ ] `README.md` includes: project description, tech stack, setup steps, how to run
- [ ] At least **two screenshots** of the app showing different seasons
- [ ] `requirements.txt` is present and complete
- [ ] `.env` is in `.gitignore` — confirmed NOT committed
- [ ] No API keys appear anywhere in the committed code
- [ ] A short **demo video** (60–90 seconds) is linked or uploaded
- [ ] Each function has at least a one-line docstring

---

> **Good luck — ship it and be proud of it.**
