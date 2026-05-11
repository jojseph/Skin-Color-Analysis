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
  "outfit_colors": [{"name": "color name 1", "hex": "#hexcode"}, {"name": "color name 2", "hex": "#hexcode"}],
  "avoid_colors": [{"name": "color name 1", "hex": "#hexcode"}, {"name": "color name 2", "hex": "#hexcode"}],
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
