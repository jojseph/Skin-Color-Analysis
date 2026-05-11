"""
analyzer.py
Sends the face image + master prompt to Google Gemini and returns parsed JSON.
"""

import base64
import json
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"


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


def _parse_json(raw_text: str) -> dict:
    """Parse JSON from Gemini response, with fallback for markdown-fenced output."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse Gemini response as JSON.\nRaw output:\n{raw_text}")


def analyze(base64_image: str) -> dict:
    """
    Main entry point. Call this from app.py.

    Args:
        base64_image: Base64-encoded JPEG string from face_utils.process_uploaded_image()

    Returns:
        dict with keys: season, undertone, palette, outfit_colors,
                        avoid_colors, makeup_tips, summary

    Raises:
        ValueError       — if JSON cannot be parsed from the Gemini response
        EnvironmentError — if GEMINI_API_KEY is missing
        Exception        — if the API call fails (network error, rate limit, etc.)
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY not found. Check your .env file.")

    client = genai.Client()

    image_part = types.Part.from_bytes(
        data=base64.b64decode(base64_image),
        mime_type="image/jpeg",
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[image_part, PROMPT],
    )

    return _parse_json(response.text)
