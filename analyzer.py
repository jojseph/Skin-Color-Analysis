"""
analyzer.py
Local Moondream2 + LoRA inference for ASTA 4-season color analysis.

Updated design:
    cropped face image -> VLM -> {"season":"Autumn"}
    analyzer.py -> deterministic generalized season recommendation template -> Streamlit result

The model is responsible only for 4-class season classification.
The full palette/recommendation output is intentionally handled here for consistency
and easier evaluation.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Compatibility patch for Moondream/Phi + PEFT on some Transformers versions.
# Some PhiConfig objects do not expose pad_token_id, but PEFT expects it.
try:
    from transformers.models.phi.configuration_phi import PhiConfig
    if not hasattr(PhiConfig, "pad_token_id"):
        PhiConfig.pad_token_id = None
except Exception:
    pass

try:
    import streamlit as st
except Exception:  # Allows command-line/unit tests without Streamlit installed.
    st = None

MODEL_ID = os.getenv("ASTA_BASE_MODEL", "vikhyatk/moondream2")
REVISION = os.getenv("ASTA_MODEL_REVISION", "2024-08-26")
ADAPTER_DIR = Path(os.getenv("ASTA_ADAPTER_DIR", "moondream_season_lora"))

PROMPT = """
Classify this face into exactly one of these 4 seasonal color classes:

Autumn, Spring, Summer, Winter.

Return ONLY this JSON format:
{"season":"Winter"}

No explanation. No markdown. No extra words.
""".strip()


SEASON_TEMPLATES = {
    "Autumn": {
        "season": "Autumn",
        "subtype": None,
        "full_season": "Autumn",
        "undertone": "warm",
        "characteristics": "Warm, earthy, rich, grounded, and muted-to-deep color harmony.",
        "palette": ["#6B3F1D", "#8B4513", "#A65F2B", "#B8860B", "#556B2F", "#7A4E2D"],
        "outfit_colors": [
            {"name": "Rust", "hex": "#B7410E"},
            {"name": "Olive Green", "hex": "#556B2F"},
            {"name": "Camel", "hex": "#C19A6B"},
            {"name": "Chocolate Brown", "hex": "#381819"},
            {"name": "Mustard", "hex": "#D4A017"},
            {"name": "Terracotta", "hex": "#E2725B"},
        ],
        "avoid_colors": [
            {"name": "Icy Blue", "hex": "#D6F0FF"},
            {"name": "Cool Silver", "hex": "#C0C0C0"},
            {"name": "Neon Pink", "hex": "#FF1493"},
            {"name": "Pure White", "hex": "#FFFFFF"},
        ],
        "makeup_tips": {
            "foundation": "Warm, golden, yellow-based, or neutral-warm undertones. Natural, satin, or soft matte finish.",
            "blush": "Peach, apricot, terracotta, warm rose, cinnamon, or muted coral.",
            "lips": "Terracotta, brick red, burnt orange, warm nude, copper rose, or brown-red.",
            "eyes": "Bronze, copper, olive, warm brown, caramel, espresso, and muted gold.",
        },
        "summary": "Your features align with Autumn, which generally suits warm, earthy, rich, and natural shades. Avoid icy, neon, and overly cool colors because they can overpower or dull Autumn harmony.",
    },
    "Spring": {
        "season": "Spring",
        "subtype": None,
        "full_season": "Spring",
        "undertone": "warm",
        "characteristics": "Warm, fresh, clear, bright, and lively color harmony.",
        "palette": ["#FF7F50", "#FFD700", "#FFA500", "#98FB98", "#40E0D0", "#FFB347"],
        "outfit_colors": [
            {"name": "Coral", "hex": "#FF7F50"},
            {"name": "Golden Yellow", "hex": "#FFD700"},
            {"name": "Fresh Green", "hex": "#32CD32"},
            {"name": "Peach", "hex": "#FFDAB9"},
            {"name": "Turquoise", "hex": "#40E0D0"},
            {"name": "Warm Aqua", "hex": "#66D9D9"},
        ],
        "avoid_colors": [
            {"name": "Charcoal Gray", "hex": "#36454F"},
            {"name": "Black", "hex": "#000000"},
            {"name": "Dusty Mauve", "hex": "#B08D9B"},
            {"name": "Muddy Olive", "hex": "#6B6B2E"},
        ],
        "makeup_tips": {
            "foundation": "Warm, golden, peachy, or neutral-warm undertones. Fresh, radiant, or dewy finish.",
            "blush": "Peach, coral-pink, apricot, warm pink, or soft watermelon.",
            "lips": "Coral, peach-red, warm pink, watermelon, sheer orange-red, or fresh rose.",
            "eyes": "Champagne, light gold, warm brown, caramel, peach beige, and soft teal accents.",
        },
        "summary": "Your features align with Spring, which generally suits warm, clear, bright, and fresh colors. Avoid heavy, dusty, dark, or muddy shades because they can make Spring features look flat.",
    },
    "Summer": {
        "season": "Summer",
        "subtype": None,
        "full_season": "Summer",
        "undertone": "cool",
        "characteristics": "Cool, soft, gentle, muted, and refined color harmony.",
        "palette": ["#B0C4DE", "#A7C7E7", "#D8BFD8", "#C8A2C8", "#B8B8B8", "#DCAE96"],
        "outfit_colors": [
            {"name": "Dusty Rose", "hex": "#DCAE96"},
            {"name": "Powder Blue", "hex": "#B0E0E6"},
            {"name": "Mauve", "hex": "#B784A7"},
            {"name": "Soft Lavender", "hex": "#E6E6FA"},
            {"name": "Slate Blue", "hex": "#6A5ACD"},
            {"name": "Cool Gray", "hex": "#B8B8B8"},
        ],
        "avoid_colors": [
            {"name": "Orange", "hex": "#FFA500"},
            {"name": "Mustard", "hex": "#D4A017"},
            {"name": "Neon Green", "hex": "#39FF14"},
            {"name": "Stark Black", "hex": "#000000"},
        ],
        "makeup_tips": {
            "foundation": "Cool, pink-based, or neutral-cool undertones. Soft matte, satin, or natural finish.",
            "blush": "Cool pink, dusty rose, mauve, soft berry, muted raspberry, or gentle rose.",
            "lips": "Rose pink, mauve, muted berry, cool nude, soft plum, or raspberry tint.",
            "eyes": "Taupe, cool gray, slate, soft navy, lavender, ash brown, and muted charcoal.",
        },
        "summary": "Your features align with Summer, which generally suits cool, soft, muted, and gentle shades. Avoid harsh black, neon colors, and strong warm oranges because they can overwhelm Summer harmony.",
    },
    "Winter": {
        "season": "Winter",
        "subtype": None,
        "full_season": "Winter",
        "undertone": "cool",
        "characteristics": "Cool, clear, high-contrast, saturated, and sharp color harmony.",
        "palette": ["#000000", "#FFFFFF", "#0047AB", "#DC143C", "#4B0082", "#C0C0C0"],
        "outfit_colors": [
            {"name": "Black", "hex": "#000000"},
            {"name": "Pure White", "hex": "#FFFFFF"},
            {"name": "Cobalt Blue", "hex": "#0047AB"},
            {"name": "True Red", "hex": "#FF0000"},
            {"name": "Emerald", "hex": "#006B54"},
            {"name": "Royal Purple", "hex": "#4B0082"},
        ],
        "avoid_colors": [
            {"name": "Camel", "hex": "#C19A6B"},
            {"name": "Muted Beige", "hex": "#D8C3A5"},
            {"name": "Mustard", "hex": "#D4A017"},
            {"name": "Dusty Brown", "hex": "#8B7D6B"},
        ],
        "makeup_tips": {
            "foundation": "Cool, neutral, neutral-olive, or pink-based undertones. Polished, satin, or matte finish.",
            "blush": "Cool rose, berry, plum, icy pink, wine rose, or clear red-pink.",
            "lips": "Blue-red, berry, burgundy, cool plum, crimson, or vivid pink.",
            "eyes": "Black liner, charcoal, navy, silver, icy shimmer, deep espresso, and jewel tones.",
        },
        "summary": "Your features align with Winter, which generally suits cool, saturated, crisp, and high-contrast shades. Avoid warm muddy earth tones because they can make Winter features look dull.",
    },
}


VALID_SEASONS = {"Spring", "Summer", "Autumn", "Winter"}

_model = None
_tokenizer = None
_adapter_mode = None


def _normalize_word(value: Any, valid_values: set[str]) -> str | None:
    if value is None:
        return None

    value = str(value).strip().lower()
    for valid in valid_values:
        if value == valid.lower():
            return valid
    return None


def _extract_season(raw_text: str) -> str | None:
    """
    Handles outputs such as:
        {"season":"Autumn"}
        Autumn
        season: Autumn
        ```json ... ```
    """
    if not raw_text:
        return None

    cleaned = str(raw_text).strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()

    # 1) Exact JSON.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return _normalize_word(parsed.get("season"), VALID_SEASONS)
    except json.JSONDecodeError:
        pass

    # 2) JSON embedded inside extra text.
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return _normalize_word(parsed.get("season"), VALID_SEASONS)
        except json.JSONDecodeError:
            pass

    # 3) Plain text extraction.
    for valid_season in sorted(VALID_SEASONS):
        if re.search(rf"\b{valid_season}\b", cleaned, re.IGNORECASE):
            return valid_season

    return None


def _template_from_prediction(season: str) -> dict:
    return copy.deepcopy(SEASON_TEMPLATES[season])


def _parse_model_response(raw_text: str) -> dict:
    season = _extract_season(raw_text)

    if season is None:
        raise ValueError(f"Could not determine season from model response. Raw output: {raw_text!r}")

    result = _template_from_prediction(season)
    result["raw_model_response"] = raw_text
    return result


def _load_base_model() -> tuple[Any, Any, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    return model, tokenizer, device


def _attach_lora_adapter(model: Any) -> tuple[Any, str]:
    """
    Attach the LoRA adapter to Moondream's text_model.

    Important:
    The Colab notebook saved the adapter from model.text_model, not from the
    full HfMoondream wrapper. Loading it into the full HfMoondream object can
    trigger errors such as:
        'HfMoondream' object has no attribute 'all_tied_weights_keys'

    So this app intentionally loads the adapter into model.text_model only.
    """
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Fine-tuned LoRA adapter folder not found: {ADAPTER_DIR}. "
            "Place the trained 4-season adapter folder in the project root, or set ASTA_ADAPTER_DIR."
        )

    if not hasattr(model, "text_model"):
        raise RuntimeError(
            "Loaded Moondream model does not expose text_model, so the LoRA "
            "adapter cannot be attached. Check MODEL_ID and REVISION."
        )

    # Patch again immediately before PEFT wraps text_model.
    dummy_tokenizer_pad = getattr(getattr(model, "text_model", None), "config", None)
    if dummy_tokenizer_pad is not None and not hasattr(dummy_tokenizer_pad, "pad_token_id"):
        object.__setattr__(dummy_tokenizer_pad, "pad_token_id", getattr(dummy_tokenizer_pad, "eos_token_id", 0))

    model.text_model = PeftModel.from_pretrained(model.text_model, str(ADAPTER_DIR))

    # Patch the wrapped PEFT model too.
    if hasattr(model.text_model, "config") and not hasattr(model.text_model.config, "pad_token_id"):
        object.__setattr__(model.text_model.config, "pad_token_id", getattr(model.text_model.config, "eos_token_id", 0))

    return model, "text_model"


def _safe_set_config_token_ids(config: Any, pad_token_id: int | None, eos_token_id: int | None) -> None:
    """
    Force token IDs onto config objects.

    Some Moondream/Phi config objects do not define pad_token_id as a normal
    attribute, and PEFT accesses it directly. Using object.__setattr__ makes
    the attribute exist even when the config class does not originally expose it.
    """
    if config is None:
        return

    if pad_token_id is not None:
        try:
            object.__setattr__(config, "pad_token_id", pad_token_id)
        except Exception:
            try:
                config.pad_token_id = pad_token_id
            except Exception:
                pass

    if eos_token_id is not None:
        try:
            object.__setattr__(config, "eos_token_id", eos_token_id)
        except Exception:
            try:
                config.eos_token_id = eos_token_id
            except Exception:
                pass


def _patch_missing_config_values(model: Any, tokenizer: Any) -> None:
    """
    Patch pad/eos token ids into every relevant Moondream/Phi config before PEFT
    loads the LoRA adapter.

    This fixes errors like:
        'PhiConfig' object has no attribute 'pad_token_id'
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None:
        pad_token_id = eos_token_id

    # Top-level wrapper config.
    _safe_set_config_token_ids(getattr(model, "config", None), pad_token_id, eos_token_id)

    # Main text model config.
    text_model = getattr(model, "text_model", None)
    _safe_set_config_token_ids(getattr(text_model, "config", None), pad_token_id, eos_token_id)

    # Common nested PEFT / HF model configs.
    nested_candidates = [
        getattr(text_model, "model", None),
        getattr(text_model, "base_model", None),
        getattr(getattr(text_model, "model", None), "model", None),
        getattr(getattr(text_model, "base_model", None), "model", None),
    ]

    for candidate in nested_candidates:
        _safe_set_config_token_ids(getattr(candidate, "config", None), pad_token_id, eos_token_id)

    # Generation configs may also be checked by some wrappers.
    _safe_set_config_token_ids(getattr(model, "generation_config", None), pad_token_id, eos_token_id)
    _safe_set_config_token_ids(getattr(text_model, "generation_config", None), pad_token_id, eos_token_id)

    # Last-resort: recursively patch config-like attributes one level deep.
    for obj in [model, text_model] + [x for x in nested_candidates if x is not None]:
        for attr in ("config", "generation_config"):
            try:
                _safe_set_config_token_ids(getattr(obj, attr, None), pad_token_id, eos_token_id)
            except Exception:
                pass


def _load_model_uncached() -> tuple[Any, Any, str]:
    model, tokenizer, device = _load_base_model()
    _patch_missing_config_values(model, tokenizer)
    model, adapter_mode = _attach_lora_adapter(model)
    model.to(device)
    model.eval()
    return model, tokenizer, adapter_mode


if st is not None:
    _cached_load_model = st.cache_resource(show_spinner="Loading ASTA VLM model...")(_load_model_uncached)
else:
    _cached_load_model = _load_model_uncached


def _load_model() -> tuple[Any, Any, str]:
    global _model, _tokenizer, _adapter_mode

    if _model is not None and _tokenizer is not None and _adapter_mode is not None:
        return _model, _tokenizer, _adapter_mode

    _model, _tokenizer, _adapter_mode = _cached_load_model()
    return _model, _tokenizer, _adapter_mode


def _run_moondream_inference(model: Any, tokenizer: Any, image: Image.Image) -> str:
    with torch.inference_mode():
        image_embeds = model.encode_image(image)

        try:
            response = model.answer_question(image_embeds, PROMPT, tokenizer)
        except TypeError:
            # Compatibility with Moondream revisions that expect the raw PIL image.
            response = model.answer_question(image, PROMPT, tokenizer)

    return str(response).strip()


def analyze(base64_image: str) -> dict:
    """
    Main entry point called by the Streamlit app.

    Args:
        base64_image: Base64-encoded JPEG string from face_utils.process_pil_image().

    Returns:
        Full generalized seasonal color result dictionary expected by the UI.
    """
    model, tokenizer, adapter_mode = _load_model()

    image_bytes = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    raw_response = _run_moondream_inference(model, tokenizer, image)
    result = _parse_model_response(raw_response)
    result["adapter_mode"] = adapter_mode
    return result


def analyze_pil_image(image: Image.Image) -> dict:
    """Convenience helper for tests and notebooks."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return analyze(encoded)
