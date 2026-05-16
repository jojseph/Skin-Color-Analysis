"""
analyzer.py
Local Moondream2 + LoRA inference for ASTA seasonal color analysis.

Current design:
    cropped face image -> VLM -> {"season":"Autumn","subtype":"Deep"}
    analyzer.py -> deterministic recommendation template -> Streamlit result

The model is responsible only for classification. The full palette/recommendation
output is intentionally handled here for consistency and easier evaluation.
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
Classify this face into exactly one of these 12 seasonal color classes:

Autumn Deep, Autumn Soft, Autumn Warm,
Spring Bright, Spring Light, Spring Warm,
Summer Cool, Summer Light, Summer Soft,
Winter Bright, Winter Cool, Winter Deep.

Return ONLY this JSON format:
{"season":"Winter","subtype":"Deep"}

No explanation. No markdown. No extra words.
"""


SEASON_SUBTYPE_TEMPLATES = {
    # ── AUTUMN ────────────────────────────────────────────────────────────────
    "Autumn Deep": {
        "season": "Autumn",
        "subtype": "Deep",
        "full_season": "Autumn Deep",
        "undertone": "warm",
        "palette": ["#4A2511", "#783F04", "#B45F06", "#38761D", "#274E13", "#7F6000"],
        "outfit_colors": [
            {"name": "Dark Tomato", "hex": "#BF360C"},
            {"name": "Olive Green", "hex": "#556B2F"},
            {"name": "Chocolate Brown", "hex": "#381819"},
            {"name": "Deep Rust", "hex": "#8B2500"},
        ],
        "avoid_colors": [
            {"name": "Icy Blue", "hex": "#E0FFFF"},
            {"name": "Lavender", "hex": "#E6E6FA"},
            {"name": "Cool Silver", "hex": "#C0C0C0"},
            {"name": "Neon Pink", "hex": "#FF1493"},
        ],
        "makeup_tips": {
            "foundation": "Warm, golden, or yellow undertones. Matte or natural finish.",
            "blush": "Terracotta, deep peach, brick red, or warm cinnamon.",
            "lips": "Warm reds, burnt orange, deep brick, or brown-toned nudes.",
            "eyes": "Bronze, copper, espresso brown, and deep forest green.",
        },
        "summary": "Your features align with Autumn Deep, which suits rich, warm, earthy, and high-depth shades.",
    },
    "Autumn Soft": {
        "season": "Autumn",
        "subtype": "Soft",
        "full_season": "Autumn Soft",
        "undertone": "warm",
        "palette": ["#8B7D6B", "#CD853F", "#D2B48C", "#556B2F", "#8FBC8F", "#BC8F8F"],
        "outfit_colors": [
            {"name": "Muted Olive", "hex": "#6B8E23"},
            {"name": "Warm Taupe", "hex": "#8B8589"},
            {"name": "Dusty Coral", "hex": "#D08C7F"},
            {"name": "Soft Camel", "hex": "#C19A6B"},
        ],
        "avoid_colors": [
            {"name": "Neon Pink", "hex": "#FF1493"},
            {"name": "Stark Black", "hex": "#000000"},
            {"name": "Icy White", "hex": "#F8FFFF"},
            {"name": "Electric Blue", "hex": "#0000FF"},
        ],
        "makeup_tips": {
            "foundation": "Warm to neutral undertones. Satin or natural finish.",
            "blush": "Soft peach, dusty coral, muted apricot, or warm rose.",
            "lips": "Muted salmon, soft brownish-pink, warm nude, or terracotta rose.",
            "eyes": "Khaki, taupe, soft moss green, and muted bronze.",
        },
        "summary": "Your features align with Autumn Soft, which suits warm, muted, earthy, and blended shades.",
    },
    "Autumn Warm": {
        "season": "Autumn",
        "subtype": "Warm",
        "full_season": "Autumn Warm",
        "undertone": "warm",
        "palette": ["#8B4513", "#D2691E", "#B8860B", "#DAA520", "#556B2F", "#8B0000"],
        "outfit_colors": [
            {"name": "Rust", "hex": "#8B4500"},
            {"name": "Mustard Yellow", "hex": "#FFDB58"},
            {"name": "Pumpkin", "hex": "#FF7518"},
            {"name": "Warm Olive", "hex": "#808000"},
        ],
        "avoid_colors": [
            {"name": "Fuchsia", "hex": "#FF00FF"},
            {"name": "Baby Blue", "hex": "#89CFF0"},
            {"name": "Cool Gray", "hex": "#808080"},
            {"name": "Icy Pink", "hex": "#F8C8DC"},
        ],
        "makeup_tips": {
            "foundation": "Distinctly warm, yellow-based, or golden. Fresh luminous finish.",
            "blush": "Rich apricot, warm brick, copper peach, or burnt coral.",
            "lips": "Copper, brick red, warm terracotta, or orange-red.",
            "eyes": "Gold, warm browns, olive, copper, and bronze.",
        },
        "summary": "Your features align with Autumn Warm, which suits rich, spicy, golden, and earthy shades.",
    },

    # ── SUMMER ────────────────────────────────────────────────────────────────
    "Summer Cool": {
        "season": "Summer",
        "subtype": "Cool",
        "full_season": "Summer Cool",
        "undertone": "cool",
        "palette": ["#4682B4", "#6495ED", "#7B68EE", "#DDA0DD", "#FFB6C1", "#708090"],
        "outfit_colors": [
            {"name": "Cornflower Blue", "hex": "#6495ED"},
            {"name": "Soft Plum", "hex": "#DDA0DD"},
            {"name": "Rose Pink", "hex": "#FFB6C1"},
            {"name": "Slate Blue", "hex": "#6A5ACD"},
        ],
        "avoid_colors": [
            {"name": "Goldenrod", "hex": "#DAA520"},
            {"name": "Warm Rust", "hex": "#8B4500"},
            {"name": "Orange", "hex": "#FFA500"},
            {"name": "Mustard", "hex": "#FFDB58"},
        ],
        "makeup_tips": {
            "foundation": "Distinctly cool, pink-based, or neutral-cool. Matte to satin finish.",
            "blush": "Cool pink, soft berry, rose, or muted raspberry.",
            "lips": "Rose pink, soft raspberry, muted berry, or cool mauve.",
            "eyes": "Slate grey, cool blue, silver, taupe, and soft charcoal.",
        },
        "summary": "Your features align with Summer Cool, which suits purely cool, soft, and refreshing shades.",
    },
    "Summer Light": {
        "season": "Summer",
        "subtype": "Light",
        "full_season": "Summer Light",
        "undertone": "cool",
        "palette": ["#E0FFFF", "#ADD8E6", "#E6E6FA", "#FFF0F5", "#F0F8FF", "#B0E0E6"],
        "outfit_colors": [
            {"name": "Powder Blue", "hex": "#B0E0E6"},
            {"name": "Soft Lavender", "hex": "#E6E6FA"},
            {"name": "Pale Pink", "hex": "#FADADD"},
            {"name": "Icy Mint", "hex": "#D8FFF0"},
        ],
        "avoid_colors": [
            {"name": "Dark Brown", "hex": "#654321"},
            {"name": "Burnt Orange", "hex": "#CC5500"},
            {"name": "Black", "hex": "#000000"},
            {"name": "Deep Burgundy", "hex": "#800020"},
        ],
        "makeup_tips": {
            "foundation": "Light, cool, or neutral undertones. Sheer or soft natural finish.",
            "blush": "Pale pink, light watermelon, soft rose, or cool peach-pink.",
            "lips": "Sheer pink, light rose gloss, soft berry tint, or cool nude.",
            "eyes": "Soft ash brown, pale grey, icy blue, lavender, and soft taupe.",
        },
        "summary": "Your features align with Summer Light, which suits delicate, airy, cool, and pastel shades.",
    },
    "Summer Soft": {
        "season": "Summer",
        "subtype": "Soft",
        "full_season": "Summer Soft",
        "undertone": "cool",
        "palette": ["#B0C4DE", "#D8BFD8", "#C0C0C0", "#778899", "#E6E6FA", "#FFE4E1"],
        "outfit_colors": [
            {"name": "Dusty Rose", "hex": "#DCAE96"},
            {"name": "Slate Blue", "hex": "#6A5ACD"},
            {"name": "Mauve", "hex": "#B784A7"},
            {"name": "Soft Gray", "hex": "#B8B8B8"},
        ],
        "avoid_colors": [
            {"name": "Neon Orange", "hex": "#FF4500"},
            {"name": "Chartreuse", "hex": "#7FFF00"},
            {"name": "Pure Black", "hex": "#000000"},
            {"name": "Bright Yellow", "hex": "#FFFF00"},
        ],
        "makeup_tips": {
            "foundation": "Cool, pink, or neutral undertones. Satin or soft natural finish.",
            "blush": "Soft mauve, dusty rose, muted pink, or gentle berry.",
            "lips": "Muted berry, plum, soft cool pink, or mauve rose.",
            "eyes": "Charcoal, taupe, cool grey, muted navy, and soft lavender.",
        },
        "summary": "Your features align with Summer Soft, which suits cool, muted, dusty, and gentle shades.",
    },

    # ── WINTER ────────────────────────────────────────────────────────────────
    "Winter Bright": {
        "season": "Winter",
        "subtype": "Bright",
        "full_season": "Winter Bright",
        "undertone": "cool",
        "palette": ["#FF007F", "#0000FF", "#39FF14", "#FF1493", "#00FA9A", "#FFFFFF"],
        "outfit_colors": [
            {"name": "Hot Pink", "hex": "#FF69B4"},
            {"name": "Cobalt Blue", "hex": "#0047AB"},
            {"name": "Pure White", "hex": "#FFFFFF"},
            {"name": "Clear Red", "hex": "#FF0000"},
        ],
        "avoid_colors": [
            {"name": "Dusty Brown", "hex": "#8B7D6B"},
            {"name": "Muted Beige", "hex": "#F5F5DC"},
            {"name": "Warm Camel", "hex": "#C19A6B"},
            {"name": "Muddy Olive", "hex": "#6B6B2E"},
        ],
        "makeup_tips": {
            "foundation": "Neutral to cool undertones. Clear, luminous, polished finish.",
            "blush": "Vibrant pink, cool berry, clear rose, or bright fuchsia.",
            "lips": "Ruby red, electric pink, blue-red, or bright berry.",
            "eyes": "High contrast liner, icy shimmer, black, silver, and cobalt accents.",
        },
        "summary": "Your features align with Winter Bright, which suits clear, high-contrast, vivid, and cool shades.",
    },
    "Winter Cool": {
        "season": "Winter",
        "subtype": "Cool",
        "full_season": "Winter Cool",
        "undertone": "cool",
        "palette": ["#0000CD", "#DC143C", "#8A2BE2", "#FF00FF", "#00FFFF", "#C0C0C0"],
        "outfit_colors": [
            {"name": "Royal Blue", "hex": "#4169E1"},
            {"name": "True Red", "hex": "#FF0000"},
            {"name": "Icy Silver", "hex": "#C0C0C0"},
            {"name": "Cool Violet", "hex": "#8A2BE2"},
        ],
        "avoid_colors": [
            {"name": "Terracotta", "hex": "#E2725B"},
            {"name": "Olive Green", "hex": "#808000"},
            {"name": "Mustard", "hex": "#FFDB58"},
            {"name": "Warm Beige", "hex": "#D8C3A5"},
        ],
        "makeup_tips": {
            "foundation": "Strictly cool, pink, blue-based, or neutral-cool. Flawless finish.",
            "blush": "Bright fuchsia, cool magenta, icy pink, or blue-based rose.",
            "lips": "True blue-red, vivid pink, berry red, or cool plum.",
            "eyes": "Silver, icy white, deep navy, charcoal, and black liner.",
        },
        "summary": "Your features align with Winter Cool, which suits icy, vivid, purely cool, and saturated shades.",
    },
    "Winter Deep": {
        "season": "Winter",
        "subtype": "Deep",
        "full_season": "Winter Deep",
        "undertone": "cool",
        "palette": ["#000000", "#191970", "#4B0082", "#800000", "#006400", "#483D8B"],
        "outfit_colors": [
            {"name": "Midnight Blue", "hex": "#191970"},
            {"name": "Burgundy", "hex": "#800000"},
            {"name": "Black", "hex": "#000000"},
            {"name": "Deep Emerald", "hex": "#006400"},
        ],
        "avoid_colors": [
            {"name": "Warm Peach", "hex": "#FFDAB9"},
            {"name": "Mustard", "hex": "#FFDB58"},
            {"name": "Light Beige", "hex": "#F5F5DC"},
            {"name": "Soft Apricot", "hex": "#FBCEB1"},
        ],
        "makeup_tips": {
            "foundation": "Cool, neutral, or neutral-olive undertones. Matte or polished finish.",
            "blush": "Deep plum, rich berry, wine rose, or cool red.",
            "lips": "Deep burgundy, classic crimson, wine, or dark berry.",
            "eyes": "Black eyeliner, deep charcoal, navy, espresso, and jewel-toned shadow.",
        },
        "summary": "Your features align with Winter Deep, which suits dark, cool, bold, and intense shades.",
    },

    # ── SPRING ────────────────────────────────────────────────────────────────
    "Spring Bright": {
        "season": "Spring",
        "subtype": "Bright",
        "full_season": "Spring Bright",
        "undertone": "warm",
        "palette": ["#FF003F", "#00CED1", "#7CFC00", "#FF1493", "#FFFF00", "#FF4500"],
        "outfit_colors": [
            {"name": "Turquoise", "hex": "#40E0D0"},
            {"name": "Clear Red", "hex": "#FF0000"},
            {"name": "Bright Coral", "hex": "#FF7F50"},
            {"name": "Clear Yellow", "hex": "#FFFF00"},
        ],
        "avoid_colors": [
            {"name": "Muted Olive", "hex": "#808000"},
            {"name": "Dusty Blue", "hex": "#5F9EA0"},
            {"name": "Charcoal Gray", "hex": "#36454F"},
            {"name": "Muddy Brown", "hex": "#70543E"},
        ],
        "makeup_tips": {
            "foundation": "Neutral to warm undertones. Clear and luminous finish.",
            "blush": "Vibrant coral-pink, clear peach, watermelon, or bright apricot.",
            "lips": "Watermelon, bright poppy red, clear coral, or warm pink.",
            "eyes": "Clear gold, bright turquoise, warm brown liner, and fresh shimmer.",
        },
        "summary": "Your features align with Spring Bright, which suits vivid, warm, clear, and high-energy shades.",
    },
    "Spring Light": {
        "season": "Spring",
        "subtype": "Light",
        "full_season": "Spring Light",
        "undertone": "warm",
        "palette": ["#FFDAB9", "#98FB98", "#AFEEEE", "#FFFACD", "#FFB6C1", "#E0FFFF"],
        "outfit_colors": [
            {"name": "Warm Peach", "hex": "#FFDAB9"},
            {"name": "Light Aqua", "hex": "#E0FFFF"},
            {"name": "Soft Coral", "hex": "#F88379"},
            {"name": "Light Mint", "hex": "#98FB98"},
        ],
        "avoid_colors": [
            {"name": "Black", "hex": "#000000"},
            {"name": "Dark Burgundy", "hex": "#800000"},
            {"name": "Deep Navy", "hex": "#000080"},
            {"name": "Heavy Brown", "hex": "#4B2E1E"},
        ],
        "makeup_tips": {
            "foundation": "Light, warm, or neutral-warm base. Dewy or fresh finish.",
            "blush": "Soft apricot, warm pink, peach, or light coral.",
            "lips": "Peach gloss, light coral, soft warm rose, or sheer watermelon.",
            "eyes": "Champagne, soft warm brown, light gold, and peachy beige.",
        },
        "summary": "Your features align with Spring Light, which suits light, warm, clear, and delicate pastel shades.",
    },
    "Spring Warm": {
        "season": "Spring",
        "subtype": "Warm",
        "full_season": "Spring Warm",
        "undertone": "warm",
        "palette": ["#FF7F50", "#FFA500", "#32CD32", "#FFD700", "#1E90FF", "#FF4500"],
        "outfit_colors": [
            {"name": "Coral", "hex": "#FF7F50"},
            {"name": "Golden Yellow", "hex": "#FFD700"},
            {"name": "Fresh Green", "hex": "#32CD32"},
            {"name": "Warm Orange", "hex": "#FFA500"},
        ],
        "avoid_colors": [
            {"name": "Slate Grey", "hex": "#708090"},
            {"name": "Dusty Mauve", "hex": "#B08D9B"},
            {"name": "Icy Blue", "hex": "#D6F0FF"},
            {"name": "Cool Burgundy", "hex": "#800020"},
        ],
        "makeup_tips": {
            "foundation": "Warm, golden, or yellow-based. Fresh radiant finish.",
            "blush": "Bright coral, warm peach, apricot, or golden pink.",
            "lips": "Warm orange-red, vibrant coral, peach-red, or bright warm rose.",
            "eyes": "Bronze, warm gold, caramel brown, and vibrant teal liner.",
        },
        "summary": "Your features align with Spring Warm, which suits bright, sunny, golden, and energetic shades.",
    },
}



VALID_SEASONS = {"Spring", "Summer", "Autumn", "Winter"}
VALID_SUBTYPES = {"Bright", "Light", "Warm", "Soft", "Cool", "Deep"}
DEFAULT_SUBTYPE = {
    "Spring": "Warm",
    "Summer": "Soft",
    "Autumn": "Warm",
    "Winter": "Cool",
}

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


def _extract_season_subtype(raw_text: str) -> tuple[str | None, str | None]:
    """
    Handles outputs such as:
        {"season":"Autumn","subtype":"Deep"}
        Autumn Deep
        season: Autumn subtype: Deep
        ```json ... ```
    """
    if not raw_text:
        return None, None

    cleaned = str(raw_text).strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()

    # 1) Exact JSON.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return (
                _normalize_word(parsed.get("season"), VALID_SEASONS),
                _normalize_word(parsed.get("subtype"), VALID_SUBTYPES),
            )
    except json.JSONDecodeError:
        pass

    # 2) JSON embedded inside extra text.
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return (
                    _normalize_word(parsed.get("season"), VALID_SEASONS),
                    _normalize_word(parsed.get("subtype"), VALID_SUBTYPES),
                )
        except json.JSONDecodeError:
            pass

    # 3) Plain text extraction.
    season = None
    subtype = None

    for valid_season in sorted(VALID_SEASONS):
        if re.search(rf"{valid_season}", cleaned, re.IGNORECASE):
            season = valid_season
            break

    for valid_subtype in sorted(VALID_SUBTYPES):
        if re.search(rf"{valid_subtype}", cleaned, re.IGNORECASE):
            subtype = valid_subtype
            break

    return season, subtype


def _template_from_prediction(season: str, subtype: str | None) -> dict:
    if subtype is not None:
        key = f"{season} {subtype}"
        if key in SEASON_SUBTYPE_TEMPLATES:
            return copy.deepcopy(SEASON_SUBTYPE_TEMPLATES[key])

    fallback_subtype = DEFAULT_SUBTYPE[season]
    fallback_key = f"{season} {fallback_subtype}"
    result = copy.deepcopy(SEASON_SUBTYPE_TEMPLATES[fallback_key])
    result["fallback_used"] = True
    result["fallback_reason"] = "Model did not return a valid subtype."
    return result


def _parse_model_response(raw_text: str) -> dict:
    season, subtype = _extract_season_subtype(raw_text)

    if season is None:
        raise ValueError(f"Could not determine season from model response. Raw output: {raw_text!r}")

    result = _template_from_prediction(season, subtype)
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
            "Place the trained adapter folder in the project root, or set ASTA_ADAPTER_DIR."
        )

    if not hasattr(model, "text_model"):
        raise RuntimeError(
            "Loaded Moondream model does not expose text_model, so the LoRA "
            "adapter cannot be attached. Check MODEL_ID and REVISION."
        )

    # Patch again immediately before PEFT wraps text_model.
    # This matters because some configs are initialized lazily.
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
        Full seasonal color result dictionary expected by the UI.
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
