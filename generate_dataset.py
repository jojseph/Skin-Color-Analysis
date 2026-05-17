"""
generate_dataset.py
Builds JSONL files for ASTA VLM fine-tuning.

Updated for the simplified 4-class seasonal color setup.

Expected project structure:
    data/rgb/
    ├── train/
    │   ├── autumn/
    │   ├── spring/
    │   ├── summer/
    │   └── winter/
    └── test/
        ├── autumn/
        ├── spring/
        ├── summer/
        └── winter/

Output:
    data/moondream_train_classification.jsonl
    data/moondream_test_classification.jsonl
    data/training_crops/<split>/<season>/...

The JSONL target intentionally contains only the main season.
The app/analyzer expands the prediction into generalized palettes and recommendations later.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image

from face_utils import process_pil_image


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_DATASET_DIR = Path("./data/rgb")
DEFAULT_CROP_DIR = Path("./data/training_crops")
DEFAULT_TRAIN_OUTPUT = Path("./data/moondream_train_classification.jsonl")
DEFAULT_TEST_OUTPUT = Path("./data/moondream_test_classification.jsonl")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
RANDOM_SEED = 42

CLASS_NAMES = ["autumn", "spring", "summer", "winter"]

CLASS_MAP: dict[str, str] = {
    "autumn": "Autumn",
    "spring": "Spring",
    "summer": "Summer",
    "winter": "Winter",
}

TRAINING_PROMPT = """
Classify this face into exactly one of these 4 seasonal color classes:

Autumn, Spring, Summer, Winter.

Return ONLY this JSON format:
{"season":"Winter"}

No explanation. No markdown. No extra words.
""".strip()

TRAINING_PROMPTS = [TRAINING_PROMPT]


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "image"


def make_answer(season: str) -> str:
    return json.dumps(
        {"season": season},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_dataset_structure(dataset_dir: Path) -> None:
    missing: list[str] = []
    unexpected: list[str] = []

    for split in ["train", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            missing.append(str(split_dir))
            continue

        found_class_dirs = {p.name.lower() for p in split_dir.iterdir() if p.is_dir()}
        expected_class_dirs = set(CLASS_MAP)

        missing.extend(str(split_dir / class_name) for class_name in sorted(expected_class_dirs - found_class_dirs))
        unexpected.extend(str(split_dir / class_name) for class_name in sorted(found_class_dirs - expected_class_dirs))

    if missing:
        print("\nMissing expected folders:")
        for item in missing:
            print(f"  - {item}")

    if unexpected:
        print("\nIgnoring unexpected folders:")
        for item in unexpected:
            print(f"  - {item}")

    if any(not (dataset_dir / split).exists() for split in ["train", "test"]):
        raise FileNotFoundError(f"Dataset must contain train/ and test/ under: {dataset_dir}")


def collect_images(dataset_dir: Path, split: str) -> list[tuple[Path, str]]:
    split_dir = dataset_dir / split
    items: list[tuple[Path, str]] = []

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        image_paths = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )

        for image_path in image_paths:
            items.append((image_path, class_name))

    return items


def balance_items(
    items: list[tuple[Path, str]],
    max_per_class: Optional[int],
) -> list[tuple[Path, str]]:
    if max_per_class is None:
        return items

    grouped: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for item in items:
        _, class_name = item
        grouped[class_name].append(item)

    balanced: list[tuple[Path, str]] = []
    for class_name, group in grouped.items():
        random.shuffle(group)
        balanced.extend(group[:max_per_class])

    random.shuffle(balanced)
    return balanced


def build_sample(
    image_path: Path,
    class_name: str,
    split: str,
    crop_dir: Path,
    use_original_if_no_face: bool,
) -> tuple[dict | None, str | None]:
    season = CLASS_MAP[class_name]

    try:
        pil_image = Image.open(image_path).convert("RGB")
        _, cropped_face = process_pil_image(pil_image)

        if cropped_face is None:
            if not use_original_if_no_face:
                return None, "no_face"
            cropped_face = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
            source_status = "original_fallback"
        else:
            source_status = "face_crop"

        safe_stem = clean_filename(image_path.stem)
        crop_name = f"{split}_{class_name}_{safe_stem}{image_path.suffix.lower()}"
        crop_path = crop_dir / split / class_name / crop_name
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cropped_face.save(crop_path)

        sample = {
            "image_path": str(crop_path).replace("\\", "/"),
            "qa": [
                {
                    "question": random.choice(TRAINING_PROMPTS),
                    "answer": make_answer(season),
                }
            ],
        }
        return sample, source_status

    except Exception as exc:
        print(f"  -> Error processing {image_path}: {exc}")
        return None, "error"


def write_jsonl(
    items: list[tuple[Path, str]],
    output_file: Path,
    split: str,
    crop_dir: Path,
    use_original_if_no_face: bool,
) -> tuple[Counter, Counter]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    status_counts: Counter = Counter()

    with output_file.open("w", encoding="utf-8") as f:
        for idx, (image_path, class_name) in enumerate(items, 1):
            if idx % 50 == 0 or idx == len(items):
                print(f"Processed {idx}/{len(items)} for {split}...")

            sample, status = build_sample(
                image_path=image_path,
                class_name=class_name,
                split=split,
                crop_dir=crop_dir,
                use_original_if_no_face=use_original_if_no_face,
            )

            status_counts[status or "unknown"] += 1
            if sample is None:
                continue

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[CLASS_MAP[class_name]] += 1

    return counts, status_counts


def print_counts(title: str, counts: Counter) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    for class_name in CLASS_NAMES:
        season = CLASS_MAP[class_name]
        print(f"{season}: {counts[season]}")


def build_dataset(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    crop_dir = Path(args.crop_dir)

    validate_dataset_structure(dataset_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    split_outputs = {
        "train": Path(args.train_output),
        "test": Path(args.test_output),
    }

    for split, output_file in split_outputs.items():
        print(f"\nBuilding {split} JSONL from {dataset_dir / split}")
        items = collect_images(dataset_dir, split)
        print(f"Found {len(items)} raw images.")

        items = balance_items(items, args.max_per_class)
        print(f"Using {len(items)} images after optional balancing.")
        random.shuffle(items)

        counts, status_counts = write_jsonl(
            items=items,
            output_file=output_file,
            split=split,
            crop_dir=crop_dir,
            use_original_if_no_face=args.use_original_if_no_face,
        )

        print(f"\nSaved: {output_file}")
        print(f"Written: {sum(counts.values())}")
        print(f"Processing status: {dict(status_counts)}")
        print_counts(f"{split.upper()} LABEL COUNTS", counts)

    print("\nDataset generation complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASTA JSONL files for 4-class VLM fine-tuning.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Root dataset folder. Default: ./data/rgb")
    parser.add_argument("--crop-dir", default=str(DEFAULT_CROP_DIR), help="Where cropped faces are saved.")
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_OUTPUT), help="Train JSONL output path.")
    parser.add_argument("--test-output", default=str(DEFAULT_TEST_OUTPUT), help="Test JSONL output path.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap per class. Default: use all images.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--skip-no-face",
        dest="use_original_if_no_face",
        action="store_false",
        help="Skip images where OpenCV cannot detect a face instead of using the resized original image.",
    )
    parser.set_defaults(use_original_if_no_face=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
