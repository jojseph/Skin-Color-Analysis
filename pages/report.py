"""
pages/1_📊_Report.py
Academic mini-project report for ASTA under the LLM/VLM track.

Current implementation:
    Deep Armocromia face dataset
    -> OpenCV face crop
    -> Moondream2 Vision-Language Model
    -> LoRA fine-tuning
    -> 4-class seasonal colour classification
    -> deterministic recommendation templates
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    import pandas as pd
except Exception:  # Keeps the page usable even if pandas is not installed.
    pd = None


# ── Page setup ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CS346 Mini-Project Report — ASTA",
    page_icon="📊",
    layout="wide",
)

st.title("📊 CS346 Mini-Project Report")
st.subheader("Agarthan Skin Tone Analyzer (ASTA): 4-Season Classification using a Fine-Tuned Vision-Language Model")
st.caption("Mini-project category: Large Language Models (LLMs) / Vision-Language Models (VLMs)")
st.divider()


# ── Helper data ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "moondream2_test_predictions.csv"
CONFUSION_CSV = PROJECT_ROOT / "data" / "moondream2_confusion_matrix.csv"

STATIC_METRICS = {
    "JSON / parse validity rate": "100.00%",
    "Season accuracy": "51.54%",
    "Test samples": "912",
    "Classes": "4 — Autumn, Spring, Summer, Winter",
}

CLASSIFICATION_REPORT = [
    {"Class": "Autumn", "Precision": 0.60, "Recall": 0.30, "F1-score": 0.40, "Support": 259},
    {"Class": "Spring", "Precision": 0.47, "Recall": 0.17, "F1-score": 0.25, "Support": 203},
    {"Class": "Summer", "Precision": 0.40, "Recall": 0.83, "F1-score": 0.54, "Support": 186},
    {"Class": "Winter", "Precision": 0.62, "Recall": 0.78, "F1-score": 0.69, "Support": 264},
]

CONFUSION_MATRIX = [
    {"Actual": "Autumn", "Autumn": 77, "Spring": 26, "Summer": 65, "Winter": 91},
    {"Actual": "Spring", "Autumn": 18, "Spring": 34, "Summer": 137, "Winter": 14},
    {"Actual": "Summer", "Autumn": 7, "Spring": 6, "Summer": 154, "Winter": 19},
    {"Actual": "Winter", "Autumn": 26, "Spring": 6, "Summer": 27, "Winter": 205},
]


def show_metric_cards(metrics: dict[str, str]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def load_optional_dataframe(path: Path):
    if pd is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None



# ── 1. Project Overview ──────────────────────────────────────────────────────────
st.header("1. Project Overview")
st.markdown("""
The **Agarthan Skin Tone Analyzer (ASTA)** is a Streamlit application that classifies a
portrait image into one of the four seasonal colour categories:

- **Autumn**
- **Spring**
- **Summer**
- **Winter**

After classification, the app maps the predicted season to a stable recommendation
template containing colour palettes, outfit colour suggestions, colours to avoid, and
makeup guidance. The model is intentionally responsible only for the **classification
step**. The recommendation text is template-based so the output remains consistent,
debuggable, and easier to evaluate.

the project uses a **pre-trained model with task-specific
fine-tuning**. The selected base model is **Moondream2**, and the model is adapted to
the seasonal colour classification task using **LoRA fine-tuning**.

| Requirement | ASTA implementation |
|---|---|
| Mini-project model category | LLMs / Vision-Language Models |
| Model strategy | Pre-trained VLM + LoRA fine-tuning |
| Main task | Face seasonal colour classification |
| Output classes | Autumn, Spring, Summer, Winter |
| Application | Streamlit web app for seasonal colour analysis |
| Final recommendations | Deterministic templates based on predicted season |
""")

st.code(
    """
User portrait image
    ↓
OpenCV face detection and 512×512 crop
    ↓
Moondream2 VLM + LoRA adapter
    ↓
Structured prediction: {"season": "Winter"}
    ↓
Template-based palette and recommendation rendering
    ↓
Streamlit result page
""".strip(),
    language="text",
)

st.divider()


# ── 3. Dataset ──────────────────────────────────────────────────────────────────
st.header("2. Dataset Used")
st.markdown("""
The project uses the **Deep Armocromia** dataset, a face seasonal colour analysis
dataset introduced in the ECCV 2024 Workshops proceedings. The original dataset is
associated with seasonal colour analysis and classification, making it directly aligned
with ASTA's task.

For this implementation, the dataset was simplified into **four main seasonal classes**
instead of subtype-level labels. This was done because early 12-class experiments were
too unstable for reliable subtype classification, while the 4-class version better fits
the practical scope of the Streamlit prototype and the mini-project timeline.
""")

st.markdown("""
**Dataset citation used in the report:**

```bibtex
@InProceedings{10.1007/978-3-031-91569-7_22,
  author="Stacchio, Lorenzo
  and Paolanti, Marina
  and Spigarelli, Francesca
  and Frontoni, Emanuele",
  editor="Del Bue, Alessio
  and Canton, Cristian
  and Pont-Tuset, Jordi
  and Tommasi, Tatiana",
  title="Deep Armocromia: A Novel Dataset for Face Seasonal Color Analysis and Classification",
  booktitle="Computer Vision -- ECCV 2024 Workshops",
  year="2025",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="352--367",
  isbn="978-3-031-91569-7"
}
```

Repository: `https://github.com/lorenzo-stacchio/Deep-Armocromia`
""")


st.subheader("2.1 JSONL Training Format")
st.markdown("""
The dataset generator converts the image folders into JSONL samples. Each sample pairs
a cropped face image with a direct question-answer target. The output intentionally
contains only the season label.
""")

st.code(
    '''{
  "image_path": "data/training_crops/train/winter/train_winter_001.jpg",
  "qa": [
    {
      "question": "Classify this face into exactly one of these 4 seasonal color classes: Autumn, Spring, Summer, Winter. Return ONLY this JSON format: {\"season\":\"Winter\"}",
      "answer": "{\"season\":\"Winter\"}"
    }
  ]
}''',
    language="json",
)

st.divider()


# ── 3. Why a VLM is an LLM ───────────────────────────────────────────────────────
st.header("3. Why This Counts as an LLM Project")
st.markdown("""
A **Vision-Language Model** is a multimodal extension of a language model. Instead of
receiving only text tokens, a VLM also receives image representations. The model then
uses a language-model backbone to produce text output.

In ASTA, the input image is encoded visually, then paired with a text prompt asking the
model to classify the face into a season. The output is generated as text in a strict
JSON format. Because the final prediction is produced through language-model generation,
the system fits the **LLM/VLM track** of the assignment.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    #### Vision side
    - Face image is loaded with PIL/OpenCV
    - Largest face is detected
    - Face is padded and resized to 512×512
    - Moondream2 encodes the image into visual embeddings
    """)
with col2:
    st.markdown("""
    #### Language side
    - A structured classification prompt is provided
    - The model generates a textual JSON answer
    - Parser extracts the `season` field
    - App maps the label to fixed recommendation templates
    """)

st.divider()


# ── 4. Model Used ────────────────────────────────────────────────────────────────
st.header("4. Model Used: Moondream2 + LoRA")
st.markdown("""
The project uses **Moondream2** as the base Vision-Language Model. Moondream2 is a
small VLM suitable for local experimentation because it can accept an image and answer
questions about the image through text generation.

Instead of training all model weights, ASTA uses **LoRA (Low-Rank Adaptation)**.
LoRA adds lightweight trainable adapter matrices to selected parts of the model. During
fine-tuning, the original model remains mostly frozen while the adapter learns the task.
This makes training more feasible in a student project environment.
""")

st.markdown("""
| Component | Role |
|---|---|
| `vikhyatk/moondream2` | Pre-trained VLM base model |
| LoRA adapter | Task-specific fine-tuned parameters |
| OpenCV Haar Cascade | Face detection and cropping |
| JSON prompt | Constrains the prediction format |
| `analyzer.py` templates | Converts season prediction into stable user-facing recommendations |
""")

st.subheader("4.1 Training Objective")
st.markdown("""
The training objective is narrow and supervised: given a cropped face image, the model
must output exactly one valid season label.
""")

st.code(
    '''PROMPT = """
Classify this face into exactly one of these 4 seasonal color classes:

Autumn, Spring, Summer, Winter.

Return ONLY this JSON format:
{"season":"Winter"}

No explanation. No markdown. No extra words.
"""''',
    language="python",
)

st.warning(
    "The model does not generate the palette itself. It only predicts the season. "
    "This avoids unstable or hallucinated recommendation text and makes evaluation clearer."
)

st.divider()


# ── 5. System Pipeline ──────────────────────────────────────────────────────────
st.header("5. System Pipeline")
st.code(
    """
┌─────────────────────────────────────────────────────────────┐
│ User Input                                                  │
│ File upload / camera capture / sample image                 │
└──────────────────────────────┬──────────────────────────────┘
                               │ PIL image
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ face_utils.py                                               │
│ OpenCV Haar Cascade → largest face → 25% padding → 512 crop │
└──────────────────────────────┬──────────────────────────────┘
                               │ cropped face image
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ analyzer.py                                                 │
│ Load Moondream2 → attach LoRA to text_model → run prompt    │
└──────────────────────────────┬──────────────────────────────┘
                               │ raw model text
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Parser                                                      │
│ json.loads → embedded JSON fallback → season-word fallback  │
└──────────────────────────────┬──────────────────────────────┘
                               │ season label
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Template Renderer                                           │
│ Palette · outfit colours · avoid colours · makeup tips      │
└──────────────────────────────┬──────────────────────────────┘
                               │ final result
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Streamlit UI                                                │
│ Displays the analysis result to the user                    │
└─────────────────────────────────────────────────────────────┘
""".strip(),
    language="text",
)

st.divider()


# ── 6. Preprocessing ─────────────────────────────────────────────────────────────
st.header("6. Image Preprocessing")
st.markdown("""
Before inference, the system performs lightweight preprocessing. The goal is not to
manually classify the skin tone using handcrafted colour thresholds. The goal is only
to standardise the input so the VLM receives a centered face image.

The preprocessing stage performs:

1. RGB conversion
2. Grayscale conversion for OpenCV face detection
3. Frontal-face detection, with profile-face fallback
4. Largest-face selection
5. 25% padding around the detected face
6. Resize to 512×512
7. JPEG/Base64 conversion when needed by the app pipeline
""")

st.info(
    "This design keeps computer vision preprocessing minimal. The learned VLM handles "
    "the classification, while OpenCV only prepares a consistent face crop."
)

st.divider()


# ── 7. Evaluation Results ────────────────────────────────────────────────────────
st.header("7. Evaluation Results")
st.markdown("""
The fine-tuned model was evaluated on the test split using strict exact-match season
classification. A prediction is considered correct only when the parsed season matches
the expected label.
""")

show_metric_cards(STATIC_METRICS)

st.subheader("7.1 Per-Class Classification Report")
if pd is not None:
    st.dataframe(pd.DataFrame(CLASSIFICATION_REPORT), use_container_width=True, hide_index=True)
else:
    st.table(CLASSIFICATION_REPORT)

st.subheader("7.2 Confusion Matrix")
st.markdown("Rows represent the **actual** class. Columns represent the **predicted** class.")

loaded_confusion = load_optional_dataframe(CONFUSION_CSV)
if loaded_confusion is not None:
    if "Unnamed: 0" in loaded_confusion.columns:
        loaded_confusion = loaded_confusion.rename(columns={"Unnamed: 0": "Actual"})
    st.dataframe(loaded_confusion, use_container_width=True, hide_index=True)
elif pd is not None:
    st.dataframe(pd.DataFrame(CONFUSION_MATRIX), use_container_width=True, hide_index=True)
else:
    st.table(CONFUSION_MATRIX)

st.subheader("7.3 Interpretation")
st.markdown("""
The evaluation shows that the pipeline is functional but still limited:

- **Parse validity is strong**: the model consistently returns a parseable season label.
- **Winter performs best** among the four classes, with the highest F1-score.
- **Summer has high recall but lower precision**, meaning the model often predicts Summer.
- **Spring is the weakest class**, suggesting overlap between Spring and nearby warm/bright
  or soft categories.
- The result is above random chance for a 4-class task, but it is not yet reliable enough
  to be treated as a professional colour-analysis system.
""")

st.divider()

# ── 10. Comparison with Classical CV ─────────────────────────────────────────────
st.header("8. Why Use a VLM Instead of a Classical Vision Pipeline?")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    #### Classical CV approach
    A traditional solution would likely require:

    1. Face detection
    2. Skin segmentation
    3. Hair/eye region extraction
    4. RGB to Lab/HSV conversion
    5. Dominant colour clustering
    6. Rule-based season mapping
    7. Manual palette lookup

    This approach is explainable, but brittle. Lighting, camera quality, makeup,
    shadows, and background colour casts can easily distort measured RGB values.
    """)

with col_b:
    st.markdown("""
    #### VLM approach
    ASTA uses the VLM to learn the mapping from face crop to season label.

    Advantages:

    - Handles image and text in one model
    - Learns from labelled examples instead of handcrafted thresholds
    - Produces structured textual output
    - Can be fine-tuned using prompt-answer pairs
    - Easier to integrate with a Streamlit app workflow

    The trade-off is that the model is less directly interpretable than a purely
    rule-based colour pipeline.
    """)

st.divider()


# ── 11. Limitations ──────────────────────────────────────────────────────────────
st.header("10. Limitations")
st.markdown("""
ASTA should be treated as a prototype and academic demonstration, not as a professional
or certified colour-analysis tool.

Current limitations include:

- **Lighting sensitivity**: warm indoor lighting, coloured walls, and shadows can alter
  apparent undertone.
- **Face detection dependency**: if OpenCV fails to detect the face correctly, the crop
  may be poor.
- **Class overlap**: seasonal classes are subjective and visually overlapping.
- **Dataset constraints**: performance depends on the quality, distribution, and label
  consistency of the Deep Armocromia images used.
- **No confidence calibration**: the current app returns a single season rather than a
  probability distribution.
- **Template generalisation**: recommendations are generalized per season, not customized
  to every individual feature.
""")

st.divider()


# ── 12. Future Work ──────────────────────────────────────────────────────────────
st.header("11. Future Work")
st.markdown("""
Possible improvements include:

1. **Improve dataset quality control** by filtering extreme lighting, heavy filters,
   occlusions, and incorrect face crops.
2. **Balance the dataset more carefully** so each season contributes equal training
   signal.
3. **Add confidence scoring** by evaluating multiple generations or adding classifier-style
   logits if supported by the model pipeline.
4. **Use a stronger VLM** if compute resources allow, while keeping the same 4-class task.
5. **Revisit subtype classification** only after the 4-class model becomes stable.
6. **Add explainability tools** such as crop previews, prediction logs, and confusion
   matrix dashboards inside the Streamlit report page.
7. **Collect local validation samples** from controlled lighting conditions to test how
   the model performs on Filipino/Cebuano users and real deployment images.
""")

st.divider()


# ── 13. Conclusion ───────────────────────────────────────────────────────────────
st.header("12. Conclusion")
st.markdown("""
ASTA demonstrates how a pre-trained **Vision-Language Model** can be adapted to a
specialized computer-vision classification task through **LoRA fine-tuning**. The
project uses the Deep Armocromia dataset, converts portrait images into cropped face
samples, trains the model to output a strict JSON season label, and integrates the
result into a Streamlit web application.

The revised 4-season setup is more appropriate than the earlier subtype-level design
because it produces clearer evaluation results and a more stable app workflow. The
current model reaches **51.54% season accuracy** with **100% parse validity** on the
reported test set. This confirms that the end-to-end VLM fine-tuning and deployment
pipeline works, although the classifier still requires stronger data quality and model
improvements before it can be considered reliable.

Overall, the project satisfies the **LLM/VLM mini-project category** by using a
pre-trained multimodal language model, applying task-specific fine-tuning, and deploying
it in a functional application.
""")

st.divider()


# ── 14. References ───────────────────────────────────────────────────────────────
st.header("13. References")
st.markdown("""
1. Stacchio, L., Paolanti, M., Spigarelli, F., & Frontoni, E. (2025). *Deep Armocromia:
   A Novel Dataset for Face Seasonal Color Analysis and Classification*. In
   **Computer Vision — ECCV 2024 Workshops**, Springer Nature Switzerland, pp. 352–367.
2. Deep Armocromia GitHub repository: `https://github.com/lorenzo-stacchio/Deep-Armocromia`
3. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
4. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
5. Moondream2 model family by Vikhyat K. / Hugging Face: `vikhyatk/moondream2`.
6. Streamlit documentation for Python web app deployment.
7. OpenCV documentation for Haar Cascade face detection.
""")

st.divider()
st.caption("Agarthan Skin Tone Analyzer · CS346 Mini-Project Report · 2026")
