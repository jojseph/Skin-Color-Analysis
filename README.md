# Agarthan Skin Tone Analyzer (ASTA)

ASTA is a Streamlit web app for seasonal colour analysis. It detects and crops a face from a portrait image, runs a locally loaded fine-tuned Vision-Language Model (VLM), and maps the model prediction to deterministic colour recommendation templates.

The current project version uses **Moondream2 + LoRA fine-tuning** for a 12-class seasonal colour classification task.

```text
Face image → Moondream2 + LoRA → {"season":"Winter","subtype":"Deep"} → template-based recommendations
```

The model predicts only the season and subtype. Palettes, outfit colours, avoid colours, makeup tips, and summaries are handled by `analyzer.py` templates for stable app output.

---

## Current project status

This project currently works as a local prototype/demo.

The trained model was evaluated on the test set with the following strict metrics:

```text
JSON / parse validity rate: 68.86%
Season accuracy: 33.55%
Subtype accuracy: 31.47%
Full class accuracy: 19.85%
```

Interpretation:

- The fine-tuning pipeline works.
- The model learned some seasonal signal above random chance.
- Exact 12-class classification is still weak.
- The app includes fallback handling so season-only outputs can still render a general seasonal palette.
- Results should be treated as AI-generated style suggestions, not professional colour analysis.

---

## Google Drive project files

The supporting project files, trained LoRA adapter, generated dataset files, evaluation outputs, and raw face dataset are available in this shared Google Drive folder:

[ASTA Google Drive Folder](https://drive.google.com/drive/folders/1dmCoBFlYuOeuWVFAysHIOuEdwtMFF4Yx?usp=sharing)

---
## Pages

| Page | Description |
|---|---|
| 🎨 Skin Tone Analyzer | Upload/capture a portrait image and get a seasonal colour result, palette, outfit suggestions, avoid colours, and makeup tips |
| 📊 LLM Report | Academic mini-project report for the CS346 VLM/LLM project |

---

## How it works

```text
Upload / Camera / Sample
↓
Face detection and crop using OpenCV
↓
Base64/PIL image processing
↓
Moondream2 base model + local LoRA adapter
↓
Model predicts season + subtype
↓
analyzer.py expands prediction using templates
↓
Streamlit renders the result
```

### Main pipeline files

| File | Purpose |
|---|---|
| `main.py` | Streamlit multipage entry point |
| `pages/app.py` | Analyzer UI |
| `pages/report.py` | Report page |
| `face_utils.py` | Face detection, crop, resize, and Base64 encoding |
| `analyzer.py` | Loads Moondream2 + LoRA, parses predictions, applies templates/fallbacks |
| `generate_dataset.py` | Builds JSONL training/testing data from `data/rgb` |
| `requirements.txt` | Local dependencies |

---

## Model design

ASTA uses a narrow VLM classification objective:

```text
Input: cropped face image
Output: {"season":"Autumn","subtype":"Deep"}
```

The 12 supported classes are:

```text
Autumn Deep
Autumn Soft
Autumn Warm
Spring Bright
Spring Light
Spring Warm
Summer Cool
Summer Light
Summer Soft
Winter Bright
Winter Cool
Winter Deep
```

The model does **not** generate the full palette or recommendation text. This is intentional because template-based recommendations are more consistent, easier to debug, and easier to evaluate.

---

## Dataset structure

The current dataset should be placed under:

```text
data/rgb/
├── train/
│   ├── autumn_deep/
│   ├── autumn_soft/
│   ├── autumn_warm/
│   ├── spring_bright/
│   ├── spring_light/
│   ├── spring_warm/
│   ├── summer_cool/
│   ├── summer_light/
│   ├── summer_soft/
│   ├── winter_bright/
│   ├── winter_cool/
│   └── winter_deep/
└── test/
    ├── autumn_deep/
    ├── autumn_soft/
    ├── autumn_warm/
    ├── spring_bright/
    ├── spring_light/
    ├── spring_warm/
    ├── summer_cool/
    ├── summer_light/
    ├── summer_soft/
    ├── winter_bright/
    ├── winter_cool/
    └── winter_deep/
```

Generate the cropped training dataset and JSONL files with:

```powershell
python generate_dataset.py --max-per-class 260
```

This creates:

```text
data/
├── moondream_train_classification.jsonl
├── moondream_test_classification.jsonl
└── training_crops/
    ├── train/
    └── test/
```

The JSONL target format is intentionally minimal:

```json
{"season":"Winter","subtype":"Deep"}
```

---

## Required local model folder

After training in Colab, copy the final LoRA adapter folder into the app root:

```text
Skin-Color-Analysis/
├── analyzer.py
├── main.py
├── moondream_season_lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── ...
```

The app expects:

```python
ADAPTER_DIR = "moondream_season_lora"
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"
```

Keep the base model revision consistent with the training notebook.

---

## Installation

### 1. Create and activate a Python 3.11 virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Important pinned packages:

```text
numpy==1.26.4
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
opencv-python==4.10.0.84
transformers==4.41.2
peft==0.8.2
accelerate==0.30.1
pyvips==3.1.1
pyvips-binary==8.18.2
```

Do not casually upgrade `transformers`, `peft`, or `accelerate`. Newer versions caused Moondream/Phi + LoRA compatibility errors during local testing.

### 3. Verify `pyvips`

```powershell
python -c "import pyvips; print('pyvips ok')"
```

Expected:

```text
pyvips ok
```

If `pyvips` fails with `libvips-42.dll` missing, reinstall:

```powershell
pip uninstall -y pyvips
pip install pyvips-binary pyvips
```

### 4. Run the app

```powershell
streamlit run main.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

---

## Common issues and fixes

### `No module named 'pyvips'`

Install:

```powershell
pip install pyvips-binary pyvips
```

Then test:

```powershell
python -c "import pyvips; print('pyvips ok')"
```

### `libvips-42.dll` missing

Use the binary package:

```powershell
pip uninstall -y pyvips
pip install pyvips-binary pyvips
```

### `'HfMoondream' object has no attribute 'all_tied_weights_keys'`

This happens when the LoRA adapter is loaded into the full Moondream wrapper instead of `model.text_model`.

Fix: `analyzer.py` should attach the adapter to:

```python
model.text_model = PeftModel.from_pretrained(model.text_model, str(ADAPTER_DIR))
```

### `'PhiConfig' object has no attribute 'pad_token_id'`

This is usually a version compatibility issue between Moondream, Phi, Transformers, and PEFT.

Use the pinned versions:

```powershell
pip uninstall -y transformers peft accelerate
pip install transformers==4.41.2 peft==0.8.2 accelerate==0.30.1
```

### `Moondream.init() got an unexpected keyword argument 'dtype'`

With `transformers==4.41.2`, use:

```python
torch_dtype=dtype
```

not:

```python
dtype=dtype
```

inside `AutoModelForCausalLM.from_pretrained(...)`.

### Hugging Face symlink warning on Windows

This is not fatal. It only means the cache may use more disk space. You can ignore it, enable Windows Developer Mode, or run as administrator.

---

## Usage tips for better results

Use portraits with:

- natural daylight
- plain white/gray background
- no heavy filters
- no strong coloured wall reflections
- face clearly visible
- minimal shadows
- glasses removed if possible

Avoid judging the model from images with strong colour casts, such as green walls or warm indoor lighting, because these can distort skin undertone.

---

## Output fields

| Field | Description |
|---|---|
| `season` | Main season: Spring, Summer, Autumn, or Winter |
| `subtype` | Subtype if confidently parsed, such as Deep, Cool, Soft, Warm, Light, or Bright |
| `full_season` | Combined result, such as Winter Deep, or general season fallback such as Winter |
| `undertone` | Warm or cool template-based undertone |
| `palette` | Six hex colours from the matching template |
| `outfit_colors` | Recommended wearable colours |
| `avoid_colors` | Colours that may clash with the predicted palette |
| `makeup_tips` | Foundation, blush, lips, and eye colour suggestions |
| `summary` | Short explanation |
| `fallback_used` | Whether fallback parsing/template logic was used |
| `raw_model_response` | Raw model output, useful for debugging |

---

## Notes for the CS346 mini-project report

This project fits under the **LLM** category because it uses a Vision-Language Model, which combines visual input encoding with language-model output generation.

Recommended report framing:

```text
The project fine-tunes a pre-trained VLM using LoRA for a narrow 12-class seasonal colour classification task. The model predicts only season and subtype, while deterministic templates generate the final recommendations.
```

Recommended metrics to report:

```text
Training loss per epoch
JSON / parse validity rate
Season accuracy
Subtype accuracy
Full class accuracy
Per-class precision, recall, and F1-score
Confusion matrix
```

Current strict evaluation results:

```text
JSON / parse validity rate: 68.86%
Season accuracy: 33.55%
Subtype accuracy: 31.47%
Full class accuracy: 19.85%
```

Conclusion:

```text
The VLM fine-tuning pipeline was successfully implemented, but strict 12-class performance remains limited. The model works as a prototype and demonstrates the training/integration pipeline, but it requires better data quality, stronger output constraints, and further evaluation before being treated as a reliable colour-analysis classifier.
```

---

## File structure

```text
Skin-Color-Analysis/
├── main.py
├── analyzer.py
├── face_utils.py
├── generate_dataset.py
├── requirements.txt
├── moondream_season_lora/
├── data/
│   ├── rgb/
│   ├── training_crops/
│   ├── moondream_train_classification.jsonl
│   └── moondream_test_classification.jsonl
├── pages/
│   ├── app.py
│   └── report.py
└── samples/
```

---

## Disclaimer

ASTA results are AI-generated style suggestions. They are not a substitute for professional personal colour analysis.
