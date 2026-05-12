# Agarthan Skin Tone Analyzer (ASTA)

A two-page Streamlit web app that analyzes a portrait photo using **OpenCV** for face detection and a **Vision Language Model (VLM)** to infer the user’s **personal color type inspired by Korean color analysis**. It classifies the user into seasonal groups (Spring, Summer, Autumn, Winter) based on **undertone, depth, and chroma**, then generates a **personalized color palette**, **outfit recommendations**, and **makeup guidance**. A built-in **LLM report page** documents the model architecture and prompt engineering approach used.

## Pages

| Page | Description |
|---|---|
| 🎨 Skin Tone Analyzer | Upload a photo, use your camera, or pick a sample — get your season, palette, outfit tips, and makeup advice |
| 📊 LLM Report | Academic mini-project report covering Transformer architecture, VLM design, Gemini 2.5 Flash, and prompt engineering |

## How it works

![App screenshot](samples/image.png)

```
Upload / Camera / Sample → Face detection & crop (OpenCV) → VLM analysis (Gemini) → Results
```

1. **Face detection** ([face_utils.py](face_utils.py)) — OpenCV's Haar cascade locates the largest face, adds 25% padding, and resizes the crop to 512×512 px
2. **VLM analysis** ([analyzer.py](analyzer.py)) — The cropped face is Base64-encoded and sent to Gemini with a structured prompt that reasons about skin undertone, depth, and colour harmony
3. **UI** ([pages/app.py](pages/app.py)) — Streamlit renders the season banner, hex colour swatches, outfit/avoid chips, and makeup tip cards

## Output

| Field | Description |
|---|---|
| `season` | Classify as Spring, Summer, Autumn, or Winter based on undertone, contrast (light vs deep), and chroma (soft vs vivid), reflecting overall color harmony. |
| `undertone` | Determine warm, cool, or neutral by analyzing skin hue bias (yellow/golden, pink/blue, or balanced), supported by lighting response and visible cues. |
| `palette` | Provide exactly 6 HEX colors optimized for the user’s undertone and season, including a mix of neutrals and accents suited to their contrast level. |
| `outfit_colors` | Recommend wearable color groups (e.g., earthy, pastel, jewel, muted) that enhance skin clarity and maintain visual balance across outfits. |
| `avoid_colors` | Identify color types that clash with undertone or contrast, including tones that make the skin appear dull, washed out, or overly harsh. |
| `makeup_tips` | Suggest foundation undertone match and complementary blush, lip, and eye shades that enhance natural complexion without overpowering it. |
| `summary` | Provide a concise 2–3 sentence explanation of how undertone, depth, and chroma define the user’s season and why the palette works. |

## Tech stack

| Layer | Technology |
|---|---|
| UI & navigation | Streamlit multi-page app |
| Face detection | OpenCV Haar cascade |
| Image processing | OpenCV, Pillow |
| VLM | Google Gemini `gemini-2.5-flash` (`google-genai` SDK) |

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Skin-Color-Analysis
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up the Gemini API key

1. Get a free API key from [ai.google.dev](https://ai.google.dev)
2. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   ```

### 3. Run the app

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Use the sidebar to switch between the Analyzer and the LLM Report.

## Usage tips

- Use a clear, well-lit portrait with your face fully visible
- A plain or blurred background gives more accurate results
- Remove glasses if possible; avoid heavy filters or extreme lighting
- Accepts JPG, JPEG, PNG, and AVIF files

## File structure

```
Skin-Color-Analysis/
├── main.py             # Entry point — Streamlit multi-page navigation
├── analyzer.py         # VLM API call, prompt, and JSON parsing
├── face_utils.py       # OpenCV face crop and Base64 encoding
├── requirements.txt    # Python dependencies
├── .env                # API key (never commit this)
├── .gitignore
├── pages/
│   ├── app.py          # Skin tone analyzer UI
│   └── report.py       # LLM mini-project report
└── samples/            # Test portrait images
```

## Notes

- Results are AI-generated and intended as inspiration, not professional colour analysis
- The Gemini free tier allows ~20 requests/day — avoid rapid sequential requests during testing
- Run `streamlit run main.py`, not `streamlit run pages/app.py`, to keep multi-page navigation working
