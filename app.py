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
    layout="wide",
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
def render_color_chips(color_items: list, style: str = "normal"):
    """Render colour names as pill-shaped chips with optional color dots. style='avoid' renders in red."""
    bg = "#FFEBEE" if style == "avoid" else "#F0F4F8"
    text = "#B71C1C" if style == "avoid" else "#1A2A3A"
    border = "#FFCDD2" if style == "avoid" else "#CBD5E0"
    
    chips_html = []
    for item in color_items:
        if isinstance(item, dict):
            name = item.get("name", "Unknown")
            hex_val = item.get("hex", "")
            dot_html = f'<span style="display:inline-block; width:12px; height:12px; background:{hex_val}; border-radius:50%; margin-right:6px; vertical-align:-1px; border:1px solid rgba(0,0,0,0.1);"></span>' if hex_val else ""
        else:
            name = str(item)
            dot_html = ""
            
        chips_html.append(
            f'<span style="background:{bg};color:{text};border:1px solid {border};'
            f'padding:4px 12px;border-radius:20px;font-size:13px;margin:3px;'
            f'display:inline-block;">{dot_html}{name}</span>'
        )
        
    st.markdown(f'<div style="margin:8px 0">{" ".join(chips_html)}</div>', unsafe_allow_html=True)


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

    st.divider()

    # Create two columns for layout: left for image, right for analysis
    img_col, analysis_col = st.columns([1, 1.5])

    with img_col:
        st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
        # Optional: show the face crop
        # st.image(face_crop, caption="Detected Face", width=150)

    with analysis_col:
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
                margin:12px 0 20px 0;
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

        # Undertone tag & Summary
        undertone = result.get("undertone", "").capitalize()
        if undertone:
            st.markdown(f"**Undertone:** `{undertone}`")

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
