import streamlit as st

pg = st.navigation([
    st.Page("pages/app.py", title="🎨 Skin Tone Analyzer"),
    st.Page("pages/report.py", title="📊 LLM Report"),
])
pg.run()
