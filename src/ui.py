"""Shared look and feel: fonts, background, cards, hero headings."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

html, body, .stApp, button, input, textarea, select {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, .hero h1, [data-testid="stMetricValue"] {
    font-family: 'Manrope', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(900px 420px at 50% -100px, rgba(14,154,167,0.14), transparent 70%),
        #F4F7FA;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 1150px !important; }

/* Hero heading */
.hero { text-align: center; padding: 1rem 0 1.8rem; }
.hero .eyebrow {
    display: inline-block; color: #0E7C86; background: rgba(14,154,167,0.10);
    border: 1px solid rgba(14,154,167,0.25); border-radius: 999px;
    padding: 0.3rem 0.9rem; font-weight: 700; font-size: 0.72rem;
    letter-spacing: 0.16em; text-transform: uppercase;
}
.hero h1 {
    font-size: clamp(2.2rem, 5vw, 3.6rem) !important; font-weight: 800 !important;
    letter-spacing: -0.035em; line-height: 1.05 !important; color: #0B1F3A;
    margin: 0.9rem 0 0.8rem !important; padding: 0 !important;
}
.hero h1 span { color: #0E9AA7; }
.hero p { color: #5B6B7F; max-width: 640px; margin: 0 auto; font-size: 1.05rem; }

/* Section headings */
h2, h3 { color: #0B1F3A; letter-spacing: -0.02em; font-weight: 700 !important; }

/* Cards: forms, KPI metrics, tables */
[data-testid="stForm"] {
    background: #FFFFFF; border: 1px solid #DDE5EE; border-radius: 18px;
    padding: 1.6rem; box-shadow: 0 10px 40px rgba(11, 31, 58, 0.07);
}
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #DDE5EE; border-left: 4px solid #0E9AA7;
    border-radius: 14px; padding: 1rem 1.2rem;
    box-shadow: 0 6px 24px rgba(11, 31, 58, 0.05);
}
[data-testid="stMetricLabel"] p {
    color: #5B6B7F; font-weight: 600; font-size: 0.76rem;
    text-transform: uppercase; letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] { font-weight: 800; color: #0B1F3A; }
[data-testid="stDataFrame"] {
    border: 1px solid #DDE5EE; border-radius: 14px; overflow: hidden;
    box-shadow: 0 6px 24px rgba(11, 31, 58, 0.05);
}
[data-testid="stAlert"] { border-radius: 12px; }

/* Inputs */
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div {
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border: none; border-radius: 10px; padding: 0.6rem 1.5rem; font-weight: 600;
    background: linear-gradient(135deg, #0B1F3A, #0E7C86); color: #FFFFFF;
    box-shadow: 0 8px 20px rgba(11, 31, 58, 0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button p, [data-testid="stFormSubmitButton"] > button p { color: #FFFFFF; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px); box-shadow: 0 12px 26px rgba(14, 154, 167, 0.35);
    color: #FFFFFF;
}

/* Tabs */
.stTabs [data-baseweb="tab"] { font-weight: 600; }

/* Navy sidebar */
section[data-testid="stSidebar"] { background: #0B1F3A; border-right: none; }
section[data-testid="stSidebar"] :is(p, span, a, label) { color: #E6EDF7; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
section[data-testid="stSidebar"] .stButton > button {
    background: #0E9AA7; box-shadow: none;
}
</style>
"""


def apply_theme():
    """Inject the shared CSS. Call once near the top of app.py."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, accent: str, subtitle: str = ""):
    """Big centered heading: navy first line, teal second line."""
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f"<h1>{title}<br><span>{accent}</span></h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )