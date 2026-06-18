import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_text
from src.database import (
    init_db, insert_review, fetch_all_reviews,
    fetch_sentiment_stats, fetch_model_metrics
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

st.set_page_config(
    page_title="Lexara — Sentiment Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f8f7f4 !important;
    color: #1a1a1a !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: #f8f7f4 !important;
}

/* Hide chrome */
#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton { 
    visibility: hidden !important; 
    display: none !important;
}

/* Main padding */
.main .block-container {
    padding: 3rem 3rem 3rem 3rem !important;
    max-width: 1300px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e4dd !important;
}
section[data-testid="stSidebar"] > div {
    padding: 2rem 1.4rem !important;
}

/* ── Logo area ── */
.lexara-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #eceae5;
}
.logo-mark {
    width: 38px;
    height: 38px;
    background: #4f46e5;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.logo-letter {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8f7f4;
}
.logo-name {
    font-family: 'Sora', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.03em;
}

/* ── Sidebar labels ── */
.sidebar-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aaa9a5;
    margin-bottom: 0.7rem;
    margin-top: 1.6rem;
}

/* ── Model cards in sidebar ── */
.model-pill {
    background: #f8f7f4;
    border: 1px solid #eceae5;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.45rem;
    transition: border-color 0.2s;
}
.model-pill:hover { border-color: #c8c4bc; }
.model-pill-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #2a2a2a;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.best-tag {
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: #4f46e5;
    color: #ffffff;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}
.model-scores {
    display: flex;
    gap: 0.5rem;
}
.score-chip {
    font-size: 0.68rem;
    color: #888;
    background: #ffffff;
    border: 1px solid #e8e4dd;
    padding: 0.1rem 0.45rem;
    border-radius: 5px;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #f8f7f4 !important;
    border: 1px solid #ddd9d0 !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Page header ── */
.page-header {
    margin-bottom: 2.5rem;
}
.page-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.04em;
    line-height: 1;
}
.page-tagline {
    font-size: 0.88rem;
    color: #999;
    margin-top: 0.4rem;
    font-weight: 400;
}
.header-rule {
    height: 1px;
    background: #e8e4dd;
    margin-top: 1.8rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border: 1px solid #e8e4dd !important;
    border-radius: 12px !important;
    padding: 0.3rem !important;
    gap: 0.15rem !important;
    margin-bottom: 2.2rem !important;
    display: inline-flex !important;
    width: auto !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #888 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.3rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
    white-space: nowrap !important;
}
.stTabs [aria-selected="true"] {
    background: #4f46e5 !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #ddd9d0 !important;
    border-radius: 14px !important;
    color: #1a1a1a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    padding: 1.2rem 1.4rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #bbb !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: #4f46e5 !important;
    color: #f8f7f4 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.8rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #4338ca !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.25) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Example buttons override */
.ex-btn .stButton > button {
    background: #ffffff !important;
    color: #555 !important;
    border: 1px solid #e8e4dd !important;
    font-weight: 400 !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 0.9rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 9px !important;
}
.ex-btn .stButton > button:hover {
    background: #f0ede8 !important;
    border-color: #ccc8c0 !important;
    transform: none !important;
    box-shadow: none !important;
    color: #222 !important;
}

/* ── Result animation ── */
@keyframes resultAppear {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes sentimentReveal {
    0%   { opacity: 0; transform: scale(0.7) rotate(-4deg); }
    60%  { transform: scale(1.06) rotate(1deg); }
    100% { opacity: 1; transform: scale(1) rotate(0deg); }
}
@keyframes barGrow {
    from { width: 0; }
    to   { width: var(--w); }
}

.result-card {
    background: #ffffff;
    border: 1px solid #e8e4dd;
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-top: 1.8rem;
    animation: resultAppear 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}
.result-sentiment {
    display: inline-block;
    font-family: 'Sora', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    animation: sentimentReveal 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) 0.1s both;
}
.result-positive { color: #16a34a; }
.result-negative { color: #dc2626; }

.result-divider {
    height: 1px;
    background: #eceae5;
    margin: 1.4rem 0;
}
.result-meta {
    display: flex;
    gap: 3rem;
    flex-wrap: wrap;
}
.meta-item { display: flex; flex-direction: column; gap: 0.3rem; }
.meta-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #bbb;
}
.meta-value {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a1a;
}

.conf-wrap { margin-top: 1.4rem; }
.conf-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #999;
    margin-bottom: 0.45rem;
}
.conf-track {
    background: #f0ede8;
    border-radius: 99px;
    height: 7px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    animation: barGrow 0.9s cubic-bezier(0.4,0,0.2,1) 0.3s both;
}
.fill-pos { background: #16a34a; --w: var(--conf-pct); }
.fill-neg { background: #dc2626; --w: var(--conf-pct); }

/* ── Section heading ── */
.sec-heading {
    font-family: 'Sora', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1a1a;
    letter-spacing: -0.01em;
    margin-bottom: 1.2rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid #eceae5;
}

/* ── Stat cards ── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.kpi-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e8e4dd;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.07); }
.kpi-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.03em;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #999;
    margin-top: 0.35rem;
}

/* ── Word bars ── */
.word-bars { display: flex; flex-direction: column; gap: 0.45rem; }
.wbar-row { display: flex; align-items: center; gap: 0.8rem; }
.wbar-word {
    font-size: 0.78rem;
    color: #555;
    width: 90px;
    flex-shrink: 0;
    text-align: right;
    font-weight: 500;
}
.wbar-track {
    flex: 1;
    background: #f0ede8;
    border-radius: 4px;
    height: 7px;
    overflow: hidden;
}
.wbar-fill-pos {
    height: 100%;
    background: #16a34a;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.wbar-fill-neg {
    height: 100%;
    background: #dc2626;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.wbar-count {
    font-size: 0.7rem;
    color: #bbb;
    width: 32px;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
}

/* ── Glass card ── */
.g-card {
    background: #ffffff;
    border: 1px solid #e8e4dd;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}

/* ── Divider ── */
.page-divider {
    height: 1px;
    background: #eceae5;
    margin: 2rem 0;
}

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e8e4dd !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: #aaa !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
}
[data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

/* ── Radio ── */
.stRadio label { color: #555 !important; font-size: 0.83rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #555 !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #e8e4dd !important; border-radius: 12px !important; }
.stDataFrame thead th {
    background: #f8f7f4 !important;
    color: #999 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f8f7f4; }
::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e8e4dd !important;
    border-radius: 10px !important;
    color: #555 !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────
PLOTLY_LIGHT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#999', size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)

def light_grid(fig):
    fig.update_xaxes(gridcolor='#eceae5', gridwidth=1,
                     zerolinecolor='#eceae5',
                     tickfont=dict(color='#aaa', size=10))
    fig.update_yaxes(gridcolor='#eceae5', gridwidth=1,
                     zerolinecolor='#eceae5',
                     tickfont=dict(color='#aaa', size=10))
    return fig

@st.cache_resource
def load_models():
    vect = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Naive Bayes':         'naive_bayes.pkl',
        'SVM':                 'svm.pkl',
    }
    return vect, {n: joblib.load(os.path.join(MODELS_DIR, f))
                  for n, f in files.items()
                  if os.path.exists(os.path.join(MODELS_DIR, f))}

@st.cache_data(ttl=30)
def cached_metrics():
    return fetch_model_metrics()

def predict(text, model, vectorizer):
    X = vectorizer.transform([preprocess_text(text)])
    pred = model.predict(X)[0]
    conf = float(max(model.predict_proba(X)[0]))
    return pred, conf

def top_words(reviews, sentiment=None, n=12):
    STOP = {'the','and','for','this','was','are','with','have','that',
            'not','but','its','it','from','you','all','has','one','they',
            'been','more','very','just','my','so','an','at','by','be',
            'or','as','on','to','is','in','of','a','i','we','he','she',
            'would','could','should','their','there','then','than','when'}
    texts = [r['review_text'] for r in reviews
             if sentiment is None or r['predicted_sentiment'] == sentiment]
    words = []
    for t in texts:
        words.extend(w for w in re.findall(r'\b[a-z]{3,}\b', t.lower()) if w not in STOP)
    return Counter(words).most_common(n)

# ── Init ─────────────────────────────────────────────────
init_db()
vectorizer, models = load_models()
metrics_data = cached_metrics()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lexara-logo">
        <div class="logo-mark"><span class="logo-letter">L</span></div>
        <span class="logo-name">Lexara</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Active Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("model", list(models.keys()),
                                   label_visibility="collapsed")

    if metrics_data:
        st.markdown('<div class="sidebar-label">Model Performance</div>',
                    unsafe_allow_html=True)
        best_f1 = max(m['f1_score'] for m in metrics_data)
        for m in metrics_data:
            is_best = abs(m['f1_score'] - best_f1) < 0.001
            best_html = '<span class="best-tag">Best</span>' if is_best else ''
            st.markdown(f"""
            <div class="model-pill">
                <div class="model-pill-name">{m['model_name']} {best_html}</div>
                <div class="model-scores">
                    <span class="score-chip">Acc {m['accuracy']:.1%}</span>
                    <span class="score-chip">F1 {m['f1_score']:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">Lexara</div>
    <div class="page-tagline">Sentiment Analysis · Amazon Reviews · 200k training samples</div>
    <div class="header-rule"></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Analyze Review", "Statistics & Trends",
    "Model Comparison", "Review History"
])

# ════════════════════════════════════════════════════════
# TAB 1 — Analyze
# ════════════════════════════════════════════════════════
with tab1:
    # Handle example injection before widgets render
    if '_pending' in st.session_state:
        st.session_state['_input_val'] = st.session_state.pop('_pending')

    col_input, col_ex = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown('''
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#aaa;margin-bottom:0.5rem;">
            Review Input
        </div>
        ''', unsafe_allow_html=True)

        if '_input_val' in st.session_state:
            st.session_state['review_textarea'] = st.session_state.pop('_input_val')

        user_input = st.text_area(
            "input",
            placeholder="Paste or type a product review. The model will classify its sentiment and save it to the database.",
            height=240,
            label_visibility="collapsed",
            key="review_textarea"
        )

        st.markdown('''
        <div style="font-size:0.72rem;color:#bbb;margin-top:0.3rem;margin-bottom:1rem;">
            Supports any language — model is trained on English reviews.
        </div>
        ''', unsafe_allow_html=True)

        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    with col_ex:
        st.markdown('''
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#aaa;margin-bottom:0.5rem;">
            Quick Examples
        </div>
        <div style="font-size:0.75rem;color:#bbb;margin-bottom:1rem;">
            Click any example to load it into the input field.
        </div>
        ''', unsafe_allow_html=True)

        examples = [
            ("Positive", "Absolutely love this — works flawlessly from day one."),
            ("Negative", "Terrible quality. Broke within a week. Avoid."),
            ("Positive", "Best purchase I have made this year. Highly recommend."),
            ("Negative", "Complete waste of money. Very disappointed."),
            ("Positive", "Exceptional build quality and fast shipping. Will order again."),
        ]
        for tag, ex in examples:
            tag_color = "#16a34a" if tag == "Positive" else ("#dc2626" if tag == "Negative" else "#999")
            tag_bg    = "#f0fdf4" if tag == "Positive" else ("#fff1f2" if tag == "Negative" else "#f5f5f5")
            st.markdown(f'''
            <div style="margin-bottom:0.5rem;">
                <span style="font-size:0.6rem;font-weight:700;letter-spacing:0.1em;
                             text-transform:uppercase;color:{tag_color};
                             background:{tag_bg};padding:0.15rem 0.5rem;
                             border-radius:4px;">{tag}</span>
            </div>''', unsafe_allow_html=True)
            st.markdown('<div class="ex-btn">', unsafe_allow_html=True)
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:15]}"):
                st.session_state['_pending'] = ex
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Read the actual current value from session state
    current_input = st.session_state.get("review_textarea", "").strip()

    if run_btn and current_input:
        user_input = current_input
        with st.spinner("Analyzing…"):
            sentiment, confidence = predict(user_input, models[selected_model], vectorizer)
            insert_review(user_input, sentiment, confidence, selected_model)

        pct = int(confidence * 100)
        is_pos     = sentiment == "Positive"
        bar_color  = "#16a34a" if is_pos else "#dc2626"
        sent_color = "#16a34a" if is_pos else "#dc2626"

        accent_border = '#e8f5e9' if is_pos else '#fef2f2'
        st.markdown(
            f'''<div style="background:#ffffff;border:1px solid {accent_border};border-left:4px solid {bar_color};border-radius:18px;padding:2.2rem 2.4rem;margin-top:1.8rem;">'''
            + f'''<div style="font-family:Sora,sans-serif;font-size:2.6rem;font-weight:700;letter-spacing:-0.04em;color:{sent_color};">{sentiment}</div>'''
            + '''<div style="height:1px;background:#eceae5;margin:1.4rem 0;"></div>'''
            + f'''<div style="display:flex;gap:3rem;flex-wrap:wrap;">'''
            + f'''<div><div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#bbb;margin-bottom:0.3rem;">Confidence</div><div style="font-family:Sora,sans-serif;font-size:1.05rem;font-weight:600;color:#1a1a1a;">{confidence:.1%}</div></div>'''
            + f'''<div><div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#bbb;margin-bottom:0.3rem;">Model</div><div style="font-family:Sora,sans-serif;font-size:1.05rem;font-weight:600;color:#1a1a1a;">{selected_model}</div></div>'''
            + f'''<div><div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#bbb;margin-bottom:0.3rem;">Characters</div><div style="font-family:Sora,sans-serif;font-size:1.05rem;font-weight:600;color:#1a1a1a;">{len(user_input)}</div></div>'''
            + f'''<div><div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#bbb;margin-bottom:0.3rem;">Words</div><div style="font-family:Sora,sans-serif;font-size:1.05rem;font-weight:600;color:#1a1a1a;">{len(user_input.split())}</div></div>'''
            + '''</div>'''
            + f'''<div style="margin-top:1.4rem;"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#999;margin-bottom:0.45rem;"><span>Confidence score</span><span style="color:#1a1a1a;font-weight:600;">{pct}%</span></div><div style="background:#f0ede8;border-radius:99px;height:7px;overflow:hidden;"><div style="width:{pct}%;height:100%;background:{bar_color};border-radius:99px;"></div></div></div>'''
            + '''</div>''',
            unsafe_allow_html=True
        )

    elif run_btn and not current_input:
        st.warning("Please enter a review before running analysis.")

# ════════════════════════════════════════════════════════
# TAB 2 — Statistics
# ════════════════════════════════════════════════════════
with tab2:
    reviews = fetch_all_reviews()
    if not reviews:
        st.markdown("""
        <div class="g-card" style="text-align:center;padding:3rem;color:#bbb;">
            No data yet — submit reviews in the Analyze tab.
        </div>""", unsafe_allow_html=True)
    else:
        stats = fetch_sentiment_stats()
        total = sum(stats.values())
        pos   = stats.get('Positive', 0)
        neg   = stats.get('Negative', 0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Reviews", f"{total:,}")
        k2.metric("Positive", f"{pos:,}", f"{pos/total:.1%}")
        k3.metric("Negative", f"{neg:,}", f"{neg/total:.1%}")
        k4.metric("Pos / Neg", f"{pos/max(neg,1):.2f}×")

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown('<div class="sec-heading">Sentiment Distribution</div>',
                        unsafe_allow_html=True)
            fig_donut = go.Figure(go.Pie(
                values=[pos, neg],
                labels=['Positive', 'Negative'],
                hole=0.6,
                marker=dict(
                    colors=['#16a34a', '#dc2626'],
                    line=dict(color='#f8f7f4', width=3)
                ),
                textinfo='label+percent',
                textfont=dict(family='Inter', size=12, color='#444'),
                hovertemplate='<b>%{label}</b><br>%{value} reviews<extra></extra>'
            ))
            fig_donut.add_annotation(
                text=f"<b>{total}</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(family='Sora', size=22, color='#1a1a1a'),
            )
            fig_donut.update_layout(**PLOTLY_LIGHT, height=300, showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)

        with cb:
            st.markdown('<div class="sec-heading">Classification Timeline</div>',
                        unsafe_allow_html=True)
            df_r = pd.DataFrame(reviews)
            df_r['classified_at'] = pd.to_datetime(df_r['classified_at'])
            df_r['ts'] = df_r['classified_at'].dt.floor('min')
            trend = (df_r.groupby(['ts','predicted_sentiment'])
                         .size().reset_index(name='count'))
            fig_line = px.line(
                trend, x='ts', y='count',
                color='predicted_sentiment',
                color_discrete_map={'Positive':'#16a34a','Negative':'#dc2626'},
                markers=True
            )
            fig_line.update_traces(line=dict(width=2.5))
            fig_line.update_layout(
                **PLOTLY_LIGHT, height=300,
                xaxis_title='', yaxis_title='',
                legend=dict(orientation='h', y=1.05, x=0,
                            font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                            title_text='')
            )
            light_grid(fig_line)
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

        wf1, wf2 = st.columns(2, gap="large")

        def render_bars(word_counts, kind):
            if not word_counts:
                st.caption("Not enough data.")
                return
            max_c = word_counts[0][1]
            fill = "wbar-fill-pos" if kind == "pos" else "wbar-fill-neg"
            rows = "".join(f"""
            <div class="wbar-row">
                <span class="wbar-word">{w}</span>
                <div class="wbar-track">
                    <div class="{fill}" style="width:{int(c/max_c*100)}%"></div>
                </div>
                <span class="wbar-count">{c}</span>
            </div>""" for w, c in word_counts)
            st.markdown(f'<div class="word-bars">{rows}</div>', unsafe_allow_html=True)

        with wf1:
            st.markdown('<div class="sec-heading">Top Words — Positive</div>',
                        unsafe_allow_html=True)
            render_bars(top_words(reviews, 'Positive'), 'pos')

        with wf2:
            st.markdown('<div class="sec-heading">Top Words — Negative</div>',
                        unsafe_allow_html=True)
            render_bars(top_words(reviews, 'Negative'), 'neg')

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-heading">Word Cloud</div>', unsafe_allow_html=True)

        wc_filter = st.radio("Filter", ["All","Positive","Negative"],
                             horizontal=True, label_visibility="collapsed")
        corpus = ' '.join(
            r['review_text'] for r in reviews
            if wc_filter == "All" or r['predicted_sentiment'] == wc_filter
        )
        if corpus.strip():
            cmap = 'Greens' if wc_filter == 'Positive' else ('Reds' if wc_filter == 'Negative' else 'Greys')
            wc = WordCloud(
                width=1000, height=360,
                background_color='#ffffff',
                colormap=cmap,
                max_words=100,
                collocations=False
            ).generate(corpus)
            fig_wc, ax = plt.subplots(figsize=(13, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            fig_wc.patch.set_facecolor('#ffffff')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════
with tab3:
    if not metrics_data:
        st.info("Run `python -m src.train` to populate model metrics.")
    else:
        df_m = pd.DataFrame(metrics_data)

        st.markdown('<div class="sec-heading">Performance Across All Metrics</div>',
                    unsafe_allow_html=True)
        metric_cols   = ['accuracy','precision','recall','f1_score']
        metric_labels = ['Accuracy','Precision','Recall','F1-Score']
        colors = ['#4f46e5','#06b6d4','#10b981','#f59e0b']

        fig_bar = go.Figure()
        for col, label, color in zip(metric_cols, metric_labels, colors):
            fig_bar.add_trace(go.Bar(
                name=label,
                x=df_m['model_name'],
                y=df_m[col],
                marker=dict(color=color, opacity=0.9, line=dict(width=0)),
                text=[f"{v:.3f}" for v in df_m[col]],
                textposition='outside',
                textfont=dict(size=10, color='#888'),
                hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y:.4f}<extra></extra>'
            ))
        fig_bar.update_layout(
            **PLOTLY_LIGHT, barmode='group', height=400,
            yaxis=dict(range=[0.82,1.0], tickformat='.0%',
                       gridcolor='#eceae5', title=''),
            xaxis=dict(title=''),
            legend=dict(orientation='h', y=1.05, x=0,
                        font=dict(size=10, color='#888'),
                        bgcolor='rgba(0,0,0,0)', title_text=''),
            bargap=0.25, bargroupgap=0.06
        )
        light_grid(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-heading">Confusion Matrices</div>',
                    unsafe_allow_html=True)

        cm_map = {
            'Logistic Regression': 'logistic_regression_confusion_matrix.png',
            'Naive Bayes':         'naive_bayes_confusion_matrix.png',
            'SVM':                 'svm_confusion_matrix.png',
        }
        cc1, cc2, cc3 = st.columns(3, gap="large")
        for col, (name, fname) in zip([cc1, cc2, cc3], cm_map.items()):
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.exists(fpath):
                with col:
                    st.markdown(f'<div style="font-size:0.72rem;font-weight:700;color:#aaa;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;">{name}</div>',
                                unsafe_allow_html=True)
                    st.image(fpath)

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-heading">Capability Radar</div>',
                    unsafe_allow_html=True)

        cats   = ['Accuracy','Precision','Recall','F1-Score']
        rcolors      = ['#4f46e5', '#06b6d4', '#f59e0b']
        rcolors_fill = ['rgba(79,70,229,0.08)', 'rgba(6,182,212,0.08)', 'rgba(245,158,11,0.08)']
        fig_radar = go.Figure()
        for i, row in df_m.iterrows():
            vals = [row['accuracy'],row['precision'],row['recall'],row['f1_score']]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself',
                fillcolor=rcolors_fill[i % len(rcolors_fill)],
                line=dict(color=rcolors[i % len(rcolors)], width=2),
                name=row['model_name'],
                hovertemplate='<b>%{theta}</b>: %{r:.4f}<extra>'+row['model_name']+'</extra>'
            ))
        fig_radar.update_layout(
            **PLOTLY_LIGHT, height=400,
            polar=dict(
                bgcolor='#ffffff',
                radialaxis=dict(visible=True, range=[0.85,1.0],
                                tickformat='.0%', gridcolor='#eceae5',
                                tickfont=dict(size=9,color='#bbb'),
                                linecolor='#eceae5'),
                angularaxis=dict(gridcolor='#eceae5',linecolor='#eceae5',
                                 tickfont=dict(size=10,color='#888'))
            ),
            legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center',
                        font=dict(size=10,color='#888'),
                        bgcolor='rgba(0,0,0,0)', title_text='')
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-heading">Summary Table</div>',
                    unsafe_allow_html=True)
        df_disp = df_m[['model_name']+metric_cols].copy()
        df_disp.columns = ['Model','Accuracy','Precision','Recall','F1-Score']
        for c in ['Accuracy','Precision','Recall','F1-Score']:
            df_disp[c] = df_disp[c].map('{:.4f}'.format)
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 4 — History
# ════════════════════════════════════════════════════════
with tab4:
    reviews = fetch_all_reviews()
    if not reviews:
        st.markdown("""
        <div class="g-card" style="text-align:center;padding:3rem;color:#bbb;">
            No reviews in the database yet.
        </div>""", unsafe_allow_html=True)
    else:
        f1, f2, _ = st.columns([1,1,2], gap="medium")
        with f1:
            fs = st.selectbox("Sentiment", ["All","Positive","Negative"])
        with f2:
            fm = st.selectbox("Model", ["All"] + list(models.keys()))

        df_h = pd.DataFrame(reviews)
        if fs != "All": df_h = df_h[df_h['predicted_sentiment'] == fs]
        if fm != "All": df_h = df_h[df_h['model_used'] == fm]

        st.markdown(f'<div style="font-size:0.72rem;color:#bbb;margin-bottom:0.8rem;">Showing {len(df_h):,} of {len(reviews):,} reviews</div>',
                    unsafe_allow_html=True)

        df_show = df_h[['review_text','predicted_sentiment',
                          'confidence_score','model_used','classified_at']].copy()
        df_show.columns = ['Review','Sentiment','Confidence','Model','Analyzed At']
        df_show['Confidence'] = df_show['Confidence'].map('{:.2%}'.format)
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=520)