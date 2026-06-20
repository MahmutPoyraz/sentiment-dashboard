import streamlit as st
import streamlit.components.v1 as components
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
    page_title="Sentiment Analysis Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #FAFAFA !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton { 
    visibility: hidden !important; 
    display: none !important;
}

.main .block-container {
    padding: 3rem 4rem !important;
    max-width: 1200px !important;
}

[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #F3F4F6 !important;
}
section[data-testid="stSidebar"] > div {
    padding: 2.5rem 1.5rem !important;
}

.page-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.05em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.page-tagline {
    font-size: 0.95rem;
    color: #6B7280;
    font-weight: 400;
}
.header-rule {
    height: 1px;
    background: linear-gradient(90deg, #E5E7EB 0%, rgba(229,231,235,0) 100%);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

.sidebar-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 0.8rem;
    margin-top: 2rem;
}

div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 2px solid #E5E7EB !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #111827 !important;
    padding: 0.3rem 0.5rem !important;
    transition: all 0.25s ease !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #4F46E5 !important;
    box-shadow: 0 4px 6px -1px rgba(79,70,229,0.1) !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.2) !important;
}

.model-pill {
    background: #FFFFFF;
    border: 1px solid #F3F4F6;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.model-pill:hover { 
    border-color: #D1D5DB; 
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    transform: translateY(-2px);
}
.model-pill-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.best-tag {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: #EEF2FF;
    color: #4F46E5;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
}
.score-chip {
    font-size: 0.7rem;
    color: #6B7280;
    background: #F9FAFB;
    border: 1px solid #F3F4F6;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #E5E7EB !important;
    gap: 2.5rem !important;
    margin-bottom: 2.5rem !important;
    padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6B7280 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 1rem 0 !important;
    border: none !important;
    transition: color 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #111827 !important; }
.stTabs [aria-selected="true"] {
    color: #111827 !important;
    border-bottom: 2px solid #111827 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

.stTextArea textarea {
    background: #FFFFFF !important;
    border: 2px solid #E5E7EB !important;
    border-radius: 16px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
    resize: none !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
.stTextArea textarea:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 4px rgba(79,70,229,0.1) !important;
}

.stButton > button[kind="primary"] {
    background: #111827 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: #374151 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1) !important;
}

/* Download button */
.stDownloadButton > button {
    background: #FFFFFF !important;
    color: #374151 !important;
    border: 2px solid #E5E7EB !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #4F46E5 !important;
    color: #4F46E5 !important;
    background: #EEF2FF !important;
    transform: translateY(-1px) !important;
}

.streamlit-expanderHeader {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    color: #374151 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(40px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 24px;
    padding: 3rem;
    margin-top: 2rem;
    box-shadow: 0 20px 25px -5px rgba(0,0,0,0.05), 0 10px 10px -5px rgba(0,0,0,0.02);
    animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.result-sentiment {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.05em;
    margin-bottom: 0.5rem;
}

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.kpi-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
    transform: translateY(-2px);
}

.section-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    color: #111827;
    margin-bottom: 1.2rem;
}

.cm-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────
PLOTLY_LIGHT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#6B7280', size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)

def light_grid(fig):
    fig.update_xaxes(gridcolor='#F3F4F6', gridwidth=1, zerolinecolor='#F3F4F6')
    fig.update_yaxes(gridcolor='#F3F4F6', gridwidth=1, zerolinecolor='#F3F4F6')
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
    cleaned = preprocess_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    conf = float(max(model.predict_proba(X)[0]))
    return pred, conf, cleaned

def top_words(reviews, sentiment=None, n=15):
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
    st.markdown('''
    <div style="margin-bottom: 2rem;">
        <span style="font-family: Plus Jakarta Sans, sans-serif; font-size: 1.6rem; font-weight: 700; color: #111827;">Settings</span>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Active Engine</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("model", list(models.keys()), label_visibility="collapsed")

    if metrics_data:
        st.markdown('<div class="sidebar-label">Model Performance</div>', unsafe_allow_html=True)
        best_f1 = max(m['f1_score'] for m in metrics_data)
        for m in metrics_data:
            is_best = abs(m['f1_score'] - best_f1) < 0.001
            best_html = '<span class="best-tag">Best</span>' if is_best else ''
            st.markdown(f"""
            <div class="model-pill">
                <div class="model-pill-name">{m['model_name']} {best_html}</div>
                <div style="display:flex;gap:0.5rem;margin-top:0.5rem;">
                    <span class="score-chip">Acc {m['accuracy']:.1%}</span>
                    <span class="score-chip">F1 {m['f1_score']:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="page-title">Sentiment Analysis Engine</div>
    <div class="page-tagline">Advanced text classification using Machine Learning models.</div>
    <div class="header-rule"></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Analyze", "Statistics", "Model Comparison", "History"
])

# ════════════════════════════════════════════════════════
# TAB 1 — Analyze
# ════════════════════════════════════════════════════════
with tab1:
    if '_pending' in st.session_state:
        st.session_state['_input_val'] = st.session_state.pop('_pending')

    col_input, col_ex = st.columns([3, 2], gap="large")

    with col_input:
        if '_input_val' in st.session_state:
            st.session_state['review_textarea'] = st.session_state.pop('_input_val')

        user_input = st.text_area(
            "input",
            placeholder="Enter text to analyze sentiment...",
            height=200,
            label_visibility="collapsed",
            key="review_textarea"
        )
        run_btn = st.button("Analyze Sentiment", type="primary")

    with col_ex:
        st.markdown('''<div style="font-family: Plus Jakarta Sans, sans-serif; font-size: 0.85rem; font-weight: 600; color: #374151; margin-bottom: 1rem;">Try an example:</div>''', unsafe_allow_html=True)
        examples = [
            ("Positive", "Absolutely love this — works flawlessly from day one. Highly recommended!"),
            ("Negative", "Terrible quality. Broke within a week. Avoid at all costs."),
            ("Positive", "Exceptional build quality and fast shipping. Will definitely order again."),
        ]
        for tag, ex in examples:
            if st.button(ex, key=f"ex_{ex[:10]}", use_container_width=True):
                st.session_state['_pending'] = ex
                st.rerun()

    current_input = st.session_state.get("review_textarea", "").strip()

    if run_btn and current_input:
        user_input = current_input
        with st.spinner("Analyzing text..."):
            sentiment, confidence, cleaned_text = predict(user_input, models[selected_model], vectorizer)
            insert_review(user_input, sentiment, confidence, selected_model)

        words_list      = user_input.split()
        word_count      = len(words_list)
        unique_words    = len(set(words_list))
        avg_word_length = sum(len(w) for w in words_list) / word_count if word_count > 0 else 0
        pct             = int(confidence * 100)
        is_pos          = sentiment == "Positive"
        bar_color       = "#10B981" if is_pos else "#EF4444"
        sent_color      = "#059669" if is_pos else "#DC2626"

        st.markdown(
            f'''<div id="analysis-result" class="result-card" style="border-left: 6px solid {bar_color};">
            <div style="font-size:0.85rem;text-transform:uppercase;letter-spacing:0.05em;font-weight:700;color:#6B7280;margin-bottom:0.5rem;">Detected Sentiment</div>
            <div class="result-sentiment" style="color:{sent_color};">{sentiment}</div>
            <div style="height:1px;background:#F3F4F6;margin:2rem 0;"></div>
            <div style="display:flex;gap:3rem;flex-wrap:wrap;">
                <div><div style="font-size:0.75rem;color:#6B7280;margin-bottom:0.5rem;">Confidence Score</div><div style="font-family:Plus Jakarta Sans,sans-serif;font-size:1.5rem;font-weight:700;color:#111827;">{confidence:.1%}</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280;margin-bottom:0.5rem;">Active Model</div><div style="font-family:Plus Jakarta Sans,sans-serif;font-size:1.2rem;font-weight:600;color:#374151;margin-top:0.3rem;">{selected_model}</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280;margin-bottom:0.5rem;">Word Count</div><div style="font-family:Plus Jakarta Sans,sans-serif;font-size:1.2rem;font-weight:600;color:#374151;margin-top:0.3rem;">{word_count}</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280;margin-bottom:0.5rem;">Unique Words</div><div style="font-family:Plus Jakarta Sans,sans-serif;font-size:1.2rem;font-weight:600;color:#374151;margin-top:0.3rem;">{unique_words}</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280;margin-bottom:0.5rem;">Avg Word Length</div><div style="font-family:Plus Jakarta Sans,sans-serif;font-size:1.2rem;font-weight:600;color:#374151;margin-top:0.3rem;">{avg_word_length:.1f} chars</div></div>
            </div>
            <div style="margin-top:2rem;"><div style="background:#F3F4F6;border-radius:99px;height:8px;overflow:hidden;"><div style="width:{pct}%;height:100%;background:{bar_color};border-radius:99px;transition:width 1s cubic-bezier(0.16,1,0.3,1);"></div></div></div>
            </div>''',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Developer Insight: Under the Hood"):
            st.markdown("**1. Text Cleaning:** The input was converted to lowercase, punctuation removed, and stopwords filtered out.")
            st.code(cleaned_text, language="text")
            st.markdown(f"**2. Vectorization & Inference:** The cleaned text was transformed using TF-IDF and classified by `{selected_model}`, yielding a {pct}% confidence score.")

        components.html(
            """<script>
                const p = window.parent.document;
                const el = p.getElementById('analysis-result');
                if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            </script>""",
            height=0, width=0
        )

    elif run_btn and not current_input:
        st.warning("Please enter text before running the analysis.")

# ════════════════════════════════════════════════════════
# TAB 2 — Statistics
# ════════════════════════════════════════════════════════
with tab2:
    reviews = fetch_all_reviews()
    if not reviews:
        st.info("No data available yet. Analyze some text first.")
    else:
        stats = fetch_sentiment_stats()
        total = sum(stats.values())
        pos   = stats.get('Positive', 0)
        neg   = stats.get('Negative', 0)

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi-card"><div style="color:#6B7280;font-size:0.85rem;margin-bottom:0.5rem;">Total Analyzed</div><div style="font-size:2rem;font-weight:700;">{total:,}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><div style="color:#6B7280;font-size:0.85rem;margin-bottom:0.5rem;">Positive</div><div style="font-size:2rem;font-weight:700;color:#10B981;">{pos:,}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><div style="color:#6B7280;font-size:0.85rem;margin-bottom:0.5rem;">Negative</div><div style="font-size:2rem;font-weight:700;color:#EF4444;">{neg:,}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><div style="color:#6B7280;font-size:0.85rem;margin-bottom:0.5rem;">Pos / Neg Ratio</div><div style="font-size:2rem;font-weight:700;">{pos/max(neg,1):.2f}x</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown('<div class="section-heading">Sentiment Distribution</div>', unsafe_allow_html=True)
            fig_donut = go.Figure(go.Pie(
                values=[pos, neg],
                labels=['Positive', 'Negative'],
                hole=0.7,
                marker=dict(colors=['#10B981', '#EF4444'], line=dict(color='#FAFAFA', width=4)),
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>%{value} records<extra></extra>'
            ))
            fig_donut.add_annotation(
                text=f"<b>{total}</b>", x=0.5, y=0.5, showarrow=False,
                font=dict(family='Plus Jakarta Sans', size=28, color='#111827'),
            )
            fig_donut.update_layout(**PLOTLY_LIGHT, height=350, showlegend=True,
                                    legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_donut, use_container_width=True)

        with cb:
            st.markdown('<div class="section-heading">Activity Timeline</div>', unsafe_allow_html=True)
            df_r = pd.DataFrame(reviews)
            df_r['classified_at'] = pd.to_datetime(df_r['classified_at'])
            df_r['ts'] = df_r['classified_at'].dt.floor('min')
            trend = df_r.groupby(['ts', 'predicted_sentiment']).size().reset_index(name='count')
            fig_line = px.line(
                trend, x='ts', y='count', color='predicted_sentiment',
                color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444'},
                markers=True
            )
            fig_line.update_traces(line=dict(width=3))
            fig_line.update_layout(**PLOTLY_LIGHT, height=350, xaxis_title='',
                                   yaxis_title='', legend=dict(orientation='h', y=1.1, x=0, title_text=''))
            light_grid(fig_line)
            st.plotly_chart(fig_line, use_container_width=True)

        # ── Word Cloud ───────────────────────────────────
        st.markdown('<div style="height:1px;background:#F3F4F6;margin:2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Word Cloud</div>', unsafe_allow_html=True)

        wc_col1, wc_col2 = st.columns([1, 3])
        with wc_col1:
            wc_filter = st.radio(
                "Filter by sentiment",
                ["All", "Positive", "Negative"],
                label_visibility="collapsed"
            )

        corpus = ' '.join(
            r['review_text'] for r in reviews
            if wc_filter == "All" or r['predicted_sentiment'] == wc_filter
        )

        if corpus.strip():
            color_map = {
                'Positive': '#10B981',
                'Negative': '#EF4444',
                'All':      '#4F46E5'
            }
            wc_colormap = 'Greens' if wc_filter == 'Positive' else ('Reds' if wc_filter == 'Negative' else 'Blues')

            wc = WordCloud(
                width=1000, height=380,
                background_color='#FFFFFF',
                colormap=wc_colormap,
                max_words=100,
                collocations=False,
                prefer_horizontal=0.85
            ).generate(corpus)

            fig_wc, ax = plt.subplots(figsize=(13, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            fig_wc.patch.set_facecolor('#FFFFFF')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc, use_container_width=True)
        else:
            st.caption("Not enough data to generate a word cloud.")

# ════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════
with tab3:
    if not metrics_data:
        st.info("Train your models to view comparative metrics.")
    else:
        df_m = pd.DataFrame(metrics_data)

        st.markdown('<div class="section-heading">Performance Metrics</div>', unsafe_allow_html=True)
        metric_cols   = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        bar_colors    = ['#4F46E5', '#06B6D4', '#10B981', '#F59E0B']

        fig_bar = go.Figure()
        for col, label, color in zip(metric_cols, metric_labels, bar_colors):
            fig_bar.add_trace(go.Bar(
                name=label, x=df_m['model_name'], y=df_m[col],
                marker=dict(color=color, opacity=0.9),
                text=[f"{v:.3f}" for v in df_m[col]],
                textposition='outside',
                textfont=dict(size=11, color='#6B7280'),
                hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y:.4f}<extra></extra>'
            ))
        fig_bar.update_layout(
            **PLOTLY_LIGHT, barmode='group', height=450,
            yaxis=dict(range=[0.8, 1.0], tickformat='.0%'),
            legend=dict(orientation='h', y=1.1, x=0)
        )
        light_grid(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Confusion Matrices ───────────────────────────
        st.markdown('<div style="height:1px;background:#F3F4F6;margin:2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Confusion Matrices</div>', unsafe_allow_html=True)

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
                    st.markdown(f'<div class="cm-label">{name}</div>', unsafe_allow_html=True)
                    st.image(fpath, use_column_width=True)
            else:
                with col:
                    st.caption(f"{name}: matrix not found. Run training first.")

        # ── Summary Table ────────────────────────────────
        st.markdown('<div style="height:1px;background:#F3F4F6;margin:2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Summary Table</div>', unsafe_allow_html=True)

        df_disp = df_m[['model_name'] + metric_cols].copy()
        df_disp.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            df_disp[c] = df_disp[c].map('{:.4f}'.format)
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 4 — History
# ════════════════════════════════════════════════════════
with tab4:
    reviews = fetch_all_reviews()
    if not reviews:
        st.info("Database is empty. Analyze some reviews to populate the history.")
    else:
        # ── Filters + Export row ─────────────────────────
        f1, f2, f3 = st.columns([1, 1, 2], gap="medium")
        with f1:
            fs = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative"])
        with f2:
            fm = st.selectbox("Filter by Model", ["All"] + list(models.keys()))

        df_h = pd.DataFrame(reviews)
        if fs != "All": df_h = df_h[df_h['predicted_sentiment'] == fs]
        if fm != "All": df_h = df_h[df_h['model_used'] == fm]

        df_show = df_h[['review_text', 'predicted_sentiment',
                          'confidence_score', 'model_used', 'classified_at']].copy()
        df_show.columns = ['Text', 'Sentiment', 'Confidence', 'Model', 'Timestamp']
        df_show['Confidence'] = df_show['Confidence'].map('{:.2%}'.format)

        # ── Export button ────────────────────────────────
        with f3:
            st.markdown('<div style="padding-top:1.9rem;"></div>', unsafe_allow_html=True)
            csv_data = df_show.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Export {len(df_show):,} records as CSV",
                data=csv_data,
                file_name=f"sentiment_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown(f'<div style="font-size:0.78rem;color:#9CA3AF;margin:0.8rem 0;">Showing {len(df_show):,} of {len(reviews):,} records</div>', unsafe_allow_html=True)

        st.dataframe(df_show, use_container_width=True, hide_index=True, height=580)