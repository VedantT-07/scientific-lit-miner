"""
app.py  ──  Scientific Literature Mining & Research Gap Discovery
═══════════════════════════════════════════════════════════════════
CEP 2 | Data Analysis & Visualisation Lab Project
Built with Streamlit · scikit-learn · gensim · Plotly · NetworkX
"""

import os
import sys
import time
import logging
import textwrap
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st

# ─── Page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="LitMine – Research Gap Discovery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "CEP 2 – Scientific Literature Mining & Research Gap Discovery\nBuilt with Streamlit + scikit-learn + Plotly"
    },
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Core typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Keep header visible so sidebar controls work */
header[data-testid="stHeader"] {
    background: transparent !important;
    visibility: visible !important;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #E6EDF3 !important;
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    z-index: 1000000 !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] svg {
    color: #E6EDF3 !important;
    fill: #E6EDF3 !important;
    visibility: visible !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid #30363D;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Main area */
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 10px;
    border-bottom: 2px solid #2E86AB;
    padding-bottom: 8px; margin: 2rem 0 1rem 0;
}
.section-header h2 { margin: 0; font-size: 1.3rem; font-weight: 600; color: #E6EDF3; }
.section-badge {
    background: #2E86AB; color: white;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: .08em;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 1rem 0; }

/* Topic card */
.topic-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
    border-left: 4px solid #2E86AB;
}
.topic-card h4 { margin: 0 0 6px 0; color: #E6EDF3; font-size: 0.95rem; }
.topic-card p  { margin: 0; color: #8b949e; font-size: 0.83rem; }

/* Gap card */
.gap-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 8px; padding: 16px 20px; margin: 10px 0;
}
.gap-card.high   { border-left: 4px solid #C73E1D; }
.gap-card.medium { border-left: 4px solid #F18F01; }
.gap-card.low    { border-left: 4px solid #44BBA4; }
.gap-score { font-size: 1.6rem; font-weight: 700; }

/* Evidence chip */
.evidence-chip {
    display: inline-block; background: #21262D;
    border: 1px solid #30363D; border-radius: 4px;
    padding: 3px 10px; margin: 3px 4px 3px 0;
    font-size: 0.75rem; color: #8b949e;
}

/* Paper row */
.paper-row {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 6px; padding: 12px 16px; margin: 6px 0;
}
.paper-row .title  { font-weight: 600; color: #E6EDF3; font-size: 0.9rem; }
.paper-row .meta   { color: #8b949e; font-size: 0.78rem; margin-top: 4px; }
.paper-row .abstract{ color: #b1bac4; font-size: 0.82rem; margin-top: 6px; }

/* Info banner */
.demo-banner {
    background: #2D2A1E; border: 1px solid #F18F01;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 1rem;
    color: #F18F01; font-size: 0.85rem;
}
.success-banner {
    background: #1A2D1A; border: 1px solid #44BBA4;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 1rem;
    color: #44BBA4; font-size: 0.85rem;
}

/* Stagger animation */
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
.fadeIn { animation: fadeIn .4s ease forwards; }

/* Tabs */
[data-baseweb="tab"] { font-size: 0.85rem; font-weight: 500; }

/* Divider */
.divider { border: none; border-top: 1px solid #30363D; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
components.html("""
<script>
    const expandSidebar = () => {
        try {
            const doc = window.parent.document;
            const btn = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                        doc.querySelector('[data-testid="stSidebarCollapsedControl"]');
            if (btn) { btn.click(); }
        } catch (e) {}
    };
    setTimeout(expandSidebar, 200);
    setTimeout(expandSidebar, 800);
</script>
""", height=0)

# ─── Lazy imports ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _import_modules():
    from data_loader import load_dataset, CATEGORY_LABELS
    from nlp_engine  import (preprocess_corpus, build_tfidf, run_lda, run_nmf,
                             label_topic, extract_top_terms, build_concept_network,
                             make_cache_key)
    from visualizations import (papers_per_year, category_distribution,
                                topic_year_heatmap, topic_word_bars,
                                topic_trend_lines, top_terms_bar,
                                concept_network_plotly, model_comparison_chart,
                                gap_bubble_chart, generate_wordcloud_fig,
                                kpi_card_html, PALETTE)
    from gap_analysis import find_research_gaps
    return {
        "load_dataset":          load_dataset,
        "CATEGORY_LABELS":       CATEGORY_LABELS,
        "preprocess_corpus":     preprocess_corpus,
        "build_tfidf":           build_tfidf,
        "run_lda":               run_lda,
        "run_nmf":               run_nmf,
        "label_topic":           label_topic,
        "extract_top_terms":     extract_top_terms,
        "build_concept_network": build_concept_network,
        "make_cache_key":        make_cache_key,
        "papers_per_year":       papers_per_year,
        "category_distribution": category_distribution,
        "topic_year_heatmap":    topic_year_heatmap,
        "topic_word_bars":       topic_word_bars,
        "topic_trend_lines":     topic_trend_lines,
        "top_terms_bar":         top_terms_bar,
        "concept_network_plotly":concept_network_plotly,
        "model_comparison_chart":model_comparison_chart,
        "gap_bubble_chart":      gap_bubble_chart,
        "generate_wordcloud_fig":generate_wordcloud_fig,
        "kpi_card_html":         kpi_card_html,
        "PALETTE":               PALETTE,
        "find_research_gaps":    find_research_gaps,
    }

M = _import_modules()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0 16px 0">
        <div style="font-size:2rem">🔬</div>
        <div style="font-weight:700;font-size:1.05rem;color:#E6EDF3">LitMine</div>
        <div style="font-size:0.72rem;color:#8b949e;letter-spacing:.1em;text-transform:uppercase">Research Gap Discovery</div>
    </div>
    <hr style="border-color:#30363D;margin:0 0 1rem 0">
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Configuration**")

    n_papers = st.slider(
        "Papers to analyse",
        min_value=500, max_value=5000, value=2500, step=250,
        help="More papers → better coverage, longer processing time."
    )

    n_topics = st.slider(
        "Number of topics (k)",
        min_value=4, max_value=16, value=8, step=1,
        help="LDA/NMF topic count. 6–10 works well for most corpora."
    )

    model_choice = st.radio(
        "Topic model",
        ["LDA (Latent Dirichlet Allocation)", "NMF (Non-negative Matrix Factorisation)"],
        index=0,
        help="LDA is probabilistic; NMF is algebraic. Compare both in the Model tab."
    )

    st.markdown("<hr style='border-color:#30363D'>", unsafe_allow_html=True)
    st.markdown("**🗂 Navigation**")
    page = st.radio(
        "Go to section",
        [
            "📊 Overview & Dataset",
            "🧠 Topic Analysis",
            "📈 Trend Explorer",
            "🕸️ Concept Network",
            "⚖️ Model Comparison",
            "🔍 Research Gaps",
            "📄 Paper Search",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#30363D'>", unsafe_allow_html=True)

    force_reload = st.button("🔄 Reload Dataset", use_container_width=True,
                             help="Clear cache and re-download/re-process data.")

    st.markdown("""
    <div style="margin-top:1rem;font-size:0.72rem;color:#484F58;text-align:center">
        CEP 2 · DAV Lab Project<br>
        Data: arXiv (Cornell University / Kaggle)
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def load_data(n: int, force: bool) -> pd.DataFrame:
    return M["load_dataset"](n_papers=n, force_reload=force)


@st.cache_data(show_spinner=False, ttl=3600)
def get_preprocessed(df_hash: str, _df: pd.DataFrame) -> list:
    return M["preprocess_corpus"](_df["combined_text"].tolist(), cache_key=df_hash)


@st.cache_data(show_spinner=False, ttl=3600)
def get_tfidf(df_hash: str, _corpus: list):
    return M["build_tfidf"](_corpus, cache_key=df_hash)


@st.cache_data(show_spinner=False, ttl=3600)
def get_lda(df_hash: str, _corpus: list, k: int):
    return M["run_lda"](_corpus, n_topics=k, cache_key=df_hash)


@st.cache_data(show_spinner=False, ttl=3600)
def get_nmf(df_hash: str, _corpus: list, k: int):
    return M["run_nmf"](_corpus, n_topics=k, cache_key=df_hash)


@st.cache_data(show_spinner=False, ttl=3600)
def get_top_terms(df_hash: str, _df: pd.DataFrame):
    return M["extract_top_terms"](_df)


@st.cache_data(show_spinner=False, ttl=3600)
def get_network(df_hash: str, _df: pd.DataFrame):
    return M["build_concept_network"](_df, cache_key=df_hash)


@st.cache_data(show_spinner=False, ttl=3600)
def get_model_comparison(df_hash: str, _corpus: list):
    k_range = [4, 5, 6, 7, 8, 9, 10, 12]
    lda_scores, nmf_scores = {}, {}
    for k in k_range:
        lda_r = M["run_lda"](_corpus, n_topics=k, cache_key=df_hash)
        nmf_r = M["run_nmf"](_corpus, n_topics=k, cache_key=df_hash)
        lda_scores[k] = lda_r["perplexity"]
        nmf_scores[k] = nmf_r["recon_error"]
    return lda_scores, nmf_scores


# ─── Load & spin ──────────────────────────────────────────────────────────────

with st.spinner("🔬 Loading and processing literature corpus …"):
    try:
        df = load_data(n_papers, force_reload)
        is_demo = df["id"].astype(str).str.startswith("demo").any()
    except Exception as exc:
        st.error(f"❌ Failed to load dataset: {exc}")
        st.code(traceback.format_exc())
        st.stop()

# Demo warning banner
if is_demo:
    st.markdown("""
    <div class="demo-banner">
        ⚠️ <b>Demo Mode Active</b> — Kaggle credentials not found or download failed.
        The charts use <b>synthetically generated</b> demo data (clearly labelled [DEMO]).
        To use real arXiv data, set up Kaggle credentials and restart.
        See the README for instructions.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="success-banner">
        ✅ <b>Real arXiv Data Loaded</b> — Analysis reflects actual research literature.
    </div>""", unsafe_allow_html=True)

# ─── Preprocessing pipeline ───────────────────────────────────────────────────
df_hash = M["make_cache_key"](df)

with st.spinner("🧹 Preprocessing text corpus …"):
    corpus = get_preprocessed(df_hash, df)

with st.spinner("⚙️ Running topic model …"):
    lda_result = get_lda(df_hash, corpus, n_topics)
    nmf_result = get_nmf(df_hash, corpus, n_topics)

# Choose active model
use_lda = model_choice.startswith("LDA")
active_result = lda_result if use_lda else nmf_result

# Build topic labels
topic_labels = []
for t_idx in range(n_topics):
    words = [w for w, _ in active_result["topic_words"][t_idx][:8]]
    topic_labels.append(M["label_topic"](words))


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER – section header
# ═══════════════════════════════════════════════════════════════════════════════

def section_header(title: str, badge: str = "", icon: str = ""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <h2>{icon} {title}</h2>
        {badge_html}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW & DATASET
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview & Dataset":
    section_header("Overview & Dataset", "arXiv Corpus", "📊")

    # KPI row
    year_range = f"{int(df['year'].min())} – {int(df['year'].max())}" if df["year"].notna().any() else "N/A"
    n_cats     = df["primary_cat"].nunique()
    avg_abs_len = int(df["abstract_clean"].str.len().mean())

    cols = st.columns(4)
    kpis = [
        ("Papers Analysed",   f"{len(df):,}",       f"Subset of arXiv corpus"),
        ("Research Domains",  str(n_cats),           f"{n_topics} topics modelled"),
        ("Year Range",        year_range,            "arXiv preprints"),
        ("Avg Abstract Len",  f"{avg_abs_len} chars","After cleaning"),
    ]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#44BBA4"]
    for col, (lbl, val, delta), color in zip(cols, kpis, colors):
        col.markdown(M["kpi_card_html"](lbl, val, delta, color), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(M["papers_per_year"](df), use_container_width=True)
    with c2:
        st.plotly_chart(M["category_distribution"](df), use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Dataset preview
    section_header("Sample Records", "Preview", "📋")
    st.markdown("""
    <div style="font-size:0.83rem;color:#8b949e;margin-bottom:.5rem">
    Showing 10 random papers from the loaded corpus. 
    Use the <b>Paper Search</b> tab to explore specific topics.
    </div>""", unsafe_allow_html=True)

    preview = df.sample(min(10, len(df)), random_state=42)[
        ["title", "year", "cat_label", "abstract_clean"]
    ].rename(columns={
        "title": "Title", "year": "Year",
        "cat_label": "Category", "abstract_clean": "Abstract (first 200 chars)"
    })
    preview["Abstract (first 200 chars)"] = preview["Abstract (first 200 chars)"].str[:200] + "…"
    st.dataframe(preview, use_container_width=True, height=350)

    section_header("Methodology", "How it works", "⚙️")
    with st.expander("Data Collection & Preprocessing Pipeline", expanded=False):
        st.markdown("""
        **Step 1 – Data Retrieval**
        - Dataset downloaded via `kagglehub` from the arXiv dataset (Cornell University / Kaggle, DSV 6015950).
        - Streaming JSON-lines parsing to avoid loading 1.7M records at once.
        - Filtered to Computer Science, ML, and interdisciplinary categories, 2010–present.

        **Step 2 – Text Cleaning**
        - Unicode normalisation, URL removal, punctuation stripping.
        - Stop-word removal (English + domain-specific), lemmatisation (WordNet).
        - Min/max document frequency filtering to remove very rare and ubiquitous terms.

        **Step 3 – Representation**
        - TF-IDF (1- and 2-grams, max 5,000 features) for term importance.
        - Count matrix for LDA; TF-IDF matrix for NMF.

        **Step 4 – Topic Modelling**
        - LDA (online variational Bayes, 30 iterations, batch size 256).
        - NMF (NNDSVDA initialisation, 300 iterations).
        - Topic labels assigned via lexical matching heuristics.

        **Step 5 – Gap Analysis**
        - Each topic scored on: size (under-representation), recency, trend slope, inter-topic entropy.
        - Only gaps backed by ≥3 papers are reported.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – TOPIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Topic Analysis":
    section_header("Topic Analysis", f"{n_topics} Topics · {model_choice[:3]}", "🧠")

    st.markdown(f"""
    <div style="font-size:0.85rem;color:#8b949e;margin-bottom:1rem">
    Topics discovered by <b>{model_choice}</b> from {len(df):,} papers.
    Each topic is a cluster of semantically related research themes.
    Click a topic card to explore its top terms.
    </div>""", unsafe_allow_html=True)

    # Topic heatmap
    st.plotly_chart(
        M["topic_year_heatmap"](df, active_result["doc_topic_matrix"], topic_labels),
        use_container_width=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    section_header("Topic Cards", "Top terms per topic", "🃏")

    # Selectable topic grid
    selected_topic = st.selectbox(
        "Select topic to inspect",
        options=list(range(n_topics)),
        format_func=lambda i: f"Topic {i+1}: {topic_labels[i]}",
    )

    c1, c2 = st.columns([2, 3])
    with c1:
        top_words = [w for w, _ in active_result["topic_words"][selected_topic][:15]]
        # Topic card
        doc_count = int((active_result["doc_topic_matrix"].argmax(axis=1) == selected_topic).sum())
        st.markdown(f"""
        <div class="topic-card">
            <h4>Topic {selected_topic+1}: {topic_labels[selected_topic]}</h4>
            <p><b>{doc_count}</b> dominant papers ({doc_count/len(df)*100:.1f}% of corpus)</p>
            <p style="margin-top:8px;color:#E6EDF3"><b>Key Terms:</b></p>
            <p>{"  ".join(f'<span class="evidence-chip">{w}</span>' for w in top_words[:10])}</p>
        </div>""", unsafe_allow_html=True)

        # Word cloud
        wc_text = " ".join(
            f"{w} " * max(1, int(s * 1000))
            for w, s in active_result["topic_words"][selected_topic][:30]
        )
        wc_fig = M["generate_wordcloud_fig"](wc_text, f"Word Cloud: {topic_labels[selected_topic]}")
        if wc_fig:
            st.plotly_chart(wc_fig, use_container_width=True)

    with c2:
        st.plotly_chart(
            M["topic_word_bars"](active_result["topic_words"], selected_topic, topic_labels[selected_topic]),
            use_container_width=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    section_header("All Topics Summary", "Overview", "📋")

    # Summary table
    rows = []
    for i in range(n_topics):
        doc_count = int((active_result["doc_topic_matrix"].argmax(axis=1) == i).sum())
        top3 = ", ".join(w for w, _ in active_result["topic_words"][i][:3])
        rows.append({
            "Topic": f"{i+1}",
            "Label": topic_labels[i],
            "Dominant Papers": doc_count,
            "% of Corpus": f"{doc_count/len(df)*100:.1f}%",
            "Top 3 Terms": top3,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – TREND EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Trend Explorer":
    section_header("Emerging Trend Analysis", "Temporal", "📈")

    st.markdown("""
    <div style="font-size:0.85rem;color:#8b949e;margin-bottom:1rem">
    Tracks how research attention shifts across topics over time.
    Topic share is computed as the mean topic probability across all papers per year.
    </div>""", unsafe_allow_html=True)

    # Controls
    top_n_trends = st.slider("Topics to display", 3, min(n_topics, 8), min(6, n_topics))

    st.plotly_chart(
        M["topic_trend_lines"](df, active_result["doc_topic_matrix"], topic_labels, top_n_trends),
        use_container_width=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    section_header("Year-by-Year Breakdown", "Detailed", "📅")

    # Year filter
    year_options = sorted(df["year"].dropna().unique().astype(int).tolist())
    if len(year_options) >= 2:
        yr_range = st.select_slider(
            "Filter year range",
            options=year_options,
            value=(year_options[0], year_options[-1]),
        )
        df_filtered = df[(df["year"] >= yr_range[0]) & (df["year"] <= yr_range[1])]
    else:
        df_filtered = df

    # Recompute dominant topic for filtered rows
    orig_idx = df_filtered.index.tolist()
    doc_topic_filtered = active_result["doc_topic_matrix"][orig_idx]
    df_trend = df_filtered.copy()
    df_trend["dominant_topic"] = doc_topic_filtered.argmax(axis=1)
    df_trend["topic_label"] = df_trend["dominant_topic"].map(
        lambda i: topic_labels[i] if i < len(topic_labels) else f"Topic {i}"
    )

    pivot = df_trend.groupby(["year", "topic_label"]).size().unstack(fill_value=0)
    pivot.index = pivot.index.astype(int)
    pivot = pivot.sort_index()

    st.markdown(f"**Papers per topic per year** ({yr_range[0] if len(year_options) >= 2 else 'all'} – {yr_range[1] if len(year_options) >= 2 else 'years'})")
    st.dataframe(pivot.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    # Fastest growing topics
    if len(pivot) >= 3:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        section_header("Fastest Growing Topics", "Trend velocity", "🚀")

        slopes = {}
        for col in pivot.columns:
            vals = pivot[col].values.astype(float)
            if len(vals) >= 2:
                x = np.arange(len(vals))
                c = np.polyfit(x, vals, 1)
                slopes[col] = c[0]

        slope_df = pd.DataFrame(
            [{"Topic": k, "Growth Slope": round(v, 2), "Direction": "↑ Growing" if v > 0 else "↓ Declining"}
             for k, v in sorted(slopes.items(), key=lambda x: -x[1])]
        )
        st.dataframe(slope_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – CONCEPT NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🕸️ Concept Network":
    section_header("Concept Co-occurrence Network", "Term Relationships", "🕸️")

    st.markdown("""
    <div style="font-size:0.85rem;color:#8b949e;margin-bottom:1rem">
    Nodes are important technical terms (by TF-IDF score).
    Edges connect terms that appear together frequently.
    Node size reflects connection degree; colour represents community cluster.
    </div>""", unsafe_allow_html=True)

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        max_nodes = st.slider("Max nodes", 20, 80, 50, 5)
    with c2:
        min_cooc  = st.slider("Min co-occurrences", 2, 20, 5, 1)

    with st.spinner("Building concept network …"):
        nodes_df, edges_df = get_network(df_hash, df)

    st.plotly_chart(
        M["concept_network_plotly"](nodes_df, edges_df, max_nodes=max_nodes),
        use_container_width=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section_header("Top Terms", "by TF-IDF score", "📊")
        with st.spinner("Computing top terms …"):
            top_terms_df = get_top_terms(df_hash, df)
        st.plotly_chart(M["top_terms_bar"](top_terms_df, top_n=25), use_container_width=True)

    with c2:
        section_header("Most Connected Terms", "by network degree", "🔗")
        if not edges_df.empty:
            degree_map = {}
            for _, row in edges_df.iterrows():
                degree_map[row.source] = degree_map.get(row.source, 0) + row.weight
                degree_map[row.target] = degree_map.get(row.target, 0) + row.weight
            deg_df = pd.DataFrame(
                [{"Term": k, "Connection Strength": int(v)} for k, v in degree_map.items()]
            ).sort_values("Connection Strength", ascending=False).head(20)
            st.dataframe(deg_df, use_container_width=True, hide_index=True)
        else:
            st.info("No network data available yet.")

    # Word cloud – global
    section_header("Global Term Cloud", "All topics", "☁️")
    all_text = " ".join(corpus)
    wc_fig = M["generate_wordcloud_fig"](all_text[:50000], "Research Corpus – Key Terms")
    if wc_fig:
        st.plotly_chart(wc_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚖️ Model Comparison":
    section_header("Topic Model Quality Comparison", "LDA vs NMF", "⚖️")

    st.markdown("""
    <div style="font-size:0.85rem;color:#8b949e;margin-bottom:1rem">
    Runs both LDA and NMF for k = 4 to 12 topics.
    Lower perplexity (LDA) and lower reconstruction error (NMF) indicate better fit.
    The "elbow" in each curve suggests the optimal number of topics.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running model comparison (this may take 1–2 minutes) …"):
        lda_scores, nmf_scores = get_model_comparison(df_hash, corpus)

    st.plotly_chart(M["model_comparison_chart"](lda_scores, nmf_scores), use_container_width=True)

    # Current model metrics
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    section_header("Current Model Metrics", f"k={n_topics}", "📏")

    c1, c2, c3, c4 = st.columns(4)
    lda_perp  = lda_result.get("perplexity", 0)
    nmf_recon = nmf_result.get("recon_error", 0)

    with c1:
        st.markdown(M["kpi_card_html"]("LDA Perplexity", f"{lda_perp:,.0f}", f"k={n_topics}", "#2E86AB"), unsafe_allow_html=True)
    with c2:
        st.markdown(M["kpi_card_html"]("NMF Recon Error", f"{nmf_recon:.2f}", f"k={n_topics}", "#A23B72"), unsafe_allow_html=True)
    with c3:
        st.markdown(M["kpi_card_html"]("Corpus Size", f"{len(corpus):,}", "documents", "#F18F01"), unsafe_allow_html=True)
    with c4:
        st.markdown(M["kpi_card_html"]("Vocabulary Size", f"{len(lda_result['feature_names']):,}", "LDA tokens", "#44BBA4"), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    section_header("LDA vs NMF: Topic-by-topic", "Side-by-side", "🔀")

    tab_lda, tab_nmf = st.tabs(["LDA Topics", "NMF Topics"])
    with tab_lda:
        lda_labels = [M["label_topic"]([w for w, _ in lda_result["topic_words"][i][:8]])
                      for i in range(n_topics)]
        for i in range(n_topics):
            top5 = ", ".join(w for w, _ in lda_result["topic_words"][i][:5])
            st.markdown(f"""
            <div class="topic-card">
                <h4>LDA Topic {i+1}: {lda_labels[i]}</h4>
                <p>{top5}</p>
            </div>""", unsafe_allow_html=True)

    with tab_nmf:
        nmf_labels = [M["label_topic"]([w for w, _ in nmf_result["topic_words"][i][:8]])
                      for i in range(n_topics)]
        for i in range(n_topics):
            top5 = ", ".join(w for w, _ in nmf_result["topic_words"][i][:5])
            st.markdown(f"""
            <div class="topic-card" style="border-left-color:#A23B72">
                <h4>NMF Topic {i+1}: {nmf_labels[i]}</h4>
                <p>{top5}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr class="divider">
    <div style="font-size:0.82rem;color:#8b949e">
    <b>Interpreting the results:</b><br>
    • <b>LDA</b> assigns probabilistic topic mixtures per document — good for exploring thematic overlap.<br>
    • <b>NMF</b> produces sparse, parts-based representations — often yields more interpretable topics.<br>
    • Neither method "proves" topics exist; they are statistical summaries of word co-occurrence patterns.<br>
    • Use multiple k values and both metrics together to choose the best configuration.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – RESEARCH GAPS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Research Gaps":
    section_header("Research Gap Discovery", "Evidence-Based", "🔍")

    st.markdown("""
    <div style="font-size:0.85rem;color:#8b949e;margin-bottom:1rem">
    Research gaps are identified by analysing topic size, recency, trend trajectory, and
    interdisciplinary spread. <b>Every gap is backed by real papers</b> — click to see the evidence.
    Gaps are not fabricated; they are patterns observed in the data.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Identifying research gaps …"):
        top_terms_df_gap = get_top_terms(df_hash, df)
        gap_df, gap_docs = M["find_research_gaps"](
            df, lda_result, topic_labels, top_terms_df_gap
        )

    if gap_df.empty:
        st.warning("Not enough data to identify meaningful research gaps. Try increasing the paper count.")
        st.stop()

    # Summary KPIs
    n_gaps  = len(gap_df)
    top_gap = gap_df.iloc[0]
    avg_score = gap_df["gap_score"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(M["kpi_card_html"]("Gaps Identified", str(n_gaps), "from topic analysis", "#C73E1D"), unsafe_allow_html=True)
    with c2:
        st.markdown(M["kpi_card_html"]("Highest Gap Score", f"{top_gap['gap_score']:.2f}", top_gap["gap_label"][:25], "#F18F01"), unsafe_allow_html=True)
    with c3:
        st.markdown(M["kpi_card_html"]("Avg Gap Score", f"{avg_score:.2f}", "0=no gap, 1=major gap", "#2E86AB"), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Bubble chart
    st.plotly_chart(M["gap_bubble_chart"](gap_df), use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    section_header("Gap Details & Evidence", "Click to expand", "📑")

    # Sort filter
    sort_by = st.selectbox("Sort gaps by", ["Gap Score (high→low)", "Paper Count (low→high)", "Average Year (oldest first)"])
    if sort_by == "Gap Score (high→low)":
        sorted_gaps = sorted(gap_docs, key=lambda x: -x["gap_score"])
    elif sort_by == "Paper Count (low→high)":
        sorted_gaps = sorted(gap_docs, key=lambda x: x["paper_count"])
    else:
        sorted_gaps = sorted(gap_docs, key=lambda x: x["avg_year"])

    for gap in sorted_gaps:
        score = gap["gap_score"]
        severity = "high" if score > 0.6 else ("medium" if score > 0.35 else "low")
        color    = {"high": "#C73E1D", "medium": "#F18F01", "low": "#44BBA4"}[severity]

        with st.expander(
            f"{'🔴' if severity=='high' else '🟡' if severity=='medium' else '🟢'} "
            f"{gap['gap_label']}  ·  Score: {score:.2f}  ·  {gap['gap_type']}",
            expanded=False,
        ):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"""
                <div class="gap-card {severity}">
                    <div style="font-size:0.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.08em">{gap['gap_type']}</div>
                    <div class="gap-score" style="color:{color}">{score:.2f}</div>
                    <p style="color:#E6EDF3;margin:8px 0">{gap['gap_description']}</p>
                    <div style="margin-top:8px">
                        <b style="color:#8b949e;font-size:0.8rem">KEY TERMS:</b><br>
                        {"".join(f'<span class="evidence-chip">{w}</span>' for w in gap['top_words'][:8])}
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:12px">
                    <b style="color:#E6EDF3">📍 Suggested Research Directions</b>
                </div>""", unsafe_allow_html=True)
                for d in gap["directions"]:
                    st.markdown(f"- {d}")

            with c2:
                st.metric("Papers", gap["paper_count"])
                st.metric("Avg Year", f"{gap['avg_year']:.0f}")
                st.metric("Evidence Papers", gap["evidence_count"])
                trend = "↓ Declining" if gap["trend_slope"] < -0.001 else "↑ Growing" if gap["trend_slope"] > 0.001 else "→ Stable"
                st.metric("Trend", trend)

            st.markdown("**📄 Supporting Evidence Papers:**")
            for ep in gap["evidence_papers"][:5]:
                title = ep.get("title", "Unknown")
                year  = ep.get("year", "N/A")
                cats  = ep.get("categories", "")
                arxiv_id = ep.get("id", "")
                link = f"https://arxiv.org/abs/{arxiv_id}" if not arxiv_id.startswith("demo") else "#"
                st.markdown(f"""
                <div class="paper-row">
                    <div class="title">{title}</div>
                    <div class="meta">📅 {year} &nbsp;·&nbsp; 🏷️ {cats}
                    {"&nbsp;·&nbsp; <a href='" + link + "' target='_blank' style='color:#2E86AB'>View on arXiv ↗</a>" if not arxiv_id.startswith("demo") else ""}
                    </div>
                </div>""", unsafe_allow_html=True)

    # Exportable summary
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    section_header("Export Gap Report", "Download", "⬇️")
    gap_export = gap_df.copy()
    csv = gap_export.to_csv(index=False)
    st.download_button(
        "⬇️ Download Gap Summary CSV",
        data=csv,
        file_name="research_gaps.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown("""
    <div style="font-size:0.78rem;color:#484F58;margin-top:1rem">
    ⚠️ <b>Caveat:</b> Research gaps identified here are statistical patterns in the sampled literature.
    They should be validated through domain expert review and a thorough manual literature search.
    The analysis does not account for unpublished work, patents, or papers outside the arXiv repository.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 – PAPER SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📄 Paper Search":
    section_header("Paper Search", "Explore the corpus", "📄")

    # Filters
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        query = st.text_input("🔍 Search by title or abstract keyword", placeholder="e.g. federated learning privacy")
    with c2:
        year_from = st.number_input("Year from", value=int(df["year"].min()) if df["year"].notna().any() else 2010, step=1)
    with c3:
        year_to = st.number_input("Year to",   value=int(df["year"].max()) if df["year"].notna().any() else 2024, step=1)

    cats = ["All"] + sorted(df["cat_label"].unique().tolist())
    cat_filter = st.selectbox("Filter by research domain", cats)

    # Apply topic filter
    topic_filter = st.multiselect(
        "Filter by modelled topic",
        options=list(range(n_topics)),
        format_func=lambda i: f"Topic {i+1}: {topic_labels[i]}",
        default=[],
    )

    # Build search mask
    mask = (df["year"].fillna(0) >= year_from) & (df["year"].fillna(9999) <= year_to)
    if cat_filter != "All":
        mask &= df["cat_label"] == cat_filter
    if query.strip():
        q = query.lower()
        mask &= (
            df["title_clean"].str.lower().str.contains(q, na=False) |
            df["abstract_clean"].str.lower().str.contains(q, na=False)
        )
    if topic_filter:
        dominant = active_result["doc_topic_matrix"].argmax(axis=1)
        mask &= pd.Series(dominant, index=df.index).isin(topic_filter)

    results = df[mask].copy()
    results["dominant_topic"] = active_result["doc_topic_matrix"][results.index].argmax(axis=1)
    results["modelled_topic"] = results["dominant_topic"].map(
        lambda i: topic_labels[i] if i < len(topic_labels) else f"Topic {i}"
    )

    st.markdown(f"**{len(results):,} papers found**")

    if len(results) == 0:
        st.info("No papers match your search criteria. Try broadening the filters.")
    else:
        show_n = st.slider("Results to show", 5, min(50, len(results)), min(15, len(results)))
        for _, row in results.head(show_n).iterrows():
            arxiv_id = str(row.get("id", ""))
            link = f"https://arxiv.org/abs/{arxiv_id}" if not arxiv_id.startswith("demo") else None
            link_html = f'<a href="{link}" target="_blank" style="color:#2E86AB;font-size:0.78rem">View on arXiv ↗</a>' if link else ""
            st.markdown(f"""
            <div class="paper-row">
                <div class="title">{row['title']}</div>
                <div class="meta">
                    📅 {row['year']} &nbsp;·&nbsp;
                    🏷️ {row['cat_label']} &nbsp;·&nbsp;
                    🧠 {row['modelled_topic']} &nbsp;&nbsp;
                    {link_html}
                </div>
                <div class="abstract">{str(row['abstract_clean'])[:350]}…</div>
            </div>""", unsafe_allow_html=True)

        # Export
        export_cols = ["id", "title", "year", "cat_label", "modelled_topic", "abstract_clean"]
        export_df   = results[export_cols].head(show_n)
        st.download_button(
            "⬇️ Export results to CSV",
            data=export_df.to_csv(index=False),
            file_name="paper_search_results.csv",
            mime="text/csv",
        )
