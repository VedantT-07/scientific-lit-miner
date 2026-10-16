"""
visualizations.py
─────────────────
All Plotly-based chart builders. Each function accepts processed data
and returns a go.Figure or similar object ready for st.plotly_chart().
"""

import re
import textwrap
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Colour palette (academic, professional) ──────────────────────────────────
PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B2D8B",
    "#00A896", "#F15946", "#028090", "#E4572E", "#76B041",
    "#D62839", "#171738", "#70B77E", "#9B2335", "#6C4675",
]

BG_COLOR   = "#0E1117"
PAPER_BG   = "#161B22"
GRID_COLOR = "#30363D"
FONT_COLOR = "#E6EDF3"
ACCENT     = "#2E86AB"

_LAYOUT_BASE = dict(
    plot_bgcolor  = BG_COLOR,
    paper_bgcolor = PAPER_BG,
    font          = dict(color=FONT_COLOR, family="Inter, Arial, sans-serif", size=12),
    margin        = dict(l=50, r=20, t=50, b=50),
    xaxis         = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis         = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR, borderwidth=1),
)


def _apply_layout(fig: go.Figure, title: str = "", **extra) -> go.Figure:
    kw = dict(_LAYOUT_BASE)
    kw.update(extra)
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=FONT_COLOR)), **kw)
    return fig


# ─── KPI Cards (HTML) ────────────────────────────────────────────────────────

def kpi_card_html(label: str, value: str, delta: str = "", color: str = ACCENT) -> str:
    delta_html = f'<div style="font-size:0.8rem;color:#6e7681;margin-top:4px">{delta}</div>' if delta else ""
    return f"""
    <div style="background:{PAPER_BG};border:1px solid {GRID_COLOR};border-radius:8px;
                padding:16px 20px;text-align:center;border-top:3px solid {color}">
        <div style="font-size:0.75rem;color:#8b949e;letter-spacing:.08em;text-transform:uppercase">{label}</div>
        <div style="font-size:2rem;font-weight:700;color:{FONT_COLOR};margin-top:4px">{value}</div>
        {delta_html}
    </div>"""


# ─── 1. Papers Per Year ───────────────────────────────────────────────────────

def papers_per_year(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("year").size().reset_index(name="count")
    counts = counts.sort_values("year")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["year"], y=counts["count"],
        marker_color=ACCENT, name="Papers",
        hovertemplate="<b>%{x}</b><br>%{y} papers<extra></extra>",
    ))
    # Rolling average
    if len(counts) >= 3:
        counts["ma"] = counts["count"].rolling(3, center=True, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=counts["year"], y=counts["ma"],
            mode="lines", line=dict(color="#F18F01", width=2, dash="dash"),
            name="3-yr avg", hovertemplate="%{y:.0f}<extra></extra>",
        ))
    return _apply_layout(fig, "Publication Volume by Year",
                         xaxis_title="Year", yaxis_title="Papers")


# ─── 2. Category Distribution ────────────────────────────────────────────────

def category_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["cat_label"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    counts = counts.head(15)

    fig = go.Figure(go.Bar(
        x=counts["count"],
        y=counts["category"],
        orientation="h",
        marker=dict(
            color=counts["count"],
            colorscale=[[0, "#1a3a4a"], [1, ACCENT]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>%{x} papers<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_layout(fig, "Papers by Research Domain",
                         xaxis_title="Count", yaxis_title="",
                         height=max(350, len(counts) * 30))


# ─── 3. Topic Heatmap (topic × year) ────────────────────────────────────────

def topic_year_heatmap(
    df: pd.DataFrame,
    doc_topic_matrix: np.ndarray,
    topic_labels: List[str],
) -> go.Figure:
    """Show how each topic's share evolves across years."""
    df2 = df.copy()
    df2["dominant_topic"] = doc_topic_matrix.argmax(axis=1)

    pivot = (
        df2.groupby(["year", "dominant_topic"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalise to row percentages
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot.columns = [topic_labels[c] if c < len(topic_labels) else f"Topic {c}" for c in pivot.columns]
    pivot = pivot.sort_index()

    fig = go.Figure(go.Heatmap(
        z=pivot.values.T,
        x=pivot.index.astype(str),
        y=pivot.columns.tolist(),
        colorscale="Blues",
        hovertemplate="Year: %{x}<br>Topic: %{y}<br>Share: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Share %", tickfont=dict(color=FONT_COLOR)),
    ))
    return _apply_layout(fig, "Topic Prevalence Over Time (%)",
                         xaxis_title="Year", yaxis_title="Topic",
                         height=max(400, len(pivot.columns) * 35))


# ─── 4. Topic Word Bar Chart ─────────────────────────────────────────────────

def topic_word_bars(topic_words: Dict, topic_idx: int, label: str) -> go.Figure:
    words_scores = topic_words[topic_idx][:15]
    words  = [w for w, _ in words_scores][::-1]
    scores = [s for _, s in words_scores][::-1]

    fig = go.Figure(go.Bar(
        x=scores, y=words,
        orientation="h",
        marker=dict(color=PALETTE[topic_idx % len(PALETTE)]),
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>",
    ))
    return _apply_layout(fig, f"Topic: {label}",
                         xaxis_title="Weight", yaxis_title="",
                         height=380)


# ─── 5. Trend Lines (topic share over years) ─────────────────────────────────

def topic_trend_lines(
    df: pd.DataFrame,
    doc_topic_matrix: np.ndarray,
    topic_labels: List[str],
    top_n: int = 6,
) -> go.Figure:
    df2 = df.copy()
    years = sorted(df2["year"].dropna().unique())

    # Topic share per year (average probability)
    records = []
    for y in years:
        mask = df2["year"] == y
        if mask.sum() < 3:
            continue
        avg = doc_topic_matrix[mask].mean(axis=0)
        for t_idx, val in enumerate(avg):
            records.append({"year": y, "topic": topic_labels[t_idx] if t_idx < len(topic_labels) else f"Topic {t_idx}", "share": val})

    trend_df = pd.DataFrame(records)

    # Pick top_n topics by mean share
    top_topics = (
        trend_df.groupby("topic")["share"].mean()
        .nlargest(top_n).index.tolist()
    )
    trend_df = trend_df[trend_df["topic"].isin(top_topics)]

    fig = go.Figure()
    for i, topic in enumerate(top_topics):
        sub = trend_df[trend_df["topic"] == topic].sort_values("year")
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["share"],
            mode="lines+markers",
            name=textwrap.shorten(topic, 30),
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{topic}</b><br>Year: %{{x}}<br>Share: %{{y:.3f}}<extra></extra>",
        ))
    return _apply_layout(fig, "Emerging Topic Trends",
                         xaxis_title="Year", yaxis_title="Average Topic Probability",
                         hovermode="x unified")


# ─── 6. TF-IDF Top Terms Bar ─────────────────────────────────────────────────

def top_terms_bar(terms_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    sub = terms_df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=sub["score"],
        y=sub["term"],
        orientation="h",
        marker=dict(
            color=sub["score"],
            colorscale=[[0, "#1a3a4a"], [1, "#2E86AB"]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>TF-IDF: %{x:.4f}<extra></extra>",
    ))
    return _apply_layout(fig, f"Top {top_n} Technical Terms (TF-IDF)",
                         xaxis_title="Mean TF-IDF Score", yaxis_title="",
                         height=max(400, top_n * 22))


# ─── 7. Concept Network (Plotly) ──────────────────────────────────────────────

def concept_network_plotly(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_nodes: int = 40,
    max_edges: int = 120,
) -> go.Figure:
    """Render concept co-occurrence network using Plotly."""
    import networkx as nx

    if edges_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough co-occurrence data to build a network.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=FONT_COLOR))
        return _apply_layout(fig, "Concept Network")

    # Build graph
    G = nx.Graph()
    # Add top nodes by degree
    top_nodes = nodes_df.sort_values("degree", ascending=False).head(max_nodes)["term"].tolist()
    node_set = set(top_nodes)

    top_edges = edges_df[
        edges_df.source.isin(node_set) & edges_df.target.isin(node_set)
    ].nlargest(max_edges, "weight")

    for _, row in top_edges.iterrows():
        G.add_edge(row.source, row.target, weight=row.weight)

    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No network data available.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=FONT_COLOR))
        return _apply_layout(fig, "Concept Network")

    # Layout
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G.nodes)))

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#30363D"),
        hoverinfo="none",
        showlegend=False,
    )

    # Nodes
    nx_degrees = dict(G.degree())
    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    node_sizes = [6 + nx_degrees.get(n, 1) * 2 for n in G.nodes]
    node_text = list(G.nodes)

    # Cluster by community
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        node_colors = []
        for n in G.nodes:
            for i, comm in enumerate(communities):
                if n in comm:
                    node_colors.append(PALETTE[i % len(PALETTE)])
                    break
            else:
                node_colors.append(ACCENT)
    except Exception:
        node_colors = ACCENT

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color=FONT_COLOR),
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=0.5, color="#161B22")),
        hovertemplate="<b>%{text}</b><br>Connections: %{marker.size}<extra></extra>",
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=BG_COLOR, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="Concept Co-occurrence Network", font=dict(size=16, color=FONT_COLOR)),
    )
    return fig


# ─── 8. Model Comparison Bar ──────────────────────────────────────────────────

def model_comparison_chart(lda_results: Dict, nmf_results: Dict) -> go.Figure:
    """Side-by-side comparison of LDA vs NMF coherence proxy."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["LDA – Log Perplexity (lower = better)",
                        "NMF – Reconstruction Error (lower = better)"],
    )

    k_vals  = sorted(lda_results.keys())
    lda_vals = [lda_results[k] for k in k_vals]
    nmf_vals = [nmf_results[k] for k in k_vals]

    fig.add_trace(go.Scatter(
        x=k_vals, y=lda_vals,
        mode="lines+markers",
        line=dict(color=PALETTE[0], width=2),
        marker=dict(size=8),
        name="LDA Perplexity",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=k_vals, y=nmf_vals,
        mode="lines+markers",
        line=dict(color=PALETTE[1], width=2),
        marker=dict(size=8),
        name="NMF Reconstruction Error",
    ), row=1, col=2)

    fig.update_layout(
        plot_bgcolor=BG_COLOR, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family="Inter, Arial, sans-serif"),
        title=dict(text="Topic Model Quality Comparison", font=dict(size=16, color=FONT_COLOR)),
        showlegend=False,
        height=350,
        xaxis=dict(title="Number of Topics (k)", gridcolor=GRID_COLOR),
        xaxis2=dict(title="Number of Topics (k)", gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        yaxis2=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ─── 9. Research Gap Bubble Chart ────────────────────────────────────────────

def gap_bubble_chart(gap_df: pd.DataFrame) -> go.Figure:
    """
    Visualise research gaps as bubbles.
    gap_df columns: gap_label, paper_count, avg_year, gap_score, evidence_count
    """
    if gap_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No gap data available.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=FONT_COLOR))
        return _apply_layout(fig, "Research Gap Map")

    fig = go.Figure()
    for i, row in gap_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row.get("avg_year", 2020)],
            y=[row.get("gap_score", 0)],
            mode="markers+text",
            marker=dict(
                size=max(15, min(60, row.get("evidence_count", 5) * 3)),
                color=PALETTE[i % len(PALETTE)],
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            text=[textwrap.shorten(row["gap_label"], 20)],
            textposition="top center",
            textfont=dict(size=9, color=FONT_COLOR),
            name=row["gap_label"],
            hovertemplate=(
                f"<b>{row['gap_label']}</b><br>"
                f"Papers: {row.get('paper_count','?')}<br>"
                f"Gap Score: {row.get('gap_score',0):.2f}<br>"
                f"Evidence Count: {row.get('evidence_count','?')}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title="Average Publication Year",
        yaxis_title="Research Gap Score",
        showlegend=False,
        height=400,
    )
    return _apply_layout(fig, "Research Gap Map (bubble size = evidence breadth)")


# ─── 10. Wordcloud ────────────────────────────────────────────────────────────

def generate_wordcloud_fig(text: str, title: str = "Word Cloud") -> Optional[go.Figure]:
    """Generate a word cloud and return as Plotly figure."""
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io, base64
        from PIL import Image

        wc = WordCloud(
            width=800, height=400,
            background_color="#0E1117",
            colormap="Blues",
            max_words=100,
            collocations=False,
        ).generate(text)

        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()

        fig = go.Figure()
        fig.add_layout_image(dict(
            source=f"data:image/png;base64,{img_b64}",
            xref="paper", yref="paper",
            x=0, y=1, sizex=1, sizey=1,
            sizing="stretch", layer="below",
        ))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=BG_COLOR, paper_bgcolor=PAPER_BG,
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
            title=dict(text=title, font=dict(size=14, color=FONT_COLOR)),
        )
        return fig
    except Exception:
        return None
