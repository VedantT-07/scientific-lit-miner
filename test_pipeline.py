"""
test_pipeline.py
Quick integration test — run with: python test_pipeline.py
"""
import sys
print("Testing data_loader...")
from data_loader import _generate_demo_data, _preprocess, CATEGORY_LABELS
df = _generate_demo_data(300)
df = _preprocess(df)
print(f"  OK: DataFrame shape {df.shape}, columns: {list(df.columns)}")

print("Testing nlp_engine...")
from nlp_engine import (
    preprocess_corpus, build_tfidf, run_lda, run_nmf,
    label_topic, build_concept_network, extract_top_terms
)

corpus = preprocess_corpus(df["combined_text"].tolist())
print(f"  OK: Corpus length {len(corpus)}, sample: {corpus[0][:60]}...")

mat, vec, fn = build_tfidf(corpus)
print(f"  OK: TF-IDF shape {mat.shape}, vocab size {len(fn)}")

lda = run_lda(corpus, n_topics=5)
print(f"  OK: LDA perplexity {lda['perplexity']:.1f}")
for i in range(5):
    words = [w for w, _ in lda["topic_words"][i][:5]]
    label = label_topic(words)
    print(f"       Topic {i+1}: {label} | {words}")

nmf = run_nmf(corpus, n_topics=5)
print(f"  OK: NMF recon error {nmf['recon_error']:.2f}")

print("Testing concept network...")
nodes_df, edges_df = build_concept_network(df, top_terms=30, min_cooccurrence=2)
print(f"  OK: Nodes {len(nodes_df)}, Edges {len(edges_df)}")

print("Testing top terms extraction...")
terms_df = extract_top_terms(df)
print(f"  OK: Top terms df shape {terms_df.shape}")
print(f"       Top 5 terms: {terms_df['term'].tolist()[:5]}")

print("Testing gap_analysis...")
from gap_analysis import find_research_gaps
labels = [label_topic([w for w, _ in lda["topic_words"][i][:8]]) for i in range(5)]
gap_df, gap_docs = find_research_gaps(df, lda, labels, terms_df)
print(f"  OK: Found {len(gap_docs)} potential gaps")
if gap_docs:
    top_gap = gap_docs[0]
    print(f"       Top gap: {top_gap['gap_label']} (score {top_gap['gap_score']:.2f})")
    print(f"       Evidence papers: {top_gap['evidence_count']}")

print("Testing visualizations...")
from visualizations import (
    papers_per_year, category_distribution,
    topic_year_heatmap, topic_word_bars, topic_trend_lines,
    top_terms_bar, concept_network_plotly, gap_bubble_chart
)

fig1 = papers_per_year(df)
print(f"  OK: papers_per_year figure type: {type(fig1).__name__}")

fig2 = category_distribution(df)
print(f"  OK: category_distribution figure type: {type(fig2).__name__}")

fig3 = topic_year_heatmap(df, lda["doc_topic_matrix"], labels)
print(f"  OK: topic_year_heatmap figure type: {type(fig3).__name__}")

fig4 = topic_word_bars(lda["topic_words"], 0, labels[0])
print(f"  OK: topic_word_bars figure type: {type(fig4).__name__}")

fig5 = top_terms_bar(terms_df)
print(f"  OK: top_terms_bar figure type: {type(fig5).__name__}")

fig6 = concept_network_plotly(nodes_df, edges_df)
print(f"  OK: concept_network_plotly figure type: {type(fig6).__name__}")

if not gap_df.empty:
    fig7 = gap_bubble_chart(gap_df)
    print(f"  OK: gap_bubble_chart figure type: {type(fig7).__name__}")

print()
print("=" * 50)
print("ALL PIPELINE TESTS PASSED [OK]")
print("=" * 50)
print()
print("Run the app with:")
print("  streamlit run app.py")
