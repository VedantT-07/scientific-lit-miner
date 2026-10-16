"""
nlp_engine.py
─────────────
Text preprocessing, TF-IDF vectorisation, topic modelling (LDA via gensim),
and all NLP utilities used across the app.
Aggressively cached to disk so Streamlit reruns don't recompute.
"""

import re
import pickle
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ─── NLTK resources (downloaded once) ─────────────────────────────────────────

def ensure_nltk():
    import nltk
    for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    ensure_nltk()
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    STOP_WORDS = set()
    LEMMATIZER = None

# Extra domain stopwords
EXTRA_STOPS = {
    "paper", "propose", "proposed", "method", "approach", "model", "result",
    "show", "also", "use", "used", "using", "based", "present", "work",
    "new", "different", "however", "thus", "furthermore", "moreover",
    "demonstrate", "existing", "consider", "provide", "compare", "improve",
    "performance", "experiment", "dataset", "achieve", "state", "art",
    "number", "two", "three", "first", "second", "large", "small", "high",
    "low", "one", "well", "make", "find", "set", "et", "al", "fig", "table",
    "section", "arxiv", "preprint", "abstract", "introduction", "conclusion",
    "demo", "data",   # filter out demo markers
}
STOP_WORDS |= EXTRA_STOPS


# ─── Text cleaning ─────────────────────────────────────────────────────────────

_RE_URL     = re.compile(r"https?://\S+")
_RE_EMAIL   = re.compile(r"\S+@\S+")
_RE_PUNCT   = re.compile(r"[^a-zA-Z\s]")
_RE_SPACE   = re.compile(r"\s+")
_RE_SHORT   = re.compile(r"\b\w{1,2}\b")

def clean_for_nlp(text: str) -> str:
    """Full NLP cleaning pipeline: lower, strip URLs, remove punct, lemmatise."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_SHORT.sub(" ", text)
    text = _RE_SPACE.sub(" ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    if LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in STOP_WORDS]  # second pass post-lemma
    return " ".join(tokens)


def preprocess_corpus(texts: List[str], cache_key: str = "") -> List[str]:
    """Apply clean_for_nlp to a list of texts with caching."""
    cache_file = CACHE_DIR / f"corpus_{cache_key}.pkl"
    if cache_key and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    cleaned = [clean_for_nlp(t) for t in texts]

    if cache_key:
        with open(cache_file, "wb") as f:
            pickle.dump(cleaned, f, protocol=4)
    return cleaned


# ─── TF-IDF Vectorisation ─────────────────────────────────────────────────────

def build_tfidf(
    corpus: List[str],
    max_features: int = 5000,
    ngram_range: Tuple = (1, 2),
    cache_key: str = "",
) -> Tuple:
    """
    Fit TF-IDF on corpus.
    Returns (matrix, vectorizer, feature_names).
    """
    cache_file = CACHE_DIR / f"tfidf_{cache_key}.pkl"
    if cache_key and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()

    result = (matrix, vec, feature_names)
    if cache_key:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f, protocol=4)
    return result


# ─── Topic Modelling ──────────────────────────────────────────────────────────

def run_lda(
    corpus: List[str],
    n_topics: int = 8,
    max_features: int = 3000,
    cache_key: str = "",
) -> Dict:
    """
    Run LDA topic modelling.
    Returns dict with: model, vectorizer, doc_topic_matrix,
                       topic_words, perplexity, coherence_proxy
    """
    cache_file = CACHE_DIR / f"lda_{cache_key}_k{n_topics}.pkl"
    if cache_key and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    vec = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        stop_words=None,  # already cleaned
    )
    dtm = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=30,
        learning_method="online",
        batch_size=256,
        n_jobs=1,
    )
    doc_topic = lda.fit_transform(dtm)
    perplexity = lda.perplexity(dtm)

    # Top words per topic
    topic_words = {}
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-20:][::-1]
        topic_words[idx] = [(feature_names[i], float(comp[i])) for i in top_idx]

    result = {
        "model": lda,
        "vectorizer": vec,
        "doc_topic_matrix": doc_topic,
        "topic_words": topic_words,
        "feature_names": feature_names,
        "perplexity": perplexity,
        "n_topics": n_topics,
        "dtm": dtm,
    }

    if cache_key:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f, protocol=4)
    return result


def run_nmf(
    corpus: List[str],
    n_topics: int = 8,
    max_features: int = 3000,
    cache_key: str = "",
) -> Dict:
    """
    Run NMF topic modelling as an alternative / comparison.
    """
    cache_file = CACHE_DIR / f"nmf_{cache_key}_k{n_topics}.pkl"
    if cache_key and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
    )
    dtm = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()

    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=300,
        init="nndsvda",
    )
    doc_topic = nmf.fit_transform(dtm)

    # Reconstruction error as quality proxy
    recon_error = nmf.reconstruction_err_

    topic_words = {}
    for idx, comp in enumerate(nmf.components_):
        top_idx = comp.argsort()[-20:][::-1]
        topic_words[idx] = [(feature_names[i], float(comp[i])) for i in top_idx]

    result = {
        "model": nmf,
        "vectorizer": vec,
        "doc_topic_matrix": doc_topic,
        "topic_words": topic_words,
        "feature_names": feature_names,
        "recon_error": recon_error,
        "n_topics": n_topics,
        "dtm": dtm,
    }

    if cache_key:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f, protocol=4)
    return result


# ─── Topic labelling heuristic ────────────────────────────────────────────────

KNOWN_TOPIC_LABELS = {
    frozenset(["neural", "network", "deep", "learn", "layer", "train", "loss"]):
        "Deep Neural Networks",
    frozenset(["language", "model", "text", "bert", "gpt", "token", "nlp", "word"]):
        "Natural Language Processing",
    frozenset(["image", "object", "detect", "vision", "convolution", "pixel", "segment"]):
        "Computer Vision",
    frozenset(["reinforcement", "reward", "policy", "agent", "environment", "action"]):
        "Reinforcement Learning",
    frozenset(["graph", "node", "edge", "network", "embedding", "link", "knowledge"]):
        "Graph Learning & Knowledge Graphs",
    frozenset(["federat", "privaci", "distribut", "commun", "client", "aggreg"]):
        "Federated & Distributed Learning",
    frozenset(["explain", "interpret", "fair", "bias", "saliency", "counterfactual"]):
        "Explainability & Fairness",
    frozenset(["robot", "grasp", "manipul", "motion", "plan", "navigation"]):
        "Robotics & Autonomous Systems",
    frozenset(["quantum", "qubit", "circuit", "entangl", "variational"]):
        "Quantum Computing",
    frozenset(["medical", "clinical", "patient", "drug", "protein", "genomic", "cancer"]):
        "Biomedical & Health AI",
    frozenset(["generat", "diffusion", "adversarial", "synthes", "style", "image"]):
        "Generative Models",
    frozenset(["optim", "gradient", "convex", "convergence", "stochastic", "bound"]):
        "Optimisation Theory",
    frozenset(["uncertain", "bayesian", "probabilistic", "inference", "prior", "posterior"]):
        "Bayesian & Probabilistic ML",
    frozenset(["speech", "audio", "acoustic", "recogni", "speaker", "asr"]):
        "Speech & Audio Processing",
    frozenset(["transfer", "domain", "adapt", "finetun", "pretrain", "few-shot", "zero-shot"]):
        "Transfer Learning & Adaptation",
}


def label_topic(words: List[str]) -> str:
    """Assign a human-readable label to a topic based on its top words."""
    word_set = set(w[:6] for w in words)  # stem-like prefix matching
    best_label = None
    best_overlap = 0
    for key, label in KNOWN_TOPIC_LABELS.items():
        overlap = len(word_set & key)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    if best_label and best_overlap >= 2:
        return best_label
    # Fallback: capitalise top 3 words
    return " / ".join(w.title() for w in words[:3])


# ─── Top Terms (TF-IDF based) ─────────────────────────────────────────────────

def extract_top_terms(
    df: pd.DataFrame,
    text_col: str = "combined_text",
    top_n: int = 50,
    by_category: bool = False,
) -> pd.DataFrame:
    """
    Extract top TF-IDF terms globally or per category.
    Returns DataFrame with columns: term, score, [category]
    """
    cleaned = preprocess_corpus(df[text_col].tolist())

    if not by_category:
        _, _, feature_names = build_tfidf(cleaned, max_features=5000)
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3, max_df=0.85, sublinear_tf=True)
        mat = vec.fit_transform(cleaned)
        mean_scores = np.asarray(mat.mean(axis=0)).flatten()
        fn = vec.get_feature_names_out()
        idx = mean_scores.argsort()[-top_n:][::-1]
        return pd.DataFrame({"term": fn[idx], "score": mean_scores[idx]})

    # Per-category version
    rows = []
    for cat in df["primary_cat"].unique():
        sub = df[df["primary_cat"] == cat]
        sub_cleaned = preprocess_corpus(sub[text_col].tolist())
        if len(sub_cleaned) < 5:
            continue
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=2, max_df=0.90, sublinear_tf=True)
        try:
            mat = vec.fit_transform(sub_cleaned)
            mean_scores = np.asarray(mat.mean(axis=0)).flatten()
            fn = vec.get_feature_names_out()
            idx = mean_scores.argsort()[-20:][::-1]
            for i in idx:
                rows.append({"term": fn[i], "score": mean_scores[i], "category": cat})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ─── Concept Co-occurrence Network ───────────────────────────────────────────

def build_concept_network(
    df: pd.DataFrame,
    text_col: str = "combined_text",
    top_terms: int = 60,
    min_cooccurrence: int = 3,
    cache_key: str = "",
) -> Tuple:
    """
    Build a concept co-occurrence network.
    Returns (nodes_df, edges_df) where edges have 'source', 'target', 'weight'.
    """
    cache_file = CACHE_DIR / f"network_{cache_key}.pkl"
    if cache_key and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    cleaned = preprocess_corpus(df[text_col].tolist())

    # Get top TF-IDF terms
    vec = TfidfVectorizer(max_features=top_terms * 3, ngram_range=(1,1), min_df=4, max_df=0.80, sublinear_tf=True)
    tfidf_mat = vec.fit_transform(cleaned)
    fn = vec.get_feature_names_out()
    mean_scores = np.asarray(tfidf_mat.mean(axis=0)).flatten()
    selected_idx = mean_scores.argsort()[-top_terms:][::-1]
    selected_terms = fn[selected_idx]

    # Binary co-occurrence
    binary_vec = CountVectorizer(vocabulary=selected_terms)
    bin_mat = (binary_vec.fit_transform(cleaned) > 0).astype(int)

    # Co-occurrence matrix
    cooc = (bin_mat.T @ bin_mat).toarray()
    np.fill_diagonal(cooc, 0)

    # Build edge list
    edges = []
    n = len(selected_terms)
    for i in range(n):
        for j in range(i + 1, n):
            w = int(cooc[i, j])
            if w >= min_cooccurrence:
                edges.append({
                    "source": selected_terms[i],
                    "target": selected_terms[j],
                    "weight": w,
                })

    edges_df = pd.DataFrame(edges)
    if edges_df.empty:
        edges_df = pd.DataFrame(columns=["source", "target", "weight"])

    # Node attributes
    term_freq = {selected_terms[i]: float(mean_scores[selected_idx[i]]) for i in range(n)}
    nodes_df = pd.DataFrame({
        "term": selected_terms,
        "tfidf_score": [term_freq[t] for t in selected_terms],
        "degree": [len(edges_df[(edges_df.source == t) | (edges_df.target == t)]) for t in selected_terms],
    })

    result = (nodes_df, edges_df)
    if cache_key:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f, protocol=4)
    return result


# ─── Cache key helper ─────────────────────────────────────────────────────────

def make_cache_key(df: pd.DataFrame) -> str:
    """Generate a stable cache key from a DataFrame's index and size."""
    h = hashlib.md5(str(len(df)).encode() + str(df.index.tolist()[:10]).encode()).hexdigest()[:8]
    return h
