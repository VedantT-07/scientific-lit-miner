"""
data_loader.py
──────────────
Handles dataset retrieval (kagglehub), caching, cleaning, and subsetting.
Designed to be memory-efficient for laptops with limited RAM.
"""

import os
import json
import pickle
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Ensure ~/.kaggle/kaggle.json exists if env vars are provided in .env
def _sync_kaggle_credentials():
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        try:
            kdir = Path.home() / ".kaggle"
            kdir.mkdir(exist_ok=True)
            kfile = kdir / "kaggle.json"
            if not kfile.exists():
                with open(kfile, "w") as f:
                    json.dump({"username": user.strip(), "key": key.strip()}, f)
        except Exception as e:
            logger.warning(f"Could not sync kaggle.json: {e}")

_sync_kaggle_credentials()

# ─── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
PROCESSED_CACHE = CACHE_DIR / "processed_papers.pkl"
RAW_CACHE       = CACHE_DIR / "raw_sample.pkl"

# ─── Categories of interest (CS + interdisciplinary) ────────────────────────
TARGET_CATEGORIES = {
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR", "cs.NE",
    "cs.RO", "cs.HC", "cs.SE", "cs.DB", "cs.DC", "cs.NI",
    "stat.ML", "stat.AP", "eess.AS", "eess.IV", "quant-ph",
    "math.OC", "physics.comp-ph",
}

CATEGORY_LABELS = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation & Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.IR": "Information Retrieval",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.HC": "Human-Computer Interaction",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.DC": "Distributed Computing",
    "cs.NI": "Networking & Internet",
    "stat.ML": "Statistics – Machine Learning",
    "stat.AP": "Statistics – Applications",
    "eess.AS": "Audio & Speech Processing",
    "eess.IV": "Image & Video Processing",
    "quant-ph": "Quantum Computing",
    "math.OC": "Optimization & Control",
    "physics.comp-ph": "Computational Physics",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Basic unicode and whitespace normalisation."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # strip non-ASCII
    text = text.strip()
    return text


def _extract_year(date_str) -> Optional[int]:
    """Extract 4-digit year from an arXiv date string like '2021-03-14'."""
    if not isinstance(date_str, str):
        return None
    m = re.search(r'(\d{4})', date_str)
    return int(m.group(1)) if m else None


def _primary_category(cats_str) -> str:
    """Return the first category from a space-separated string."""
    if not isinstance(cats_str, str):
        return "unknown"
    return cats_str.strip().split()[0]


# ─── Core loader ─────────────────────────────────────────────────────────────

def load_dataset(
    n_papers: int = 3000,
    force_reload: bool = False,
    categories: Optional[set] = None,
) -> pd.DataFrame:
    """
    Load, filter, and preprocess a subset of the arXiv dataset.

    Returns a DataFrame with columns:
        id, title, abstract, categories, primary_cat,
        year, combined_text, title_clean, abstract_clean
    """
    if not force_reload and PROCESSED_CACHE.exists():
        logger.info("Loading preprocessed data from cache …")
        with open(PROCESSED_CACHE, "rb") as f:
            df = pickle.load(f)
        # If cached version has fewer papers than requested, reload
        if len(df) >= min(n_papers, 2000):
            return df

    # ── Step 1: try kagglehub ────────────────────────────────────────────────
    raw_path = _fetch_raw_dataset()

    # ── Step 2: parse JSON-lines ─────────────────────────────────────────────
    df = _parse_jsonl(raw_path, categories or TARGET_CATEGORIES, n_papers)

    if df is None or len(df) == 0:
        raise RuntimeError(
            "Dataset could not be loaded. Check your Kaggle credentials "
            "and internet connection. See README for setup instructions."
        )

    # ── Step 3: clean & enrich ───────────────────────────────────────────────
    df = _preprocess(df)

    # ── Step 4: cache ────────────────────────────────────────────────────────
    with open(PROCESSED_CACHE, "wb") as f:
        pickle.dump(df, f, protocol=4)

    logger.info(f"Loaded {len(df):,} papers and saved to cache.")
    return df


def _fetch_raw_dataset() -> Optional[Path]:
    """
    Try kagglehub first; fall back to a bundled demo CSV if unavailable.
    Returns the path to the main JSON-lines file (arxiv-metadata-oai-snapshot.json).
    """
    # Check cache first
    raw_json_candidates = list(CACHE_DIR.glob("**/*.json"))
    for c in raw_json_candidates:
        if "arxiv" in c.name.lower() and c.stat().st_size > 1_000_000:
            logger.info(f"Using cached raw file: {c}")
            return c

    try:
        import kagglehub
        logger.info("Downloading dataset via kagglehub …")
        path = kagglehub.dataset_download("Cornell-University/arxiv")
        base = Path(path)
        # Find the JSONL file
        for f in base.rglob("*.json"):
            if f.stat().st_size > 1_000_000:
                return f
    except Exception as e:
        logger.warning(f"kagglehub download failed: {e}")

    # Try environment variable path
    env_path = os.environ.get("ARXIV_DATASET_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    return None


def _parse_jsonl(
    path: Optional[Path],
    categories: set,
    n_papers: int,
) -> Optional[pd.DataFrame]:
    """Stream-parse the arXiv JSON-lines file, keeping only relevant papers."""
    if path is None or not path.exists():
        logger.warning("Raw dataset path not found – using demo data.")
        return _generate_demo_data(n_papers)

    records = []
    cat_list = list(categories)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by category
                cats = obj.get("categories", "")
                if not any(c in cats for c in cat_list):
                    continue

                # Only papers from 2010 onward
                year = _extract_year(obj.get("update_date", ""))
                if year is None or year < 2010:
                    continue

                title    = obj.get("title", "")
                abstract = obj.get("abstract", "")
                if len(abstract) < 80:  # skip stub entries
                    continue

                records.append({
                    "id":         obj.get("id", ""),
                    "title":      title,
                    "abstract":   abstract,
                    "categories": cats,
                    "year":       year,
                    "authors":    obj.get("authors", ""),
                })

                if len(records) >= n_papers:
                    break

        logger.info(f"Parsed {len(records):,} papers from raw file.")
        return pd.DataFrame(records) if records else None

    except Exception as e:
        logger.error(f"Error parsing JSONL: {e}")
        return _generate_demo_data(n_papers)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text and add derived columns."""
    df = df.copy()

    df["title_clean"]    = df["title"].apply(_clean_text)
    df["abstract_clean"] = df["abstract"].apply(_clean_text)
    df["primary_cat"]    = df["categories"].apply(_primary_category)
    df["cat_label"]      = df["primary_cat"].map(CATEGORY_LABELS).fillna(df["primary_cat"])

    # Combined text for NLP
    df["combined_text"] = (df["title_clean"] + " " + df["abstract_clean"]).str.strip()

    # Drop rows with too-short text
    df = df[df["combined_text"].str.len() > 100].reset_index(drop=True)

    # Year as integer
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


# ─── Demo data (fallback) ─────────────────────────────────────────────────────

def _generate_demo_data(n: int = 3000) -> pd.DataFrame:
    """
    Generate a realistic-looking demo dataset when Kaggle is unavailable.
    Each record is clearly marked as DEMO so the UI can warn the user.
    """
    import random
    rng = random.Random(42)
    np.random.seed(42)

    topics = {
        "Deep Learning":    ["neural network", "deep learning", "convolutional", "transformer", "attention mechanism", "gradient descent", "batch normalization", "dropout", "LSTM", "ResNet"],
        "NLP":              ["natural language processing", "language model", "text classification", "sentiment analysis", "named entity recognition", "machine translation", "BERT", "GPT", "tokenization", "word embeddings"],
        "Computer Vision":  ["image recognition", "object detection", "semantic segmentation", "generative adversarial", "image synthesis", "optical flow", "3D reconstruction", "face recognition", "point cloud", "depth estimation"],
        "Reinforcement Learning": ["reinforcement learning", "policy gradient", "reward function", "Markov decision process", "Q-learning", "actor-critic", "exploration strategy", "multi-agent", "environment model", "value function"],
        "Graph Learning":   ["graph neural network", "knowledge graph", "graph convolution", "node classification", "link prediction", "graph attention", "heterogeneous graph", "subgraph", "network embedding", "community detection"],
        "Federated Learning":["federated learning", "privacy-preserving", "distributed training", "communication efficiency", "differential privacy", "data heterogeneity", "model aggregation", "edge computing", "secure aggregation", "non-IID"],
        "Explainable AI":   ["explainability", "interpretability", "saliency map", "SHAP", "LIME", "feature importance", "attention visualization", "counterfactual", "fairness", "bias detection"],
        "Robotics":         ["robot learning", "manipulation", "autonomous navigation", "sim-to-real", "motion planning", "grasping", "human-robot interaction", "SLAM", "tactile sensing", "dexterous hand"],
        "Quantum ML":       ["quantum computing", "quantum circuit", "variational quantum", "qubit", "quantum advantage", "noise mitigation", "quantum annealing", "hybrid quantum-classical", "entanglement", "quantum kernel"],
        "Biomedical AI":    ["medical imaging", "drug discovery", "protein structure", "genomics", "clinical NLP", "electronic health records", "biomarker", "pathology", "radiology", "cancer detection"],
    }

    topic_names = list(topics.keys())
    cat_map = {
        "Deep Learning": "cs.LG", "NLP": "cs.CL", "Computer Vision": "cs.CV",
        "Reinforcement Learning": "cs.AI", "Graph Learning": "cs.LG",
        "Federated Learning": "cs.DC", "Explainable AI": "cs.AI",
        "Robotics": "cs.RO", "Quantum ML": "quant-ph", "Biomedical AI": "cs.AI",
    }

    records = []
    for i in range(n):
        topic  = rng.choice(topic_names)
        terms  = topics[topic]
        year   = rng.randint(2015, 2024)
        w1, w2 = rng.sample(terms, 2)

        title = f"[DEMO] {topic}: {w1.title()} and {w2.title()} in Modern Research"
        abstract = (
            f"[DEMO DATA – not real] This paper investigates {w1} combined with {w2} "
            f"for {topic.lower()} applications. We propose a novel approach leveraging "
            f"{rng.choice(terms)} to address limitations in existing methods. "
            f"Experimental results on benchmark datasets demonstrate significant improvements. "
            f"Our analysis reveals that {rng.choice(terms)} plays a crucial role in "
            f"achieving state-of-the-art performance. Future work will explore {rng.choice(terms)}."
        )
        records.append({
            "id": f"demo.{i:05d}",
            "title": title,
            "abstract": abstract,
            "categories": cat_map[topic],
            "year": year,
            "authors": f"Author {i % 500}",
        })

    return pd.DataFrame(records)
