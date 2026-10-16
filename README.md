# Scientific Literature Mining & Research Gap Discovery
## CEP 2 — Data Analysis & Visualisation Lab Project

---

## 📌 Project Summary

**LitMine** is an interactive web application that mines the [arXiv preprint dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) to:

- Identify major research themes via **Topic Modelling** (LDA & NMF)
- Track how topics evolve over time (**Trend Analysis**)
- Map relationships between technical concepts (**Concept Network**)
- Compare **LDA vs NMF** topic model quality
- Surface **evidence-backed Research Gaps** with supporting papers

Built with **Python + Streamlit + scikit-learn + Plotly + NetworkX**.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd c:\Projects\DAV_LabAssg
pip install -r requirements.txt
```

### 2. Set up Kaggle credentials (for real arXiv data)

**Option A — `.env` file (easiest, just like in JS/Node!):**
Open the `.env` file in the project folder and fill in:
```ini
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

**Option B — API token file:**
Place your downloaded `kaggle.json` at:
`C:\Users\<YourName>\.kaggle\kaggle.json`

**Option C — Terminal environment variables:**
```powershell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY      = "your_api_key"
```

> **No Kaggle account?** The app falls back to demo data automatically. All charts work — the data is labelled `[DEMO]` so you always know what you're looking at.

### 3. Run the application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
DAV_LabAssg/
├── app.py              # Main Streamlit application (7 interactive pages)
├── data_loader.py      # Dataset retrieval, filtering, preprocessing
├── nlp_engine.py       # TF-IDF, LDA, NMF, concept network building
├── visualizations.py   # All Plotly chart builders
├── gap_analysis.py     # Evidence-based research gap identification
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── cache/              # Auto-created: cached models and processed data
```

---

## 🗂 Application Pages

| Page | Description |
|------|-------------|
| **📊 Overview & Dataset** | KPI cards, publication trends, category distribution, methodology |
| **🧠 Topic Analysis** | Topic heatmap, topic word bars, word clouds, summary table |
| **📈 Trend Explorer** | Topic share over time, year-by-year breakdown, growth rates |
| **🕸️ Concept Network** | Interactive co-occurrence graph, top TF-IDF terms |
| **⚖️ Model Comparison** | LDA vs NMF quality curves (k=4–12), side-by-side topics |
| **🔍 Research Gaps** | Scored gaps, bubble chart, evidence papers, suggested directions |
| **📄 Paper Search** | Full-text search + filter by year, domain, topic + CSV export |

---

## ⚙️ Configuration

Adjust these from the **sidebar** at runtime:

| Setting | Default | Effect |
|---------|---------|--------|
| Papers to analyse | 2,500 | More = better coverage, slower |
| Number of topics (k) | 8 | Topic granularity |
| Topic model | LDA | Switch to NMF for comparison |

---

## 📊 Technical Implementation

### Text Processing Pipeline
1. Unicode normalisation, URL/email stripping
2. Domain-specific stop-word removal (100+ terms)
3. WordNet lemmatisation
4. Min/max document frequency filtering (3–85%)

### Topic Modelling
- **LDA** — Online variational Bayes, 30 iterations, batch=256
- **NMF** — NNDSVDA initialisation, 300 iterations, TF-IDF input
- Quality metrics: perplexity (LDA), reconstruction error (NMF)

### Research Gap Scoring
Each topic is scored (0–1) based on:
- **35%** Representation size (small = under-served)
- **25%** Publication recency (older = potentially stagnant)  
- **25%** Trend slope (declining = fading attention)
- **15%** Topic entropy (diffuse = interdisciplinary void)

### Caching Strategy
All heavy computations (model fitting, vectorisation, network building) are cached to `cache/` via `pickle`. Processing ~3,000 papers takes **~30–60 seconds on first run**, then loads instantly.

---

## 🧪 Requirements

- Python 3.9+
- RAM: ≥ 4 GB (tested with 2,500–4,000 papers)
- Storage: ~500 MB (dataset download + model cache)
- Internet: Required for initial dataset download (Kaggle)

---

## 📚 Dataset

**arXiv Dataset** — Cornell University / Kaggle  
- Source: [kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Content: 1.7M+ scholarly papers across STEM fields  
- Loaded subset: CS, ML, AI, NLP, CV, Robotics, Quantum Computing (2010–present)  
- Access: `kagglehub.dataset_download("Cornell-University/arxiv")`

---

## ⚠️ Limitations & Caveats

- Analysis is limited to the **sampled subset** (2,000–5,000 papers); full corpus may show different patterns.
- Research gaps are **statistical patterns**, not definitive findings — validate with domain experts.
- arXiv skews toward **preprints** (unreviewed) and CS/Physics fields.
- Topic labels are assigned via **lexical heuristics**; manual review improves accuracy.
- Demo data is **synthetic** — do not present it as real research findings.

---

*CEP 2 · Data Analysis & Visualisation Lab · Built with Python, Streamlit, scikit-learn, Plotly*
