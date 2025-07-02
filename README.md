# AI Trend Clustering 
## Topic Trend Analysis (2020-2025) with BERTopic and Sentence-Transformers

Clustering and visualizing topic trends in AI research papers using transformer-based sentence embeddings and BERTopic.

Inspired by [Benchmarking topic models on scientific articles using BERTeley](https://www.sciencedirect.com/science/article/pii/S2949719123000419) by Chagnon E, Pandolfi R, Donatelli J, Ushizima D.

![Diagram of the BERTeley workflow.](https://ars.els-cdn.com/content/image/1-s2.0-S2949719123000419-gr1.jpg)

- Fetches abstracts from arXiv (e.g., cs.CL, cs.AI)
- Generates embeddings via `sentence-transformers`
- Clusters topics with BERTopic (UMAP + HDBSCAN)
- Analyzes topic frequency over time
- Visualizes trends as interactive HTML plots

## Research
Exact project research paper exposé proposal found in [EXPOSE.md](EXPOSE.md).

## Usage
### Option A · Interactive (recommended)

Launch the Streamlit dashboard; it lets you fetch, embed, cluster and explore in one place:

```bash
streamlit run app.py
```

*Sidebar →* choose arXiv category, start-year & max-results, hit **Run fetch→embed→cluster**.  
A new folder is created under `runs/` (e.g. `runs/2024-07-03_1508_cs.CL/`) containing

```
papers.json      # raw abstracts
embeddings.npy   # sentence-transformer vectors
topic_model/     # BERTopic artefacts
```

Tabs provide a 3-D topic map, temporal trend plot and a pipeline explanation.

### Option B · Command-line pipeline

```bash
# run everything head-less, results end up in runs/<timestamp>_<cat>/
python -m src.pipeline --category cs.CL --start-year 2020 --max-results 1000
```

### Option C · Individual steps (advanced)

```bash
python -m src.fetcher   --category cs.CL --start_year 2020 --max_results 1000 --out mypapers.json
python -m src.embedder  --metadata mypapers.json --out myemb.npy
python -m src.cluster   --metadata mypapers.json --embeddings myemb.npy --out my_topic_model
```

## Output
- `runs/<timestamp>_<category>/topic_model/` → loadable by BERTopic & Streamlit  
- `runs/<...>/papers.json` & `embeddings.npy` for downstream analysis  
- HTML exports (`topics_3d.html`, `topic_trends.html`) can be generated from the dashboard

## Key Features
- **Automated data ingestion** from the arXiv API (`cs.AI`, `cs.CL`, `cs.LG`, configurable).
- **Sentence‑Transformers embeddings** (`all‑MiniLM‑L6‑v2` by default).
- **Unsupervised topic discovery** with **BERTopic** (UMAP + HDBSCAN).
- **Trends over time** (papers per topic per month/year) with interactive HTML visualisations.
- **Quick evaluation** via topic‑coherence metric and manual inspection helpers.

---

## Tech Stack
| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| Data Source | arXiv API |
| Embeddings | `sentence-transformers` |
| Topic Modelling | **BERTopic** (UMAP + HDBSCAN) |
| Data Wrangling | pandas, numpy |
| Visualisation | Plotly, Matplotlib |
| Notebooks (optional) | Jupyter / `.ipynb` |

## Quick‑Start

### 1 · Clone & Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 · Run the Pipeline
```bash
# download abstracts (customise category/date)
python arxiv_fetcher.py --category cs.CL --start_year 2020
# generate embeddings
python embed.py
# create topic model
python cluster.py
# build interactive trend plot
python trends.py --bins 20
```
The resulting **topic_trends.html** opens in any browser.

## Evaluation
Run
```bash
python evaluation.py
```
to print the overall topic‑coherence score.  
Manual sanity‑check notebooks are located in `notebooks/`.