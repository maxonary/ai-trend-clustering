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

## Usage
1. Download metadata (customise with flags):
   ```bash
   python arxiv_fetcher.py --category cs.CL --start_year 2020 --max_results 500
   ```
2. Compute embeddings (optionally choose a different model):
   ```bash
   python embed.py --model all-MiniLM-L6-v2
   ```
3. Train the BERTopic model:
   ```bash
   python cluster.py
   ```
4. Build the interactive timeline visualisation:
   ```bash
   python trends.py --bins 20
   ```

## Output
- `topic_trends.html`: Interactive visualization of topic evolution

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