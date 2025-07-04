import umap
import json
import pathlib
import datetime as dt

from bertopic import BERTopic
import plotly.graph_objects as go
import streamlit as st

from fetcher import fetch_arxiv
from embedder import generate_embeddings
from cluster import run_clustering

# ------------------ UI ------------------
st.title("BERTopic Explorer")

def list_run_dirs(root="runs"):
    root_p = pathlib.Path(root)
    if not root_p.exists():
        return []
    return sorted([d for d in root_p.iterdir() if d.is_dir()], reverse=True)

# Sidebar: existing runs selector
run_dirs = list_run_dirs()
selected_run = st.sidebar.selectbox("Select run to explore", run_dirs, format_func=str)

# derive paths from selection
if selected_run:
    model_path = str(selected_run / "topic_model")
    metadata_path = str(selected_run / "papers.json")
else:
    model_path = "topic_model"
    metadata_path = "arxiv_papers.json"

# Sidebar controls ----------------------
st.sidebar.header("Pipeline Controls")
category = st.sidebar.text_input("arXiv category", "cs.CL")
start_year = st.sidebar.number_input("Start year", 2000, 2030, 2020)
max_results = st.sidebar.number_input("Max results", 100, 10000, 500, step=100)
if st.sidebar.button("Run fetch→embed→cluster"):

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = pathlib.Path("runs") / f"{timestamp}_{category}"
    run_dir.mkdir(parents=True, exist_ok=True)

    papers_json = run_dir / "papers.json"
    embeddings_file = run_dir / "embeddings.npy"
    model_dir = run_dir / "topic_model"

    with st.spinner("Fetching abstracts from arXiv…"):
        fetch_arxiv(category, max_results, start_year, str(papers_json))

    with st.spinner("Generating embeddings…"):
        generate_embeddings(str(papers_json), out_file=str(embeddings_file))

    with st.spinner("Clustering topics…"):
        run_clustering(str(papers_json), str(embeddings_file), str(model_dir))

    run_dirs.insert(0, run_dir)
    st.sidebar.success("Pipeline completed → new run added to selector")

# Tabs for visualisation and docs
tab3d, tabTrend, tabInfo = st.tabs(["3-D Topic Map", "Topic Trends", "How it works"])

# ---------- Load Model (cached) ----------
@st.cache_resource  # prevents re-loading model every slider move
def load_model(path):
    return BERTopic.load(path)

model = load_model(model_path)

# -------------- 3-D MAP TAB --------------
with tab3d:
    st.subheader("3-D Topic Relation Map")
    st.markdown(
        "Each point is a topic; proximity indicates semantic similarity "
        "after projecting the high-dimensional topic embeddings to three "
        "UMAP components. Bubble size = topic frequency."
    )
    n_neighbors = st.slider("UMAP n_neighbors", 2, 100, 15)
    min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)

    topic_info = model.get_topic_info().query("Topic != -1")
    embeddings = model.topic_embeddings_[topic_info.Topic.values]

    coords = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings)

    sizes = topic_info.Count.values
    sizes = 20 + 40 * (sizes - sizes.min())/(sizes.max()-sizes.min()+1e-6)

    def short_label(tid: int, top_n: int = 3):
        return " ".join([w for w, _ in model.get_topic(tid)[:top_n]])

    hover = [
        f"<b>{short_label(row.Topic)}</b><br>(Topic {row.Topic} · {row.Count} docs)"
        for _, row in topic_info.iterrows()
    ]

    fig3d = go.Figure(
        go.Scatter3d(
            x=coords[:,0], y=coords[:,1], z=coords[:,2],
            mode="markers",
            marker=dict(size=sizes, color=coords[:,0],
                        colorscale="Viridis", opacity=0.8),
            text=hover, hoverinfo="text"
        )
    )
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Semantic Dim 1",
            yaxis_title="Semantic Dim 2",
            zaxis_title="Semantic Dim 3"
        ),
        margin=dict(l=0, r=0, b=0, t=20)
    )
    st.plotly_chart(fig3d, use_container_width=True)

# -------------- TREND TAB ---------------
with tabTrend:
    st.subheader("Topic Trends Over Time")
    nr_bins = st.slider("Number of time bins", 5, 40, 20)

    @st.cache_data(show_spinner=False)
    def load_metadata(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        abstracts = [d["abstract"] for d in data]
        dates = [dt.datetime.strptime(d["date"], "%Y-%m-%d") for d in data]
        return abstracts, dates

    try:
        abstracts, dates = load_metadata(metadata_path)

        @st.cache_data(show_spinner=False)
        def compute_topics_over_time(abstracts, dates, bins):
            return model.topics_over_time(abstracts, dates, nr_bins=bins)

        topics_over_time = compute_topics_over_time(tuple(abstracts), tuple(dates), nr_bins)

        figTrend = model.visualize_topics_over_time(topics_over_time)
        st.plotly_chart(figTrend, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate trends: {e}")


# ---------- INFO TAB -------------------
with tabInfo:
    st.subheader("How the Pipeline Works")
    st.markdown(
        """
    **Workflow**

    1. **Fetch** – Download recent arXiv abstracts (`arxiv_fetcher.py`).
    2. **Embed** – Convert abstracts to sentence embeddings with *all-MiniLM-L6-v2* (`embed.py`).
    3. **Cluster** – Use **BERTopic** (UMAP + HDBSCAN) to extract topic clusters (`cluster.py`).
    4. **Visualise** – Produce the 3-D topic map and time-series plots.

    ```
    arXiv API --> fetcher --> papers.json
                         |
                         v
                 embedder (SBERT) --> embeddings.npy
                         |
                         v
                BERTopic cluster  --> topic_model/
                         |
        +----------------+----------------+
        |                                 |
    3-D topic map                    Trend plot
    ```

    *UMAP axes are abstract components; bubbles indicate topic frequency.*
    """
    )