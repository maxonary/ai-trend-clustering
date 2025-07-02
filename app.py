import streamlit as st
import umap
import plotly.graph_objects as go
from bertopic import BERTopic
import json
from datetime import datetime

# ------------------ UI ------------------
st.title("BERTopic Explorer")

model_path = st.text_input("BERTopic model path", "topic_model")
metadata_path = st.text_input("Paper metadata JSON", "arxiv_papers.json")

# Tabs for 3-D map and timeline visualisation
tab3d, tabTrend = st.tabs(["3-D Topic Map", "Topic Trends"])

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

    hover = [
        f"Topic {row.Topic}<br>Size: {row.Count}<br>{model.get_topic(row.Topic)[:5]}"
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
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3"
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
        dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in data]
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