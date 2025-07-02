import streamlit as st
import umap
import plotly.graph_objects as go
from bertopic import BERTopic

st.title("3-D Topic Relation Explorer")

model_path = st.text_input("BERTopic model path", "topic_model")
n_neighbors = st.slider("UMAP n_neighbors", 2, 100, 15)
min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)

@st.cache_resource  # prevents re-loading model every slider move
def load_model(path):
    return BERTopic.load(path)

model = load_model(model_path)

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

fig = go.Figure(
    go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        mode="markers",
        marker=dict(size=sizes, color=coords[:,0],
                    colorscale="Viridis", opacity=0.8),
        text=hover, hoverinfo="text"
    )
)
fig.update_layout(margin=dict(l=0,r=0,b=0,t=20))

st.plotly_chart(fig, use_container_width=True)