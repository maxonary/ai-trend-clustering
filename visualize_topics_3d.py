import argparse
import plotly.graph_objects as go
import umap
from bertopic import BERTopic


def visualize_topics_3d(model_dir: str = "topic_model", output_html: str = "topics_3d.html", n_neighbors: int = 15, min_dist: float = 0.1):
    """Create an interactive 3-D scatter plot of topic relations.

    Parameters
    ----------
    model_dir : str
        Path to the saved BERTopic model (folder or .pt file).
    output_html : str
        File to write the interactive Plotly figure to.
    n_neighbors : int
        UMAP neighbours (affects the balance between local/global structure).
    min_dist : float
        UMAP min_dist parameter (controls clustering tightness).
    """

    model: BERTopic = BERTopic.load(model_dir)

    # Topic embeddings (shape: n_topics x embedding_dim)
    topic_embeddings = model.topic_embeddings_
    topic_info = model.get_topic_info()
    # Remove outlier topic -1 if present
    topic_info = topic_info[topic_info.Topic != -1]
    embeddings = topic_embeddings[topic_info.Topic.values]

    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    # Bubble size proportional to topic frequency
    sizes = topic_info["Count"].values
    sizes = 20 + 40 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)

    hover_text = [
        f"<b>Topic {row.Topic}</b><br>Size: {row['Count']}<br>{model.get_topic(row.Topic)[:5]}"  # show top 5 terms
        for _, row in topic_info.iterrows()
    ]

    fig = go.Figure(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=sizes,
            color=coords[:, 0],  # colour by first component
            colorscale="Viridis",
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo="text"
    ))

    fig.update_layout(title="3-D Topic Relation Map", margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(output_html)
    print(f"[visualize_topics_3d] Saved interactive 3-D plot ➜ {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BERTopic topic relations in 3-D space.")
    parser.add_argument("--model", default="topic_model", help="Path to the saved BERTopic model.")
    parser.add_argument("--out", default="topics_3d.html", help="Output HTML filename.")
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist parameter.")

    args = parser.parse_args()
    visualize_topics_3d(args.model, args.out, args.neighbors, args.min_dist) 