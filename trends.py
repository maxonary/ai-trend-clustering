import json
from datetime import datetime
from bertopic import BERTopic
import argparse

def visualize_trends(metadata_path: str = "arxiv_papers.json", model_dir: str = "topic_model", output_html: str = "topic_trends.html", nr_bins: int = 20):
    """Generate a Plotly HTML of topic frequency over time.

    Parameters
    ----------
    metadata_path : str
        Path to the JSON file with paper metadata.
    model_dir : str
        Directory containing the serialized BERTopic model.
    output_html : str
        Path of the output HTML file.
    nr_bins : int
        Number of temporal bins to aggregate over (passed to `topics_over_time`).
    """

    with open(metadata_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    abstracts = [p["abstract"] for p in papers]
    dates = [datetime.strptime(p["date"], "%Y-%m-%d") for p in papers]

    model = BERTopic.load(model_dir)
    topics_over_time = model.topics_over_time(abstracts, dates, nr_bins=nr_bins)

    fig = model.visualize_topics_over_time(topics_over_time)
    fig.write_html(output_html)
    print(f"[trends] Topic trend visualization saved ➜ {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize topic frequency over time using a fitted BERTopic model.")
    parser.add_argument("--metadata", default="arxiv_papers.json", help="Path to the JSON file with paper metadata.")
    parser.add_argument("--model", default="topic_model", help="Directory where the BERTopic model is stored.")
    parser.add_argument("--out", default="topic_trends.html", help="HTML file to write.")
    parser.add_argument("--bins", type=int, default=20, help="Number of time bins for the visualization.")

    args = parser.parse_args()
    visualize_trends(args.metadata, args.model, args.out, args.bins)