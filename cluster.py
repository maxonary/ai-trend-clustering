import json
import numpy as np
from bertopic import BERTopic
import argparse
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------
# Topic clustering script
# -------------------------

def run_clustering(metadata_path: str = "arxiv_papers.json", embeddings_path: str = "embeddings.npy", model_outdir: str = "topic_model", ngram_max: int = 2, min_df: int = 5):
    """Fit a BERTopic model on the pre-computed embeddings.

    Parameters
    ----------
    metadata_path : str
        Path to the JSON file with the document metadata (abstracts).
    embeddings_path : str
        Path to the `.npy` file containing the embeddings.
    model_outdir : str
        Directory where the BERTopic model files will be saved.
    ngram_max : int
        Maximum n-gram length for the vectorizer.
    min_df : int
        Ignore terms that appear in fewer than min_df documents.
    """

    with open(metadata_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    abstracts = [p["abstract"] for p in papers]
    embeddings = np.load(embeddings_path)

    # Custom vectorizer to remove stop-words and allow n-grams
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, ngram_max), min_df=min_df)

    topic_model = BERTopic(verbose=True,
                           calculate_probabilities=True,
                           language="english",
                           vectorizer_model=vectorizer)
    topic_model.fit(abstracts, embeddings)
    topic_model.save(model_outdir)

    print(f"[cluster] BERTopic model with {topic_model.get_topic_freq().shape[0] - 1} topics saved ➜ {model_outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster document embeddings into topics using BERTopic.")
    parser.add_argument("--metadata", default="arxiv_papers.json", help="Path to the JSON file containing paper metadata.")
    parser.add_argument("--embeddings", default="embeddings.npy", help="Path to the .npy embeddings file")
    parser.add_argument("--out", default="topic_model", help="Directory where the fitted topic model will be stored.")
    parser.add_argument("--ngram", type=int, default=2, help="Maximum n-gram length for the vectorizer (default=2).")
    parser.add_argument("--min_df", type=int, default=5, help="Ignore terms that appear in fewer than min_df documents (default=5).")

    args = parser.parse_args()
    run_clustering(args.metadata, args.embeddings, args.out, args.ngram, args.min_df)