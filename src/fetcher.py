import arxiv
import json
import argparse
from tqdm import tqdm
from arxiv import UnexpectedEmptyPageError

def fetch_arxiv(category: str = "cs.CL", max_results: int = 500, start_year: int = 2020, out_file: str = "arxiv_papers.json"):
    """Fetch metadata from arXiv using the newer arxiv.Client API.

    Parameters
    ----------
    category : str
        arXiv category to query (e.g. "cs.CL", "cs.LG").
    max_results : int
        Maximum number of results to retrieve *after* the `start_year` filter.
    start_year : int
        Only include papers published on/after the given year.
    out_file : str
        Path to the JSON file in which to store the fetched metadata.
    """

    search = arxiv.Search(query=f"cat:{category}",
                          sort_by=arxiv.SortCriterion.SubmittedDate,
                          sort_order=arxiv.SortOrder.Descending)

    client = arxiv.Client(page_size=100, delay_seconds=3)  # be gentle on the API

    papers = []
    iterator = client.results(search)
    with tqdm(total=max_results, desc="Downloading metadata", unit="paper") as pbar:
        while len(papers) < max_results:
            try:
                res = next(iterator)
            except StopIteration:
                break  # no more results
            except UnexpectedEmptyPageError:
                # arXiv returned an empty page; treat as end of results
                break

            if res.published.year < start_year:
                break  # older than requested window

            papers.append({
                "title": res.title.strip().replace("\n", " "),
                "abstract": res.summary.strip().replace("\n", " "),
                "date": res.published.strftime("%Y-%m-%d")
            })
            pbar.update(1)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"[arxiv_fetcher] Saved {len(papers)} papers ➜ {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch recent papers from arXiv and store simple metadata as JSON.")
    parser.add_argument("--category", default="cs.CL", help="arXiv category to query, e.g. 'cs.CL'.")
    parser.add_argument("--max_results", type=int, default=500, help="Maximum number of papers to store after filtering by year.")
    parser.add_argument("--start_year", type=int, default=2020, help="Only include papers published on/after this year.")
    parser.add_argument("--out", default="arxiv_papers.json", help="Path to the output JSON file.")

    args = parser.parse_args()
    fetch_arxiv(args.category, args.max_results, args.start_year, args.out)