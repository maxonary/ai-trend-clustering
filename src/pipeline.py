import click 
import datetime
import pathlib
from src.fetcher import fetch_arxiv
from src.embedder import generate_embeddings
from src.cluster import run_clustering

@click.command()
@click.option('--category', default='cs.CL')
@click.option('--start-year', default=2020, type=int)
@click.option('--max-results', default=500, type=int)
def run(category, start_year, max_results):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
    run_dir = pathlib.Path('runs') / f'{timestamp}_{category}'
    run_dir.mkdir(parents=True)

    papers_json = run_dir / 'papers.json'
    fetch_arxiv(category, start_year, max_results, papers_json)

    embeddings_file = run_dir / 'embeddings.npy'
    generate_embeddings(papers_json, embeddings_file)

    model_dir = run_dir / 'topic_model'
    run_clustering(papers_json, embeddings_file, model_dir)

    print(f'Finished. Outputs in {run_dir}')

if __name__ == '__main__':
    run()
