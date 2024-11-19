from pathlib import Path
CACHE_DIR = Path('recommendation_cache')
CACHE_PATHS = {
    'graph': CACHE_DIR / 'final_complete_graph.pkl',
    'dataframe': CACHE_DIR / 'final_graph_dataframe.pkl',
    'annoy_index': CACHE_DIR / 'annoy_index.ann',
    'faiss_index': CACHE_DIR / 'faiss_index.pkl', 
    'graph_metrics': CACHE_DIR / 'graph_metrics.pkl',
    'embeddings_norm': CACHE_DIR / 'normalized_embeddings.pkl',
    'degree_scores': CACHE_DIR / 'degree_scores.pkl',
    'model': CACHE_DIR / 'model_cache.pkl',
    'node_embeddings': CACHE_DIR / 'node_embeddings.pt'
}