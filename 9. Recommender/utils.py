import numpy as np
import faiss
from annoy import AnnoyIndex
import networkx as nx
from loguru import logger
from constants import CACHE_PATHS, CACHE_DIR
import pickle

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build and return a FAISS index for fast similarity search"""
    if CACHE_PATHS['faiss_index'].exists():
        logger.info("Loading cached FAISS index...")
        with open(CACHE_PATHS['faiss_index'], 'rb') as f:
            index = pickle.load(f)
        logger.success("FAISS index loaded successfully!")
        return index
    else:
        logger.info("Building FAISS index...")
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(embeddings.astype('float32'))
        logger.success(f"FAISS index built with {index.ntotal} vectors")
        
        # Cache the index
        logger.info("Caching FAISS index...")
        with open(CACHE_PATHS['faiss_index'], 'wb') as f:
            pickle.dump(index, f)
        logger.success("FAISS index cached successfully!")
        
        return index

def build_ann_index(embeddings: np.ndarray, n_trees: int = 10) -> AnnoyIndex:
    """Build and return an Annoy index for approximate nearest neighbor search"""
    if CACHE_PATHS['annoy_index'].exists():
        logger.info("Loading cached Annoy index...")
        index = AnnoyIndex(embeddings.shape[1], 'angular')
        index.load(str(CACHE_PATHS['annoy_index']))
        logger.success("Annoy index loaded successfully!")
        return index
    else:
        logger.info("Building Annoy index...")
        # Initialize Annoy index
        dimension = embeddings.shape[1]
        index = AnnoyIndex(dimension, 'angular')
        
        # Add items to the index
        for i in range(len(embeddings)):
            index.add_item(i, embeddings[i])
        
        # Build the index with specified number of trees
        index.build(n_trees)
        logger.info("Saving Annoy index to disk...")
        index.save(str(CACHE_PATHS['annoy_index']))
        logger.success(f"Annoy index built with {len(embeddings)} vectors and {n_trees} trees")
        
        return index

def cache_normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings for cosine similarity calculations"""
    if CACHE_PATHS['embeddings_norm'].exists():
        logger.info("Loading cached normalized embeddings...")
        with open(CACHE_PATHS['embeddings_norm'], 'rb') as f:
            normalized = pickle.load(f)
        logger.success("Normalized embeddings loaded successfully!")
        return normalized
    else:
        logger.info("Normalizing embeddings...")
        
        # Calculate L2 norm of embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        
        # Normalize embeddings
        normalized = embeddings / norms
        
        logger.info("Caching normalized embeddings...")
        with open(CACHE_PATHS['embeddings_norm'], 'wb') as f:
            pickle.dump(normalized, f)
        logger.success("Embeddings normalized and cached successfully")
        
        return normalized

def cache_graph_metrics(graph: nx.Graph) -> dict:
    """Calculate and cache various graph metrics"""
    if CACHE_PATHS['graph_metrics'].exists():
        logger.info("Loading cached graph metrics...")
        with open(CACHE_PATHS['graph_metrics'], 'rb') as f:
            metrics = pickle.load(f)
        logger.success("Graph metrics loaded successfully!")
        return metrics
    
    logger.info("Calculating graph metrics...")
    metrics = {
        'pagerank': nx.pagerank(graph),
        'degree': dict(graph.degree()),
        'core_number': nx.core_number(graph)
    }
    
    # Normalize metrics to [0,1] range
    for metric_name, metric_dict in metrics.items():
        values = np.array(list(metric_dict.values()))
        min_val = values.min()
        max_val = values.max()
        
        # Avoid division by zero
        if max_val - min_val > 0:
            metrics[metric_name] = {
                k: (v - min_val) / (max_val - min_val)
                for k, v in metric_dict.items()
            }
        else:
            logger.warning(f"Could not normalize {metric_name} - all values are identical")
            metrics[metric_name] = {k: 0.0 for k in metric_dict.keys()}
    
    logger.info("Caching graph metrics...")
    with open(CACHE_PATHS['graph_metrics'], 'wb') as f:
        pickle.dump(metrics, f)
    logger.success("Graph metrics calculated, normalized and cached successfully")
    
    return metrics