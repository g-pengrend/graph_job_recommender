import torch
import pickle
import ast
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

from constants import CACHE_PATHS, CACHE_DIR
from utils import build_faiss_index, build_ann_index, cache_normalized_embeddings, cache_graph_metrics

def get_language_models():
    """Cache transformer model and graph embeddings"""
    logger.info(f"Loading resources from cache directory: {CACHE_DIR}")

    # Load graph
    if CACHE_PATHS['graph'].exists():
        with open(CACHE_PATHS['graph'], 'rb') as f:
            graph = pickle.load(f)
        logger.success("Graph loaded successfully.")
    else:
        logger.error(f"Error: Graph pickle file does not exist at {CACHE_PATHS['graph']}")
        raise FileNotFoundError(f"Graph file not found: {CACHE_PATHS['graph']}")

    # Load node embeddings
    if CACHE_PATHS['node_embeddings'].exists():
        logger.info("Loading node embeddings from cache...")
        node_embeddings = torch.load(CACHE_PATHS['node_embeddings'], weights_only=True)
        logger.success("Node embeddings loaded successfully!")
    else:
        logger.error(f"Node embeddings file not found: {CACHE_PATHS['node_embeddings']}")
        raise FileNotFoundError(f"Node embeddings file not found: {CACHE_PATHS['node_embeddings']}")

    # Load transformer model
    if CACHE_PATHS['model'].exists():
        logger.info("Loading transformer model from cache...")
        with open(CACHE_PATHS['model'], 'rb') as f:
            cache = pickle.load(f)
            tokenizer = cache['tokenizer']
            model = cache['model']
        logger.success("Transformer model loaded from cache successfully.")
    else:
        logger.info("Downloading and caching model...")
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        
        with open(CACHE_PATHS['model'], 'wb') as f:
            pickle.dump({'tokenizer': tokenizer, 'model': model}, f)
        logger.success("Model downloaded and cached successfully!")

    # Determine device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.success("CUDA is available. Using GPU.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.success("MPS is available. Using Apple Silicon GPU.")
    
    logger.success(f"Using device: {device}")
    model = model.cpu()
    try:
        if device != 'cpu':
            model = model.to(device)
            logger.success(f"Model moved to {device} successfully.")
    except Exception as e:
        logger.error(f"Failed to move model to {device}, falling back to CPU: {e}")
        device = 'cpu'
    
    llm_model = 'gemma3n:latest'
    return graph, node_embeddings, tokenizer, model, device, llm_model

def load_global_resources():
    """Loads all models, pre-parsed coordinates, and graph metrics into memory"""
    graph, node_embeddings, tokenizer, model, device, llm_model = get_language_models()
    
    logger.info("Loading dataframe from cache...")
    df = pd.read_pickle(CACHE_PATHS['dataframe'])
    
    # Pre-parse coordinates
    logger.info("Parsing coordinates for heatmap and spatial queries...")
    parsed_coords = []
    for val in df['lat_long']:
        if isinstance(val, tuple):
            parsed_coords.append((float(val[0]), float(val[1])))
        elif isinstance(val, str):
            try:
                t = ast.literal_eval(val)
                parsed_coords.append((float(t[0]), float(t[1])))
            except Exception:
                parsed_coords.append((1.3521, 103.8198))
        else:
            parsed_coords.append((1.3521, 103.8198))
            
    df['latitude'] = [c[0] for c in parsed_coords]
    df['longitude'] = [c[1] for c in parsed_coords]
    
    # Precompute lightweight heatmap array: [[lat, lon], ...]
    heatmap_coords = [[round(c[0], 5), round(c[1], 5)] for c in parsed_coords]
    
    embeddings_np = node_embeddings.numpy()
    faiss_index = build_faiss_index(embeddings_np)
    annoy_index = build_ann_index(embeddings_np)
    normalized_embeddings = cache_normalized_embeddings(embeddings_np)
    graph_metrics = cache_graph_metrics(graph)
    
    mlb = MultiLabelBinarizer()
    mlb.fit([['contract'], ['fulltime'], ['internship'], ['parttime'], ['temporary']])
    
    logger.success(f"All ML & Graph resources loaded successfully. Total jobs: {len(df)}")
    
    return {
        'graph': graph,
        'node_embeddings': node_embeddings,
        'tokenizer': tokenizer,
        'model': model,
        'device': device,
        'llm_model': llm_model,
        'df': df,
        'heatmap_coords': heatmap_coords,
        'faiss_index': faiss_index,
        'annoy_index': annoy_index,
        'normalized_embeddings': normalized_embeddings,
        'graph_metrics': graph_metrics,
        'mlb': mlb
    }
