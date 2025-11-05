import torch
import pickle

from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from constants import CACHE_PATHS, CACHE_DIR
from utils import build_faiss_index, build_ann_index, cache_normalized_embeddings, cache_graph_metrics

def get_language_models():
    """Cache both the transformer model and LLM instance"""
    logger.info("Loading language models...")

    # Load graph if available
    if CACHE_PATHS['graph'].exists():
        with open(CACHE_PATHS['graph'], 'rb') as f:
            graph = pickle.load(f)
    else:
        logger.error("Error: Graph pickle file does not exist. Please check the path.")
    # Load node embeddings
    if CACHE_PATHS['node_embeddings'].exists():
        logger.info("Loading node embeddings from cache...")
        node_embeddings = torch.load(CACHE_PATHS['node_embeddings'])
        logger.success("Node embeddings loaded successfully!")
    else:
        logger.error(f"Node embeddings file not found: {CACHE_PATHS['node_embeddings']}")
        raise FileNotFoundError("Node embeddings file not found.")

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
        
        # Cache the model
        with open(CACHE_PATHS['model'], 'wb') as f:
            pickle.dump({
                'tokenizer': tokenizer,
                'model': model
            }, f)
        logger.success("Model downloaded and cached successfully!")

    # Determine device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.success("CUDA is available. Using GPU.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.zeros(1, 384).to('mps')
            test_result = test_tensor + 1
            del test_tensor, test_result
            device = 'mps'
            logger.success("MPS is available. Using Apple Silicon GPU.")
        except Exception as e:
            logger.error(f"MPS initialization failed, falling back to CPU: {e}")
            device = 'cpu'
    
    logger.success(f"Using device: {device}")
    
    # Move model to device
    model = model.cpu()
    try:
        if device != 'cpu':
            model = model.to(device)
            logger.success(f"Model moved to {device} successfully.")
    except Exception as e:
        logger.error(f"Failed to move model to {device}, falling back to CPU: {e}")
        device = 'cpu'
    
    # Load LLM model name
    # llm_model = 'capybarahermes-2.5-mistral-7b.Q5_K_M.gguf:latest'
    llm_model = 'gemma3n:latest'
    logger.success("LLM model name loaded successfully.")
    
    return graph, node_embeddings, tokenizer, model, device, llm_model

def load_global_resources():
    """Load and cache all resources needed for the application"""
    # Get language models
    graph, node_embeddings, tokenizer, model, device, llm_model = get_language_models()
    
    # Load job URL data
    df = pd.read_pickle(CACHE_PATHS['dataframe'])
    
    # Initialize search indices
    embeddings_np = node_embeddings.numpy()
    faiss_index = build_faiss_index(embeddings_np)
    annoy_index = build_ann_index(embeddings_np)
    normalized_embeddings = cache_normalized_embeddings(embeddings_np)
    graph_metrics = cache_graph_metrics(graph)
    
    # Initialize MLBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([['contract'], ['fulltime'], ['internship'], ['parttime'], ['temporary']])
    
    return {
        'graph': graph,
        'node_embeddings': node_embeddings,
        'tokenizer': tokenizer,
        'model': model,
        'device': device,
        'llm_model': llm_model,
        'df': df,
        'faiss_index': faiss_index,
        'annoy_index': annoy_index,
        'normalized_embeddings': normalized_embeddings,
        'graph_metrics': graph_metrics,
        'mlb': mlb
    }