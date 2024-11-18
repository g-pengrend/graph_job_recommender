import streamlit as st
from dataclasses import dataclass
from geopy.geocoders import Nominatim
import ollama
import re
import PyPDF2
import networkx as nx
import torch
import time
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import ast
import traceback
import pickle
from pathlib import Path
import faiss
import annoy
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from colorama import Fore, Style, Back
import pandas as pd
from datetime import datetime

def print_message_with_timestamp(message, color):
    """Prints a message with the current date and time in the specified color."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}{current_time} - {message}{Style.RESET_ALL}")

def success(message):
    """Prints a success message with black text and green highlight."""
    print_message_with_timestamp(message, f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}")

def info(message):
    """Prints an info message with black text and cyan highlight."""
    print_message_with_timestamp(message, f"{Style.BRIGHT}{Fore.BLACK}{Back.CYAN}")

def warning(message):
    """Prints a warning message with black text and yellow highlight."""
    print_message_with_timestamp(message, f"{Style.BRIGHT}{Fore.BLACK}{Back.YELLOW}")

def error(message):
    """Prints an error message with bold white text and red highlight."""
    print_message_with_timestamp(message, f"{Style.BRIGHT}{Fore.WHITE}{Back.RED}")

    
# Move set_page_config to the very top, before any other Streamlit operations
st.set_page_config(page_title="Job Recommendation System", page_icon=":briefcase:", layout="wide")
success("Streamlit page configuration set successfully.")

# Initialize the geocoder
geolocator = Nominatim(user_agent="job_recommender_v1")
success("Geocoder initialized successfully.")

# Define cache paths
CACHE_DIR = Path('./demo/models')
CACHE_DIR.mkdir(exist_ok=True)
info(f"Cache directory created at: {CACHE_DIR}")

CACHE_PATHS = {
    'graph': CACHE_DIR / 'final_complete_graph.pkl',
    'annoy_index': CACHE_DIR / 'annoy_index.ann',
    'faiss_index': CACHE_DIR / 'faiss_index.pkl', 
    'graph_metrics': CACHE_DIR / 'graph_metrics.pkl',
    'embeddings_norm': CACHE_DIR / 'normalized_embeddings.pkl',
    'degree_scores': CACHE_DIR / 'degree_scores.pkl',
    # 'subgraphs': CACHE_DIR / 'subgraphs.pkl',
    'model': CACHE_DIR / 'model_cache.pkl',
    'node_embeddings': CACHE_DIR / 'node_embeddings.pt'
}
success("Cache paths defined successfully.")

@st.cache_resource
def get_language_models():
    """Cache both the transformer model and LLM instance"""
    info("Loading language models...")

    # Load graph if available
    if CACHE_PATHS['graph'].exists():
        with open(CACHE_PATHS['graph'], 'rb') as f:
            graph = pickle.load(f)
    else:
        error("Error: Graph pickle file does not exist. Please check the path.")
    
    # Load node embeddings
    if CACHE_PATHS['node_embeddings'].exists():
        info("Loading node embeddings from cache...")
        node_embeddings = torch.load(CACHE_PATHS['node_embeddings'], weights_only=True)
        success("Node embeddings loaded successfully!")
    else:
        error(f"Node embeddings file not found: {CACHE_PATHS['node_embeddings']}")
        raise FileNotFoundError("Node embeddings file not found. Please ensure node embeddings have been generated and saved from step 8.")

    # Load transformer model
    if CACHE_PATHS['model'].exists():
        info("Loading transformer model from cache...")
        with open(CACHE_PATHS['model'], 'rb') as f:
            cache = pickle.load(f)
            tokenizer = cache['tokenizer']
            model = cache['model']
        success("Transformer model loaded from cache successfully.")
    else:
        warning("No cached model found. A new model will be loaded.")

    # Determine device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        success("CUDA is available. Using GPU.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.zeros(1, 384).to('mps')
            test_result = test_tensor + 1
            del test_tensor, test_result
            device = 'mps'
            success("MPS is available. Using Apple Silicon GPU.")
        except Exception as e:
            error(f"MPS initialization failed, falling back to CPU: {e}")
            device = 'cpu'
    
    success(f"Using device: {device}")
    
    # Move model to device
    model = model.cpu()
    try:
        if device != 'cpu':
            model = model.to(device)
            success(f"Model moved to {device} successfully.")
    except Exception as e:
        error(f"Failed to move model to {device}, falling back to CPU: {e}")
        device = 'cpu'
    
    # Load LLM model name
    llm_model = 'capybarahermes-2.5-mistral-7b.Q5_K_M.gguf:latest'
    success("LLM model name loaded successfully.")
    
    return graph, node_embeddings, tokenizer, model, device, llm_model

# Replace the existing model loading calls with the cached version
graph, node_embeddings, tokenizer, model, device, llm_model = get_language_models()
success("Language models initialized successfully.")

def process_job_description_with_LLM(document_text):
    """Process a single document using the cached LLM setup."""
    prompt = f"""You are an expert in understanding job descriptions and extracting the details and even nuanced requirements for the job. Your goal is to read the input slowly and take time to consider what is written, extract the information and break it down into these 3 aspects:
    1. responsibilites 
    2. qualifications
    3. skills, technical and non-technical
and summarize it in point form line by line.
With each aspect answered, ensure that each of the aspects are properly differentiated and avoid overlaps as much as possible."""
    
    try:
        response = ollama.chat(
            model=llm_model,  # Use the cached model name
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': document_text}
            ]
        )
        response_text = response['message']['content']
        success("Job description processed successfully.")
        
        # Clean the response text
        cleaned_text = re.sub(r'[^A-Za-z0-9\s.,]', '', response_text)
        cleaned_text = re.sub(r'(?<!\d)(\d+)\.(?!\d)', '', cleaned_text).strip()
        
        return cleaned_text
    except Exception as e:
        error(f"Error processing document: {e}")
        return None

def extract_text_from_resume(pdf_file):
    """Extract text content from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    success("Resume loaded and text extracted successfully.")
    return text

# Function to build FAISS index
def build_faiss_index(embeddings):
    info("\nBuilding/Loading FAISS index...")
    if CACHE_PATHS['faiss_index'].exists():
        success("Loading cached FAISS index...")
        with open(CACHE_PATHS['faiss_index'], 'rb') as f:
            index = pickle.load(f)
        success("FAISS index loaded successfully!")
        return index
            
    success("Building new FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    success("FAISS index initialized.")
    
    info("Normalizing embeddings...")
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    success("Embeddings normalized successfully.")
    
    info("Adding vectors to index...")
    index.add(normalized_embeddings.astype('float32'))
    success("Vectors added to FAISS index successfully.")
    
    info("Saving index to disk...")
    with open(CACHE_PATHS['faiss_index'], 'wb') as f:
        pickle.dump(index, f)
    success("FAISS index built and saved successfully!")
    return index

# Function to build Annoy index
def build_ann_index(embeddings_np, n_trees=100):
    info("\nBuilding/Loading Annoy index...")
    if CACHE_PATHS['annoy_index'].exists():
        success("Loading cached Annoy index...")
        index = annoy.AnnoyIndex(embeddings_np.shape[1], 'angular')
        index.load(str(CACHE_PATHS['annoy_index']))
        success("Annoy index loaded successfully!")
        return index
    
    success("Building new Annoy index...")
    index = annoy.AnnoyIndex(embeddings_np.shape[1], 'angular')
    
    for i in range(len(embeddings_np)):
        if i % 1000 == 0:
            info(f"Adding item {i}/{len(embeddings_np)} to Annoy index...")
        index.add_item(i, embeddings_np[i])
    
    success("Building index with trees...")
    index.build(n_trees)
    success("Annoy index built successfully.")
    
    info("Saving index to disk...")
    index.save(str(CACHE_PATHS['annoy_index']))
    success("Annoy index saved successfully!")
    return index

# Function to cache normalized embeddings
def cache_normalized_embeddings(embeddings_np):
    info("\nPreparing normalized embeddings...")
    if CACHE_PATHS['embeddings_norm'].exists():
        success("Loading cached normalized embeddings...")
        with open(CACHE_PATHS['embeddings_norm'], 'rb') as f:
            normalized = pickle.load(f)
        success("Normalized embeddings loaded successfully!")
        return normalized
    
    success("Computing normalized embeddings...")
    normalized = embeddings_np / np.linalg.norm(embeddings_np, axis=1)[:, np.newaxis]
    success("Normalized embeddings computed successfully.")
    
    info("Saving normalized embeddings to disk...")
    with open(CACHE_PATHS['embeddings_norm'], 'wb') as f:
        pickle.dump(normalized, f)
    success("Normalized embeddings cached successfully!")
    return normalized

def compute_pagerank_torch(graph, damping=0.85, max_iter=100, tol=1e-6):
    info("\nComputing PageRank scores...")
    # Create node ID to index mapping
    node_map = {node: idx for idx, node in enumerate(graph.nodes())}
    reverse_map = {idx: node for node, idx in node_map.items()}
    n = len(node_map)
    
    # Convert edges using the mapping
    edges = [(node_map[e[0]], node_map[e[1]]) for e in graph.edges()]
    row = np.array([e[0] for e in edges])
    col = np.array([e[1] for e in edges])
    data = np.ones(len(edges))
    adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))

    success("Adjacency matrix created successfully.")
    info("Normalizing adjacency matrix...")
    out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1
    D_inv = csr_matrix((1.0 / out_degree, (np.arange(n), np.arange(n))), shape=(n, n))
    stochastic_matrix = D_inv @ adj_matrix

    success("Adjacency matrix normalized successfully.")
    info("Converting to PyTorch sparse format...")
    coo_matrix = stochastic_matrix.tocoo()
    indices = torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)
    values = torch.tensor(coo_matrix.data, dtype=torch.float32)
    sparse_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([n, n])).cuda()

    success("Sparse matrix created successfully.")
    info("Initializing PageRank computation...")
    pagerank_vector = torch.ones(n, device="cuda") / n

    for i in range(max_iter):
        new_pagerank_vector = (1 - damping) / n + damping * torch.sparse.mm(sparse_matrix.t(), pagerank_vector.unsqueeze(1)).squeeze()
        if torch.norm(new_pagerank_vector - pagerank_vector, p=1) < tol:
            success(f"PageRank converged after {i + 1} iterations")
            break
        pagerank_vector = new_pagerank_vector

    success("PageRank computation completed!")
    # Convert back to original node IDs
    pagerank_dict = {reverse_map[i]: score for i, score in enumerate(pagerank_vector.cpu().numpy())}
    return pagerank_dict

# Function to cache graph metrics
def cache_graph_metrics(graph):
    info("\nComputing/Loading graph metrics...")
    if CACHE_PATHS['graph_metrics'].exists():
        success("Loading cached graph metrics...")
        with open(CACHE_PATHS['graph_metrics'], 'rb') as f:
            metrics = pickle.load(f)
        success("Graph metrics loaded successfully!")
        return metrics

    info("Computing graph metrics...")
    info("Computing degree centrality...")
    degree = dict(graph.degree())
    max_degree = max(degree.values())
    
    success("Degree centrality computed successfully.")
    info("Computing PageRank...")
    pagerank = compute_pagerank_torch(graph)
    
    success("PageRank computed successfully.")
    info("Computing core numbers...")
    core_numbers = nx.core_number(graph)
    
    success("Core numbers computed successfully.")
    metrics = {
        'degree': degree,
        'max_degree': max_degree,
        'pagerank': pagerank,
        'core_numbers': core_numbers
    }
    
    info("Saving graph metrics to disk...")
    with open(CACHE_PATHS['graph_metrics'], 'wb') as f:
        pickle.dump(metrics, f)
    success("Graph metrics cached successfully!")
    return metrics

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_job_title_embedding(text, tokenizer, model, device='cpu'):
    info(f"\nGenerating embedding for job title: {text}")
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    success("Job title embedding generated successfully!")
    
    return sentence_embedding.cpu().numpy().squeeze()

# Initialize tokenizer, model, and device
graph, node_embeddings, tokenizer, model, device, llm_model = get_language_models()
success("Tokenizer, model, and device initialized successfully.")

def get_job_description_embedding(text, tokenizer, model, max_chunk_length=512, overlap=50, device='cpu'):
    info("\nGenerating job description embedding...")
    tokens_per_chunk = max_chunk_length - 2  
    words = text.split()
    chunks = [' '.join(words[i:i+tokens_per_chunk]) for i in range(0, len(words), tokens_per_chunk - overlap)]
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        info(f"Processing chunk {i+1}/{len(chunks)}...")
        encoded_input = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_chunk_length).to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        embeddings.append(sentence_embedding.cpu().numpy())

    if embeddings:
        success("Averaging chunk embeddings...")
        return np.mean(embeddings, axis=0).squeeze()
    else:
        warning("No embeddings generated, returning zero vector")
        return np.zeros((384,))

# MultiLabelBinarizer setup for job type decoding
mlb = MultiLabelBinarizer()
mlb.fit([['contract'], ['fulltime'], ['internship'], ['parttime'], ['temporary']])
success("MultiLabelBinarizer for job types set up successfully.")

@dataclass
class UserPreferences:
    def __init__(self):
        self.job_title: Optional[str] = None
        self.job_description: Optional[str] = None
        self.location: Optional[Tuple[float, float]] = None
        self.location_name: Optional[str] = None
        self.max_distance_km: float = 10.0
        
        # Initialize base weights
        self.weights = {
            # Graph Structure Metrics (0.3 total)
            'pagerank': 0.15,      # Global importance
            'degree': 0.10,        # Local connectivity
            'core_number': 0.05,   # Network centrality
            
            # Semantic Relevance (0.5 total)
            'title_similarity': 0.25,      # Direct role match
            'description_similarity': 0.25, # Skills/experience match
            
            # Geographic Preference (0.2 total)
            'location_proximity': 0.20
        }
    
    def adjust_weights(self):
        """Dynamically adjust weights based on provided preferences"""
        adjusted_weights = {}
        
        if not self.job_title and not self.job_description:
            # No semantic inputs: Increase graph metrics importance
            adjusted_weights = {
                'pagerank': 0.35,
                'degree': 0.35,
                'core_number': 0.10,
                'location_proximity': 0.20 if self.location else 0
            }
        
        elif self.job_title and not self.job_description:
            # Only title provided
            adjusted_weights = {
                'title_similarity': 0.40,
                'pagerank': 0.20,
                'degree': 0.10,
                'core_number': 0.10,
                'location_proximity': 0.20 if self.location else 0
            }
        
        elif self.job_description and not self.job_title:
            # Only description provided
            adjusted_weights = {
                'description_similarity': 0.40,
                'pagerank': 0.20,
                'degree': 0.10,
                'core_number': 0.10,
                'location_proximity': 0.20 if self.location else 0
            }
        
        if adjusted_weights:
            self.weights = adjusted_weights
            
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

@st.cache_resource
def load_global_resources():
    """Load and cache all resources needed for the application"""
    # Get language models
    graph, node_embeddings, tokenizer, model, device, llm_model = get_language_models()
    
    # Load job URL data
    df = pd.read_pickle('./demo/models/final_complete_graph_dataframe.pkl')
    
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

# Load resources once at startup
resources = load_global_resources()

def get_graph_based_recommendations(
    preferences: UserPreferences,
    n_hops: int = 2,
    top_k: int = 5,
    n_candidates: int = 1000
) -> List[dict]:
    """Generate recommendations using both FAISS and Annoy for hybrid search"""
    print("\nStarting recommendation generation...")
    start_time = time.time()
    
    # Adjust weights based on provided preferences
    preferences.adjust_weights()
    
    print("\nGenerating candidate nodes...")
    candidate_nodes = set()
    
    # Get candidates based on job title
    if preferences.job_title:
        print(f"Finding similar jobs based on title: {preferences.job_title}")
        title_embedding = get_job_title_embedding(
            preferences.job_title, 
            tokenizer, 
            model, 
            device
        )
        title_embedding = title_embedding / np.linalg.norm(title_embedding)
        
        _, I_faiss_title = resources['faiss_index'].search(
            title_embedding.astype('float32').reshape(1,-1), 
            top_k * 8
        )
        faiss_candidates_title = {f"job_{idx}" for idx in I_faiss_title[0]}
        
        annoy_candidates_title = {f"job_{idx}" for idx in resources['annoy_index'].get_nns_by_vector(
            title_embedding, 
            top_k * 4,
            search_k=-1
        )}
        
        candidate_nodes.update(faiss_candidates_title.union(annoy_candidates_title))
    
    # Get additional candidates based on job description
    if preferences.job_description:
        print("Finding similar jobs based on description...")
        desc_embedding = get_job_description_embedding(
            preferences.job_description,
            tokenizer,
            model,
            device=device
        )
        desc_embedding = desc_embedding / np.linalg.norm(desc_embedding)
        
        _, I_faiss_desc = resources['faiss_index'].search(
            desc_embedding.astype('float32').reshape(1,-1), 
            top_k * 8
        )
        faiss_candidates_desc = {f"job_{idx}" for idx in I_faiss_desc[0]}
        
        annoy_candidates_desc = {f"job_{idx}" for idx in resources['annoy_index'].get_nns_by_vector(
            desc_embedding, 
            top_k * 4,
            search_k=-1
        )}
        
        candidate_nodes.update(faiss_candidates_desc.union(annoy_candidates_desc))
    
    # Fallback to PageRank if no title or description provided
    if not candidate_nodes:
        print("No job title or description specified, using PageRank for candidate selection...")
        weights = np.array([resources['graph_metrics']['pagerank'][node] for node in resources['graph'].nodes()])
        weights = weights / weights.sum()
        candidate_nodes = set(np.random.choice(
            list(resources['graph'].nodes()),
            size=min(top_k * 2, len(resources['graph'])),
            p=weights,
            replace=False
        ))
    
    print("\nComputing job scores...")
    job_scores = []
    
    for node in candidate_nodes:
        try:
            attrs = resources['graph'].nodes[node]
            score_components = {}
            
            # Graph metrics (30% total weight by default)
            score_components['pagerank'] = resources['graph_metrics']['pagerank'][node]
            score_components['degree'] = resources['graph_metrics']['degree'][node] / resources['graph_metrics']['max_degree']
            score_components['core_number'] = resources['graph_metrics']['core_numbers'][node] / max(resources['graph_metrics']['core_numbers'].values())
            
            # Semantic similarity (50% total weight by default)
            if preferences.job_title:
                similarity = np.dot(
                    title_embedding,
                    resources['normalized_embeddings'][int(node.split('_')[1])]
                )
                similarity = (similarity + 1) / 2  # Scale to [0,1]
                score_components['title_similarity'] = similarity
            
            if preferences.job_description:
                desc_similarity = np.dot(
                    desc_embedding,
                    resources['normalized_embeddings'][int(node.split('_')[1])]
                )
                desc_similarity = (desc_similarity + 1) / 2  # Scale to [0,1]
                score_components['description_similarity'] = desc_similarity
            
            # Location proximity (20% weight by default)
            if preferences.location:
                job_location = (
                    ast.literal_eval(attrs['lat_long'])
                    if isinstance(attrs['lat_long'], str)
                    else tuple(attrs['lat_long'])
                )
                distance = geodesic(preferences.location, job_location).kilometers
                location_score = np.exp(-distance / preferences.max_distance_km)
                score_components['location_proximity'] = location_score
            
            # Calculate final weighted score
            final_score = sum(
                score * preferences.weights.get(component, 0)
                for component, score in score_components.items()
            )
            
            # Create job recommendation object
            job_data = {
                'job_id': node,
                'company': attrs['company'],
                'job_type': list(mlb.inverse_transform(np.array(ast.literal_eval(attrs['job_type_encoded'])))[0]),
                'is_remote': attrs['is_remote'],
                'location': job_location if preferences.location else None,
                'distance': distance if preferences.location else None,
                'user_location': preferences.location_name if preferences.location else None,
                'score_components': score_components,
                'weights_used': preferences.weights,
                'final_score': final_score
            }
            
            job_scores.append(job_data)
            
        except Exception as e:
            print(f"Error processing node {node}: {str(e)}")
            continue
    
    # Sort by final score and return top_k recommendations
    recommendations = sorted(job_scores, key=lambda x: x['final_score'], reverse=True)[:top_k]
    
    print(f"\nRecommendation generation completed in {time.time() - start_time:.2f} seconds")
    return recommendations

def print_detailed_recommendations(recommendations: list[dict]) -> None:
    """Print detailed recommendations to terminal for debugging"""
    info("\n" + "="*50)
    info("DETAILED RECOMMENDATIONS SUMMARY")
    info("="*50)
    
    for idx, rec in enumerate(recommendations, 1):
        info(f"\nRECOMMENDATION #{idx}")
        info("-" * 30)
        info(f"Job ID: {rec['node_id']}")
        info(f"Company: {rec.get('company', 'N/A')}")
        info(f"Title: {rec.get('title', 'N/A')}")
        info(f"Job Type: {', '.join(rec['job_type'])}")
        info("\nLocation Details:")
        info(f"- Coordinates: {rec['location']}")
        info(f"- Distance: {rec['distance']:.2f} km")
        info(f"- Remote: {'Yes' if rec['is_remote'] else 'No'}")
        
        info("\nScoring Details:")
        info(f"Final Score: {rec['final_score']:.4f}")
        info("Score Components:")
        for component, score in rec['score_components'].items():
            info(f"- {component}: {score:.4f}")
        
        info("\nLinks:")
        info(f"Job URL: {rec.get('job_url', 'N/A')}")
        info(f"Direct URL: {rec.get('job_url_direct', 'N/A')}")
        info("-" * 50)

def main():
    st.title("Job Recommendation System")
    st.write("Welcome to the Job Recommendation System!")

    # Initialize preferences at the start of main
    preferences = UserPreferences()
    
    with st.form("input_form"):
        job_title_input = st.text_input("Enter your preferred job title", key="job_title")
        job_title_importance = st.radio("How important is it to you?", 
                              ("Not important at all", "Not important", "Important", "Very important"), 
                              key="job_title_importance")
        
        location_input = st.text_input("Enter your preferred location or postal code", key="location")
        location_importance = st.radio("How important is it to you?", 
                              ("Not important at all", "Not important", "Important", "Very important"), 
                              key="location_importance")

        resume_pdf_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
        if resume_pdf_file is not None:
            success("Resume uploaded successfully!")

        resume_importance = st.radio("How important is it to you?", 
                              ("Not important at all", "Not important", "Important", "Very important"), 
                              key="resume_importance")

        submitted = st.form_submit_button("Generate Recommendations")

    if submitted:
        st.header("Top Recommendations:")

        with st.spinner("Generating recommendations... Please wait..."):
            preferences.job_title = job_title_input
            preferences.max_distance_km = 20.0

            try:
                # Try to geocode the location input directly
                location = geolocator.geocode(location_input)
            except Exception as e:
                error(f"Error geocoding location: {e}")
                location = None

            if location is not None:
                preferences.location = (location.latitude, location.longitude)
                preferences.location_name = location.address
                success("Location geocoded successfully.")
            else:
                preferences.location = (1.3521, 103.8198)
                preferences.location_name = "Singapore"
                warning("Location not found, defaulting to Singapore.")

            if resume_pdf_file is not None:
                preferences.job_description = process_job_description_with_LLM(extract_text_from_resume(resume_pdf_file))
                # preferences.job_description = extract_text_from_resume(resume_pdf_file)
            else:
                preferences.job_description = ""

            if job_title_importance == "Very important":
                preferences.weights['title_similarity'] = 2.0  # Double the weight
                # Reduce other weights proportionally
                preferences.weights['degree'] *= 0.5
                preferences.weights['pagerank'] *= 0.5
                preferences.weights['core_number'] *= 0.5
            elif job_title_importance == "Important":
                preferences.weights['title_similarity'] = 1.0
            elif job_title_importance == "Not important":
                preferences.weights['title_similarity'] = 0.33
            else:
                preferences.weights['title_similarity'] = 0.01
            
            if location_importance == "Very important":
                preferences.weights['location_proximity'] = 1.0
            elif location_importance == "Important":
                preferences.weights['location_proximity'] = 0.66
            elif location_importance == "Not important":
                preferences.weights['location_proximity'] = 0.33
            else:
                preferences.weights['location_proximity'] = 0.01

            if resume_importance == "Very important":
                preferences.weights['description_similarity'] = 2.0  # Double the weight
                # Reduce other weights proportionally
                preferences.weights['degree'] *= 0.5
                preferences.weights['pagerank'] *= 0.5
                preferences.weights['core_number'] *= 0.5
            elif resume_importance == "Important":
                preferences.weights['description_similarity'] = 1.0
            elif resume_importance == "Not important":
                preferences.weights['description_similarity'] = 0.33
            else:
                preferences.weights['description_similarity'] = 0.01

            # Calculate average weight of user preferences
            preference_weights = [
                preferences.weights['title_similarity'],
                preferences.weights['location_proximity'],
                preferences.weights['description_similarity']
            ]
            avg_preference_weight = sum(preference_weights) / len(preference_weights)
            
            # Scale graph metrics relative to average preference weight
            # If preferences are less important, graph metrics get higher weights
            graph_metric_weight = max(0.1, 1.0 - avg_preference_weight)
            preferences.weights['degree'] = graph_metric_weight
            preferences.weights['pagerank'] = graph_metric_weight
            preferences.weights['core_number'] = graph_metric_weight

            recommendations = get_graph_based_recommendations(preferences)

            # Before creating the DataFrame
            info("\nPreparing final data:")
            data = []
            if recommendations:
                for rec in recommendations:
                    info(f"\nProcessing recommendation for job {rec['node_id']}:")
                    info(f"Original location: {rec['location']}")
                    info(f"Original distance: {rec['distance']}")
                    
                    # Get job details from the DataFrame
                    job_data = resources['df'][resources['df']['node_id'] == rec['job_id']].iloc[0]
                    
                    # Create the data entry with explicit values
                    entry = {
                        "Company": str(rec['company']),
                        "Job Title": str(job_data['title']),
                        "Job Type": ', '.join(rec['job_type']),
                        "Distance (km)": float(rec['distance']),  # Explicit conversion
                        "Remote": "Yes" if rec['is_remote'] else "No",
                        "Job URL": str(job_data['job_url']),
                        "Direct URL": str(job_data['job_url_direct']),
                        "Final Score": float(rec['final_score']),
                    }
                    
                    # Add score components
                    for component, score in rec['score_components'].items():
                        entry[f"{component} Score"] = float(score)
                    
                    info(f"Created entry with distance: {entry['Distance (km)']}km")
                    data.append(entry)

            # Create DataFrame
            if data:
                recommendations_df = pd.DataFrame(data)
                info("\nFinal DataFrame head:")
                info(recommendations_df[['Company', 'Job Title', 'Distance (km)', 'Final Score']].head().to_string())
                st.write(recommendations_df)
            else:
                st.write("No recommendations available.")
if __name__ == "__main__":
    main()