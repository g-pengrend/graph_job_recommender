from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import time
import ast
import re
import torch
import torch.nn.functional as F
import ollama
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from loguru import logger
import sys

from weight_calculator import calculate_weights
from resource_loader import load_global_resources
from preloaded_preferences import get_preloaded_preferences
from user_preference import UserPreferences

# Initialize geocoder and resources
geolocator = Nominatim(user_agent="job_recommender_v1")
GLOBAL_RESOURCES = None

def init_resources():
    """Initialize all required resources"""
    global GLOBAL_RESOURCES
    if GLOBAL_RESOURCES is None:
        logger.info("Loading resources...")
        GLOBAL_RESOURCES = load_global_resources()
        logger.success("Resources loaded successfully!")

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on transformer output"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_job_description_embedding(text: str, tokenizer, model, max_chunk_length: int = 512, overlap: int = 50, device: str = 'cpu') -> np.ndarray:
    """Generate embedding for job description text"""
    logger.info("\nGenerating job description embedding...")
    tokens_per_chunk = max_chunk_length - 2  
    words = text.split()
    chunks = [' '.join(words[i:i+tokens_per_chunk]) for i in range(0, len(words), tokens_per_chunk - overlap)]
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
        encoded_input = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_chunk_length).to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        embeddings.append(sentence_embedding.cpu().numpy())

    if embeddings:
        logger.info("Averaging chunk embeddings...")
        return np.mean(embeddings, axis=0).squeeze()
    else:
        logger.warning("No embeddings generated, returning zero vector")
        return np.zeros((384,))

def get_job_title_embedding(text: str, tokenizer, model, device: str = 'cpu') -> np.ndarray:
    """Generate embedding for job title"""
    logger.info(f"\nGenerating embedding for job title: {text}")
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    logger.success("Job title embedding generated successfully!")
    
    return sentence_embedding.cpu().numpy().squeeze()

def process_job_description_with_LLM(document_text: str) -> Optional[str]:
    """Process job description using LLM for better matching"""
    model_name = 'capybarahermes-2.5-mistral-7b.Q5_K_M.gguf:latest'
    prompt = """You are an expert in understanding job descriptions and extracting the details and even nuanced requirements for the job. Your goal is to read the input slowly and take time to consider what is written, extract the information and break it down into these 3 aspects:
    1. responsibilites 
    2. qualifications
    3. skills, technical and non-technical
and summarize it in point form line by line.
With each aspect answered, ensure that each of the aspects are properly differentiated and avoid overlaps as much as possible."""
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': document_text}
            ]
        )
        response_text = response['message']['content']
        
        # Clean the response text
        cleaned_text = re.sub(r'[^A-Za-z0-9\s.,]', '', response_text)
        cleaned_text = re.sub(r'(?<!\d)(\d+)\.(?!\d)', '', cleaned_text).strip()
        
        return cleaned_text
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None

def get_graph_based_recommendations(
    preferences: UserPreferences,
    n_hops: int = 2,
    top_k: int = 5,
    n_candidates: int = 1000
) -> List[dict]:
    """Generate recommendations using both FAISS and Annoy for hybrid search"""
    logger.info("\nStarting recommendation generation...")
    start_time = time.time()
    
    # Unpack resources
    graph = GLOBAL_RESOURCES['graph']
    node_embeddings = GLOBAL_RESOURCES['node_embeddings']
    tokenizer = GLOBAL_RESOURCES['tokenizer']
    model = GLOBAL_RESOURCES['model']
    device = GLOBAL_RESOURCES['device']
    faiss_index = GLOBAL_RESOURCES['faiss_index']
    annoy_index = GLOBAL_RESOURCES['annoy_index']
    normalized_embeddings = GLOBAL_RESOURCES['normalized_embeddings']
    graph_metrics = GLOBAL_RESOURCES['graph_metrics']
    mlb = GLOBAL_RESOURCES['mlb']
    df = GLOBAL_RESOURCES['df']
    
    embeddings_np = node_embeddings.numpy()
    logger.info(f"Embeddings shape: {embeddings_np.shape}")
    
    # Create mapping from string ID to graph node name
    id_to_node = {row['id']: f"job_{idx}" for idx, row in df.iterrows()}
    node_to_id = {f"job_{idx}": row['id'] for idx, row in df.iterrows()}
    
    # Generate candidate nodes
    logger.info("\nGenerating candidate nodes...")
    candidate_indices = set()
    
    if preferences.job_title:
        logger.info(f"Finding similar jobs to title: {preferences.job_title}")
        title_embedding = get_job_title_embedding(preferences.job_title, tokenizer, model, device)
        title_embedding = title_embedding / np.linalg.norm(title_embedding)
        
        # Search with FAISS
        D_faiss, I_faiss = faiss_index.search(title_embedding.astype('float32').reshape(1,-1), top_k)
        faiss_candidates = set(I_faiss[0])
        
        # Search with Annoy
        annoy_candidates = set(annoy_index.get_nns_by_vector(title_embedding, top_k, search_k=-1))
        candidate_indices = faiss_candidates.union(annoy_candidates)
        logger.info(f"Found {len(candidate_indices)} candidate indices")
    else:
        logger.info("No job title specified, using PageRank for candidate selection...")
        weights = np.array([graph_metrics['pagerank'][f"job_{i}"] for i in range(len(df))])
        weights = weights / weights.sum()
        candidate_indices = set(np.random.choice(
            len(df),
            size=min(n_candidates, len(df)),
            p=weights,
            replace=False
        ))

    # Process candidates in batches
    batch_size = 100
    job_scores = []
    total_batches = len(candidate_indices) // batch_size + (1 if len(candidate_indices) % batch_size else 0)
    
    for batch_start in range(0, len(candidate_indices), batch_size):
        batch_indices = list(candidate_indices)[batch_start:batch_start + batch_size]
        current_batch = batch_start // batch_size + 1
        logger.info(f"\nProcessing batch {current_batch}/{total_batches}...")
        
        # Process each candidate in the batch
        batch_scores = process_candidate_batch(
            batch_indices, 
            graph, 
            preferences, 
            normalized_embeddings,
            graph_metrics,
            tokenizer,
            model,
            device,
            node_to_id,
            mlb,
            df
        )
        job_scores.extend(batch_scores)
    
    # Separate recommendations by distance
    within_range = []
    outside_range = []
    
    for score in job_scores:
        try:
            if preferences.location:
                job_loc = (
                    ast.literal_eval(score['location'])
                    if isinstance(score['location'], str)
                    else score['location']
                )
                distance = geodesic(preferences.location, job_loc).kilometers
                score['distance'] = distance
                
                if distance <= preferences.max_distance_km:
                    within_range.append(score)
                else:
                    outside_range.append(score)
            else:
                within_range.append(score)
                
        except Exception as e:
            logger.error(f"Error processing location for job {score['index']}: {e}")
            outside_range.append(score)

    # Sort both lists by final_score
    within_range.sort(key=lambda x: x['final_score'], reverse=True)
    outside_range.sort(key=lambda x: x['final_score'], reverse=True)

    # Take top_k from each group
    recommendations = {
        'within_range': within_range[:top_k],
        'outside_range': outside_range[:top_k]
    }
    
    # Log statistics
    logger.info("\nRecommendation Statistics:")
    logger.info(f"- Total jobs processed: {len(job_scores)}")
    logger.info(f"- Jobs within range: {len(within_range)}")
    logger.info(f"- Jobs outside range: {len(outside_range)}")
    
    # Calculate averages for each group
    if within_range:
        avg_within = np.mean([r['final_score'] for r in within_range[:top_k]])
        logger.info(f"- Average score (top {top_k} within range): {avg_within:.3f}")
    else:
        logger.info(f"- Average score (within range): N/A")
        
    if outside_range:
        avg_outside = np.mean([r['final_score'] for r in outside_range[:top_k]])
        logger.info(f"- Average score (top {top_k} outside range): {avg_outside:.3f}")
    else:
        logger.info(f"- Average score (outside range): N/A")

    logger.success(f"\nRecommendation generation completed in {time.time() - start_time:.2f} seconds")
    return recommendations

def process_candidate_batch(
    batch_indices: List[int],
    graph,
    preferences: UserPreferences,
    normalized_embeddings: np.ndarray,
    graph_metrics: Dict,
    tokenizer,
    model,
    device: str,
    node_to_id: Dict,
    mlb,
    df
) -> List[Dict]:
    """Process a batch of candidate indices and return their scores"""
    batch_scores = []
    
    for idx in batch_indices:
        try:
            node = f"job_{idx}"
            attrs = graph.nodes[node]
            
            score_components = calculate_score_components(
                idx,
                node,
                attrs,
                preferences,
                normalized_embeddings,
                graph_metrics,
                tokenizer,
                model,
                device
            )
            
            # Calculate final score
            final_score = sum(
                score * preferences.weights.get(component, 0)
                for component, score in score_components.items()
            )
            
            # Decode job type
            job_type_encoded = attrs['job_type_encoding']
            if isinstance(job_type_encoded, str):
                job_type_encoded = ast.literal_eval(job_type_encoded)
            job_type_encoded = np.array(job_type_encoded)
            if job_type_encoded.ndim == 1:
                job_type_encoded = job_type_encoded.reshape(1, -1)
            job_type_decoded = mlb.inverse_transform(job_type_encoded)[0]
            
            # Create recommendation entry
            job_data = df.loc[df['id'] == node_to_id[node]].iloc[0]
            batch_scores.append({
                'index': node_to_id[node],
                'company': attrs['company'],
                'title': job_data['title'],
                'job_type': job_type_decoded,
                'location': attrs['lat_long'],
                'address': job_data['address'],
                'is_remote': attrs['is_remote'],
                'job_url': job_data['job_url'],
                'job_url_direct': job_data['job_url_direct'],
                'score_components': score_components,
                'final_score': final_score
            })
            
        except Exception as e:
            logger.error(f"Error processing index {idx}: {str(e)}")
            continue
    
    return batch_scores

def calculate_score_components(
    idx: int,
    node: str,
    attrs: Dict,
    preferences: UserPreferences,
    normalized_embeddings: np.ndarray,
    graph_metrics: Dict,
    tokenizer,
    model,
    device: str
) -> Dict[str, float]:
    """Calculate individual score components for a job candidate"""
    score_components = {}
    
    # Calculate semantic similarities
    if preferences.job_title:
        title_embedding = get_job_title_embedding(preferences.job_title, tokenizer, model, device)
        raw_similarity = np.dot(title_embedding, normalized_embeddings[idx])
        score_components['title_similarity'] = (raw_similarity + 1) / 2

    if preferences.job_description:
        desc_embedding = get_job_description_embedding(
            preferences.job_description,
            tokenizer,
            model,
            device=device
        )
        raw_similarity = np.dot(desc_embedding, normalized_embeddings[idx])
        score_components['description_similarity'] = (raw_similarity + 1) / 2
    
    # Add graph metrics
    score_components['degree'] = graph_metrics['degree'][node]
    score_components['pagerank'] = graph_metrics['pagerank'][node]
    score_components['core_number'] = graph_metrics['core_numbers'][node]
    
    return {k: max(0, min(1, v)) for k, v in score_components.items()}

def get_personalized_recommendations(preload_preferences: bool = False) -> None:
    """Interactive function to get user preferences and return recommendations"""
    logger.info("\nJob Recommendation System")
    logger.info("------------------------")
    
    preferences = get_user_preferences(preload_preferences)
    
    logger.info("\nGenerating recommendations...")
    recommendations = get_graph_based_recommendations(preferences=preferences)
    
    display_recommendations(recommendations, preferences)

def get_user_preferences(preload_preferences: bool) -> UserPreferences:
    """Get user preferences either through preload or user input"""
    if preload_preferences:
        return get_preloaded_preferences()
    else:
        return get_interactive_preferences()

def display_recommendations(recommendations: Dict[str, List[dict]], preferences: UserPreferences) -> None:
    """Display job recommendations in a formatted way"""
    if not recommendations['within_range'] and not recommendations['outside_range']:
        logger.warning("\nNo recommendations found.")
        return
    
    # Display statistics
    logger.info("\nRecommendation Statistics:")
    logger.info(f"- Within range recommendations: {len(recommendations['within_range'])}")
    logger.info(f"- Outside range recommendations: {len(recommendations['outside_range'])}")
    
    # Display within range recommendations
    if recommendations['within_range']:
        logger.info("\nTop Job Recommendations Within Range:")
        logger.info("===================================")
        display_recommendation_group(recommendations['within_range'], preferences)
    else:
        logger.info("\nNo recommendations found within your preferred distance.")
    
    # Display outside range recommendations
    if recommendations['outside_range']:
        logger.info("\nTop Job Recommendations Outside Range:")
        logger.info("===================================")
        display_recommendation_group(recommendations['outside_range'], preferences)
    else:
        logger.info("\nNo recommendations found outside your preferred distance.")

def display_recommendation_group(recommendations: List[dict], preferences: UserPreferences) -> None:
    """Helper function to display a group of recommendations"""
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n{i}. {rec['company']}")
        logger.info("   " + "="*(len(rec['company'])))
        
        # Display job info
        logger.info(f"   Title: {rec['title']}")
        job_types = ', '.join(rec['job_type'])
        logger.info(f"   Type: {job_types}")
        logger.info(f"   Remote: {'Yes' if rec['is_remote'] else 'No'}")
        
        try:
            lat, lon = ast.literal_eval(rec['location']) if isinstance(rec['location'], str) else rec['location']
            if preferences.location:
                distance = rec.get('distance') or geodesic(preferences.location, (lat, lon)).kilometers
                logger.info(f"   Location: ({lat:.2f}, {lon:.2f}) - {distance:.1f}km from your location")
            else:
                logger.info(f"   Location: ({lat:.2f}, {lon:.2f})")
            
            # Display address
            if rec.get('address'):
                logger.info(f"   Address: {rec['address']}")
                
        except Exception as e:
            logger.error(f"Error displaying location: {e}")
            logger.info("   Location: Information unavailable")
        
        # Display score components
        logger.info("\n   Score Components:")
        for component, score in rec['score_components'].items():
            weight = preferences.weights.get(component, 0)
            weighted_score = score * weight
            logger.info(f"   - {component}: {score:.3f} (weight: {weight:.3f}, contribution: {weighted_score:.3f})")
        
        logger.info(f"\n   Final Score: {rec['final_score']:.3f}")
        
        # Display URLs
        logger.info("\n   Links:")
        if rec.get('job_url_direct'):
            logger.info(f"   - Direct: {rec['job_url_direct']}")
        if rec.get('job_url'):
            logger.info(f"   - Platform: {rec['job_url']}")
        
        logger.info("\n" + "-"*50)

def get_interactive_preferences() -> UserPreferences:
    """Get user preferences through interactive input"""
    logger.info("\nPlease provide your job preferences:")
    
    # Job Title
    job_title = input("\nJob Title (press Enter to skip): ").strip() or None
    if job_title:
        title_importance = get_importance_level("How important is the job title match?")
    else:
        title_importance = "Important"
    
    # Job Description
    logger.info("\nJob Description (press Enter for a New Line, another Enter to end, or empty line to skip):")
    lines = []
    while True:
        line = input()
        if not line and not lines:  # Empty description
            job_description = None
            break
        if not line:  # End of description
            job_description = "\n".join(lines)
            break
        lines.append(line)
    
    if job_description:
        description_importance = get_importance_level("How important is the job description match?")
    else:
        description_importance = "Important"
    
    # Location
    location_name = input("\nPostal Code (Singapore preferbly) (press Enter to skip): ").strip() or None
    location = None
    if location_name:
        try:
            location_data = geolocator.geocode(location_name)
            if location_data:
                location = (location_data.latitude, location_data.longitude)
                logger.success(f"Location found: {location_data.address}")
                location_importance = get_importance_level("How important is the location match?")
                max_distance_km = float(input("\nMaximum distance in kilometers (default: 10): ") or 10)
            else:
                logger.warning("Location not found, proceeding without location preference")
                location_name = None
                location_importance = "Important"
                max_distance_km = 10.0
        except Exception as e:
            logger.error(f"Error processing location: {e}")
            location_name = None
            location_importance = "Important"
            max_distance_km = 10.0
    else:
        location_importance = "Important"
        max_distance_km = 10.0
    
    return UserPreferences(
        location=location,
        location_name=location_name,
        job_title=job_title,
        job_description=job_description,
        max_distance_km=max_distance_km,
        title_importance=title_importance,
        description_importance=description_importance,
        location_importance=location_importance
    )

def get_importance_level(prompt: str) -> str:
    """Helper function to get importance level from user"""
    valid_levels = {
        "1": "Very important",
        "2": "Important",
        "3": "Not important",
        "4": "Not important at all"
    }
    
    while True:
        logger.info(f"\n{prompt}")
        logger.info("1: Very important")
        logger.info("2: Important")
        logger.info("3: Not important")
        logger.info("4: Not important at all")
        
        choice = input("Choose importance level (1-4, default: 2): ").strip() or "2"
        
        if choice in valid_levels:
            return valid_levels[choice]
        else:
            logger.warning("Invalid choice. Please select 1-4.")

def run_recommendation_loop():
    """Run the recommendation system in a loop"""
    while True:
        try:
            # Ask if user wants to use preloaded preferences
            mode = input("\nChoose mode:\n1. Use preloaded preferences (demo)\n2. Enter preferences manually\nChoice (1/2): ").strip()
            if mode not in ['1', '2']:
                logger.warning("Please enter 1 or 2")
                continue

            # Get recommendations based on mode
            get_personalized_recommendations(preload_preferences=(mode == '1'))
            
            # Ask if user wants to try again
            again = input("\nWould you like to try another search? (yes/no): ").lower().strip()
            if not again.startswith('y'):
                break
                
        except KeyboardInterrupt:
            logger.info("\nRecommendation system terminated by user.")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            break

def main():
    """Main function to run the job recommendation system"""
    logger.info("\nWelcome to the Job Recommendation System!")
    logger.info("=====================================")
    
    # Initialize resources first
    init_resources()
    
    # Run the recommendation loop
    run_recommendation_loop()
    
    logger.info("\nThank you for using the Job Recommendation System!")

if __name__ == "__main__":
    try:
        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            sink=sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            colorize=True,
            level="INFO"
        )
        
        # Create cache directory if it doesn't exist
        CACHE_DIR = Path('recommendation_cache')
        CACHE_DIR.mkdir(exist_ok=True)
        
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
