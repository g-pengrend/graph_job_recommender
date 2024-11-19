import streamlit as st
st.set_page_config(page_title="Job Recommendation System", layout="wide")

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import ollama
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from loguru import logger
import sys
from PyPDF2 import PdfReader
import tempfile
import re
import ast
import time

# Import your existing modules
from weight_calculator import calculate_weights
from resource_loader import load_global_resources
from preloaded_preferences import get_preloaded_preferences, JOB_PREFERENCES, LOCATIONS, IMPORTANCE_LEVELS
from user_preference import UserPreferences

# Initialize global variables
GLOBAL_RESOURCES = None
geolocator = Nominatim(user_agent="job_recommender_v1")

# Add cache decorator for resource loading
@st.cache_resource
def load_cached_resources():
    """Load and cache all required resources"""
    logger.info("Loading resources...")
    resources = load_global_resources()
    logger.success("Resources loaded successfully!")
    return resources

# Initialize resources at startup
GLOBAL_RESOURCES = load_cached_resources()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on transformer output"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_job_description_embedding(text: str, tokenizer, model, max_chunk_length: int = 512, overlap: int = 50, device: str = 'cpu') -> np.ndarray:
    with st.spinner("Generating job description embedding..."):
        tokens_per_chunk = max_chunk_length - 2  
        words = text.split()
        chunks = [' '.join(words[i:i+tokens_per_chunk]) for i in range(0, len(words), tokens_per_chunk - overlap)]
        
        embeddings = []
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            progress_bar.progress((i + 1) / len(chunks))
            encoded_input = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_chunk_length).to(device)
            
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
            embeddings.append(sentence_embedding.cpu().numpy())

        if embeddings:
            return np.mean(embeddings, axis=0).squeeze()
        else:
            return np.zeros((384,))

def get_job_title_embedding(text: str, tokenizer, model, device: str = 'cpu') -> np.ndarray:
    """Generate embedding for job title"""
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    
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
        st.error(f"Error processing document: {e}")
        return None

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

def get_graph_based_recommendations(
    preferences: UserPreferences,
    n_hops: int = 2,
    top_k: int = 5,
    n_candidates: int = 1000
) -> Dict[str, List[dict]]:
    """Generate recommendations using both FAISS and Annoy for hybrid search"""
    with st.spinner("Starting recommendation generation..."):
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
        
        # Create mapping from string ID to graph node name
        id_to_node = {row['id']: f"job_{idx}" for idx, row in df.iterrows()}
        node_to_id = {f"job_{idx}": row['id'] for idx, row in df.iterrows()}
        
        # Generate candidate nodes
        with st.spinner("Generating candidate nodes..."):
            candidate_indices = set()
            
            if preferences.job_title:
                title_embedding = get_job_title_embedding(preferences.job_title, tokenizer, model, device)
                title_embedding = title_embedding / np.linalg.norm(title_embedding)
                
                # Search with FAISS
                D_faiss, I_faiss = faiss_index.search(title_embedding.astype('float32').reshape(1,-1), top_k)
                faiss_candidates = set(I_faiss[0])
                
                # Search with Annoy
                annoy_candidates = set(annoy_index.get_nns_by_vector(title_embedding, top_k, search_k=-1))
                candidate_indices = faiss_candidates.union(annoy_candidates)
            else:
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
        
        progress_bar = st.progress(0)
        for batch_start in range(0, len(candidate_indices), batch_size):
            batch_indices = list(candidate_indices)[batch_start:batch_start + batch_size]
            current_batch = batch_start // batch_size + 1
            progress_bar.progress(current_batch / total_batches)
            
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
                st.error(f"Error processing location for job {score['index']}: {e}")
                outside_range.append(score)

        # Sort both lists by final_score
        within_range.sort(key=lambda x: x['final_score'], reverse=True)
        outside_range.sort(key=lambda x: x['final_score'], reverse=True)

        recommendations = {
            'within_range': within_range[:top_k],
            'outside_range': outside_range[:top_k]
        }

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
            st.error(f"Error processing index {idx}: {str(e)}")
            continue
    
    return batch_scores

def streamlit_get_preferences() -> UserPreferences:
    """Get user preferences through Streamlit UI"""
    st.subheader("Job Preferences")
    
    # Mode selection
    mode = st.radio(
        "Choose mode:",
        ["Use preloaded preferences (demo)", "Enter preferences manually"],
        horizontal=True
    )
    
    if mode == "Use preloaded preferences (demo)":
        # Job Profile Selection
        job_options = {k: v['title'] for k, v in JOB_PREFERENCES.items()}
        selected_job = st.selectbox(
            "Select a job profile",
            options=list(job_options.keys()),
            format_func=lambda x: f"{x}. {job_options[x]}"
        )
        
        job_pref = JOB_PREFERENCES[selected_job]

        # Show selected job details
        with st.expander("View Selected Job Details"):
            st.markdown("**Job Description:**")
            st.text(job_pref["description"])
        
        # Importance Levels
        col1, col2 = st.columns(2)
        with col1:
            title_importance = st.select_slider(
                "Title importance",
                options=IMPORTANCE_LEVELS,
                value=job_pref["title_importance"]
            )
        with col2:
            description_importance = st.select_slider(
                "Description importance",
                options=IMPORTANCE_LEVELS,
                value=job_pref["description_importance"]
            )
        
        # Location Selection
        location_option = st.radio(
            "Location Options",
            ["Choose from preset locations", "Enter custom location", "Skip location"],
            horizontal=True
        )
        
        location_name = None
        location_coords = None
        
        if location_option == "Choose from preset locations":
            location_choice = st.selectbox(
                "Select a location",
                options=list(LOCATIONS.keys()),
                format_func=lambda x: f"{x}. {LOCATIONS[x][0]}"
            )
            location_name, location_coords = LOCATIONS[location_choice]
            st.success(f"Selected location: {location_name}")
            
        elif location_option == "Enter custom location":
            location_input = st.text_input("Enter postal code or location name")
            if location_input:
                try:
                    with st.spinner("Finding location..."):
                        location_data = geolocator.geocode(location_input)
                        if location_data:
                            location_name = location_data.address
                            location_coords = (location_data.latitude, location_data.longitude)
                            st.success(f"Location found: {location_name}")
                        else:
                            st.warning("Location not found")
                except Exception as e:
                    st.error(f"Error processing location: {e}")
        
        # Max Distance (if location is selected)
        max_distance_km = 10.0
        if location_coords:
            max_distance_km = st.slider(
                "Maximum distance (km)",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=0.5
            )
        
        return UserPreferences(
            location=location_coords,
            location_name=location_name,
            job_title=job_pref["title"],
            job_description=job_pref["description"],
            max_distance_km=max_distance_km,
            title_importance=title_importance,
            description_importance=description_importance
        )
    
    else:
        # Manual preference entry (existing code)
        job_title = st.text_input("Job Title (optional)", "")
        if job_title:
            title_importance = st.select_slider(
                "How important is the job title match?",
                options=IMPORTANCE_LEVELS,
                value="Important"
            )
        else:
            title_importance = "Important"
        
        # Job Description
        st.write("Job Description (optional)")
        desc_tab1, desc_tab2 = st.tabs(["📝 Text Input", "📄 Upload PDF"])
        
        with desc_tab1:
            job_description = st.text_area("Enter job description", "", height=200)
        
        with desc_tab2:
            uploaded_file = st.file_uploader("Upload Resume/CV (PDF)", type=['pdf'])
            if uploaded_file:
                # Extract text from PDF
                raw_text = extract_text_from_pdf(uploaded_file)
                if raw_text:
                    st.success("PDF processed successfully!")
                    
                    use_ai = st.checkbox("Process with AI (recommended)", value=True)
                    
                    if use_ai:
                        # Process through LLM
                        with st.spinner("Processing document through AI..."):
                            processed_text = process_job_description_with_LLM(raw_text)
                        
                        if processed_text:
                            st.success("AI processing completed!")
                            job_description = processed_text
                            with st.expander("View processed text"):
                                st.text(job_description)
                        else:
                            st.error("Error processing document through AI")
                            job_description = raw_text
                            with st.expander("View raw text"):
                                st.text(job_description)
                    else:
                        job_description = raw_text
                        with st.expander("View raw text"):
                            st.text(job_description)
        
        if job_description:
            description_importance = st.select_slider(
                "How important is the job description match?",
                options=IMPORTANCE_LEVELS,
                value="Important"
            )
        else:
            description_importance = "Important"
        
        # Location
        location_name = st.text_input("Postal Code (Singapore preferably)", "")
        location = None
        
        if location_name:
            try:
                with st.spinner("Finding location..."):
                    location_data = geolocator.geocode(location_name)
                    if location_data:
                        location = (location_data.latitude, location_data.longitude)
                        st.success(f"Location found: {location_data.address}")
                        max_distance_km = st.number_input("Maximum distance in kilometers", 
                                                        min_value=1.0, 
                                                        max_value=100.0, 
                                                        value=10.0)
                    else:
                        st.warning("Location not found, proceeding without location preference")
                        location_name = None
                        max_distance_km = 10.0
            except Exception as e:
                st.error(f"Error processing location: {e}")
                location_name = None
                max_distance_km = 10.0
        else:
            max_distance_km = 10.0
        
        return UserPreferences(
            location=location,
            location_name=location_name,
            job_title=job_title,
            job_description=job_description,
            max_distance_km=max_distance_km,
            title_importance=title_importance,
            description_importance=description_importance
        )

def display_streamlit_recommendations(recommendations: Dict[str, List[dict]], preferences: UserPreferences) -> None:
    """Display job recommendations in Streamlit"""
    if not recommendations['within_range'] and not recommendations['outside_range']:
        st.warning("No recommendations found.")
        return
    
    # Display statistics
    st.subheader("Recommendation Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Within Range", len(recommendations['within_range']))
    with col2:
        st.metric("Outside Range", len(recommendations['outside_range']))
    
    # Display within range recommendations
    if recommendations['within_range']:
        st.subheader("Top Job Recommendations Within Range")
        display_streamlit_recommendation_group(recommendations['within_range'], preferences)
    else:
        st.info("No recommendations found within your preferred distance.")
    
    # Display outside range recommendations
    if recommendations['outside_range']:
        st.subheader("Top Job Recommendations Outside Range")
        display_streamlit_recommendation_group(recommendations['outside_range'], preferences)
    else:
        st.info("No recommendations found outside your preferred distance.")

def display_streamlit_recommendation_group(recommendations: List[dict], preferences: UserPreferences) -> None:
    """Helper function to display a group of recommendations in Streamlit"""
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"{i}. {rec['company']} - {rec['title']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Job Details**")
                st.write(f"Type: {', '.join(rec['job_type'])}")
                st.write(f"Remote: {'Yes' if rec['is_remote'] else 'No'}")
                
                if rec.get('address'):
                    st.write(f"Address: {rec['address']}")
                
                try:
                    lat, lon = ast.literal_eval(rec['location']) if isinstance(rec['location'], str) else rec['location']
                    if preferences.location:
                        distance = rec.get('distance') or geodesic(preferences.location, (lat, lon)).kilometers
                        st.write(f"Distance: {distance:.1f}km from your location")
                except Exception:
                    st.write("Location: Information unavailable")
            
            with col2:
                st.write("**Score Components**")
                for component, score in rec['score_components'].items():
                    weight = preferences.weights.get(component, 0)
                    weighted_score = score * weight
                    st.write(f"{component}: {score:.3f}")
                st.metric("Final Score", f"{rec['final_score']:.3f}")
            
            st.write("**Links**")
            if rec.get('job_url_direct'):
                st.write(f"[Direct Link]({rec['job_url_direct']})")
            if rec.get('job_url'):
                st.write(f"[Platform Link]({rec['job_url']})")

def main():
    st.title("Job Recommendation System")
    
    if not GLOBAL_RESOURCES:
        st.error("Failed to load resources. Please refresh the page.")
        return
        
    preferences = streamlit_get_preferences()
    
    # Generate recommendations button
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            recommendations = get_graph_based_recommendations(preferences=preferences)
            display_streamlit_recommendations(recommendations, preferences)

if __name__ == "__main__":
    main()