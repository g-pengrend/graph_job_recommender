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
# from weight_calculator import calculate_weights
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
    score_components['core_numbers'] = graph_metrics['core_numbers'][node]
    
    return {k: max(0, min(1, v)) for k, v in score_components.items()}

def get_graph_based_recommendations(
    preferences: UserPreferences
) -> Dict[str, List[dict]]:
    """Get recommendations based on direct cosine similarity"""
    with st.spinner("Starting recommendation generation..."):
        # Unpack only needed resources
        df = GLOBAL_RESOURCES['df']
        tokenizer = GLOBAL_RESOURCES['tokenizer']
        model = GLOBAL_RESOURCES['model']
        device = GLOBAL_RESOURCES['device']
        graph = GLOBAL_RESOURCES['graph']
        graph_metrics = GLOBAL_RESOURCES['graph_metrics']
        
        # Initialize results containers
        within_range = []
        outside_range = []
        
        # Calculate similarities based on priority or fall back to graph metrics
        with st.spinner("Calculating similarities..."):
            if preferences.priority == "Job Title" and preferences.job_title:
                title_embedding = get_job_title_embedding(preferences.job_title, tokenizer, model, device)
                similarities = np.dot(np.array(df['job_title_embedding'].tolist()), title_embedding)
                
            elif preferences.priority == "Job Description" and preferences.job_description:
                desc_embedding = get_job_description_embedding(
                    preferences.job_description,
                    tokenizer,
                    model,
                    device=device
                )
                similarities = np.dot(np.array(df['job_description_embedding'].tolist()), desc_embedding)
            
            else:
                # Fallback to graph metrics if no title or description provided
                st.info("No title or description provided. Using graph metrics for recommendations.")
                similarities = np.array([
                    graph_metrics['pagerank'][f"job_{i}"] * 0.4 +
                    graph_metrics['degree'][f"job_{i}"] * 0.4 +
                    graph_metrics['core_numbers'][f"job_{i}"] * 0.2
                    for i in range(len(df))
                ])
            
            # Get indices sorted by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Process results
            for idx in sorted_indices:
                if (len(within_range) >= preferences.within_range_count and 
                    len(outside_range) >= preferences.outside_range_count):
                    break
                    
                try:
                    job_data = df.iloc[idx]
                    similarity_score = similarities[idx]
                    
                    # Process job type
                    job_type = job_data['job_type']
                    if isinstance(job_type, str):
                        try:
                            job_type = ast.literal_eval(job_type)
                        except:
                            job_type = [job_type]
                    elif not isinstance(job_type, (list, tuple)):
                        job_type = [str(job_type)]
                    
                    job_entry = {
                        'index': job_data['id'],
                        'company': job_data['company'],
                        'title': job_data['title'],
                        'job_type': job_type,
                        'location': job_data['lat_long'],
                        'address': job_data['address'],
                        'is_remote': job_data['is_remote'],
                        'job_url': job_data['job_url'],
                        'job_url_direct': job_data['job_url_direct'],
                        'final_score': float(similarity_score)
                    }
                    
                    # Check distance if location preference exists
                    if preferences.location and job_data['lat_long']:
                        try:
                            job_loc = (
                                ast.literal_eval(job_data['lat_long'])
                                if isinstance(job_data['lat_long'], str)
                                else job_data['lat_long']
                            )
                            distance = geodesic(preferences.location, job_loc).kilometers
                            job_entry['distance'] = distance
                            
                            if distance <= preferences.max_distance_km:
                                if len(within_range) < preferences.within_range_count:
                                    within_range.append(job_entry)
                            else:
                                if len(outside_range) < preferences.outside_range_count:
                                    outside_range.append(job_entry)
                        except:
                            if len(outside_range) < preferences.outside_range_count:
                                outside_range.append(job_entry)
                    else:
                        if len(within_range) < preferences.within_range_count:
                            within_range.append(job_entry)
                
                except Exception as e:
                    st.error(f"Error processing index {idx}: {str(e)}")
                    continue
            
            return {
                'within_range': within_range[:preferences.within_range_count],
                'outside_range': outside_range[:preferences.outside_range_count]
            }

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
            
            # Get similarity score based on priority
            if preferences.priority == "Job Title" and preferences.job_title:
                title_embedding = get_job_title_embedding(preferences.job_title, tokenizer, model, device)
                final_score = np.dot(title_embedding, normalized_embeddings[idx])
                # Convert from [-1,1] to [0,1] range
                final_score = (final_score + 1) / 2
            elif preferences.priority == "Job Description" and preferences.job_description:
                desc_embedding = get_job_description_embedding(
                    preferences.job_description,
                    tokenizer,
                    model,
                    device=device
                )
                final_score = np.dot(desc_embedding, normalized_embeddings[idx])
                # Convert from [-1,1] to [0,1] range
                final_score = (final_score + 1) / 2
            else:
                # Fallback to graph metrics if no priority
                final_score = (
                    graph_metrics['pagerank'][node] * 0.4 +
                    graph_metrics['degree'][node] * 0.4 +
                    graph_metrics['core_numbers'][node] * 0.2
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
                'final_score': final_score
            })
            
        except Exception as e:
            st.error(f"Error processing index {idx}: {str(e)}")
            continue
    
    return batch_scores

def streamlit_get_preferences() -> UserPreferences:
    """Get user preferences through Streamlit UI"""
    # Initialize all session state variables if not exists
    if 'processed_pdf_text' not in st.session_state:
        st.session_state.processed_pdf_text = None
    if 'raw_pdf_text' not in st.session_state:
        st.session_state.raw_pdf_text = None
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    if 'processed_text_input' not in st.session_state:
        st.session_state.processed_text_input = None
    if 'current_text_input' not in st.session_state:
        st.session_state.current_text_input = None
    if 'location_cache' not in st.session_state:
        st.session_state.location_cache = {}
    if 'current_job_title' not in st.session_state:
        st.session_state.current_job_title = None
    if 'processed_job_title' not in st.session_state:
        st.session_state.processed_job_title = None
    if 'active_document' not in st.session_state:
        st.session_state.active_document = None  # Can be 'text' or 'pdf'
    if 'active_version' not in st.session_state:
        st.session_state.active_version = None  # Can be 'raw' or 'ai'
    if 'final_description' not in st.session_state:
        st.session_state.final_description = None

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
        
        # Priority selection if both title and description are available
        priority = None
        if job_pref["title"] and job_pref["description"]:
            priority = st.radio(
                "Which should be prioritized for matching?",
                ["Job Title", "Job Description"],
                help="Select which criteria should be used for finding similar jobs"
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
            priority=priority,
            within_range_count=5,  # Default value
            outside_range_count=5  # Default value
        )
    
    else:  # Manual preference entry
        # Job Title processing with caching
        job_title = st.text_input("Job Title (optional)", "")
        if job_title and job_title != st.session_state.current_job_title:
            st.session_state.current_job_title = job_title
            with st.spinner("Processing job title..."):
                processed_title = process_job_description_with_LLM(job_title)
                st.session_state.processed_job_title = processed_title
                if processed_title:
                    with st.expander("View processed job title"):
                        st.text(processed_title)
        
        # Job Description
        st.write("Job Description (optional)")
        desc_tab1, desc_tab2 = st.tabs(["📝 Text Input", "📄 Upload PDF"])
        
        with desc_tab1:
            text_input = st.text_area("Enter job description", "", height=200)
            
            if text_input:
                # Submit button for text input
                if st.button("Submit Text"):
                    st.session_state.current_text_input = text_input
                    with st.spinner("Processing text with AI..."):
                        processed_text = process_job_description_with_LLM(text_input)
                        if processed_text:
                            st.session_state.processed_text_input = processed_text
                            st.success("Text processed successfully!")
                
                # Show both versions in expandable sections if text is submitted
                if st.session_state.current_text_input:
                    st.write("Available Versions:")
                    with st.expander("View Raw Text"):
                        st.text(text_input)
                    
                    if st.session_state.processed_text_input:
                        with st.expander("View AI Processed Text"):
                            st.text(st.session_state.processed_text_input)
                    
                    # Selection buttons below the expandable sections
                    st.write("---")  # Add a separator
                    st.write("Select version to use:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Use Raw Text"):
                            st.session_state.active_document = 'text'
                            st.session_state.active_version = 'raw'
                            st.session_state.final_description = text_input
                            st.success("Raw text selected for recommendations!")
                    
                    with col2:
                        if st.session_state.processed_text_input and st.button("Use AI Processed Text"):
                            st.session_state.active_document = 'text'
                            st.session_state.active_version = 'ai'
                            st.session_state.final_description = st.session_state.processed_text_input
                            st.success("AI processed text selected for recommendations!")
        
        with desc_tab2:
            uploaded_file = st.file_uploader("Upload Resume/CV (PDF)", type=['pdf'])
            
            if uploaded_file:
                # Process PDF button
                if st.button("Upload PDF to system"):
                    with st.spinner("Processing PDF..."):
                        raw_text = extract_text_from_pdf(uploaded_file)
                        if raw_text:
                            st.session_state.raw_pdf_text = raw_text
                            st.session_state.current_pdf_name = uploaded_file.name
                            processed_text = process_job_description_with_LLM(raw_text)
                            if processed_text:
                                st.session_state.processed_pdf_text = processed_text
                            st.success("PDF processed successfully!")
                
                # Show both versions in expandable sections if PDF is processed
                if st.session_state.raw_pdf_text:
                    st.write("Available Versions:")
                    with st.expander("View Raw PDF Text"):
                        st.text(st.session_state.raw_pdf_text)
                    
                    if st.session_state.processed_pdf_text:
                        with st.expander("View AI Processed PDF Text"):
                            st.text(st.session_state.processed_pdf_text)
                    
                    # Selection buttons below the expandable sections
                    st.write("---")  # Add a separator
                    st.write("Select version to use:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Use Raw PDF"):
                            st.session_state.active_document = 'pdf'
                            st.session_state.active_version = 'raw'
                            st.session_state.final_description = st.session_state.raw_pdf_text
                            st.success("Raw PDF selected for recommendations!")
                    
                    with col2:
                        if st.session_state.processed_pdf_text and st.button("Use AI Processed PDF"):
                            st.session_state.active_document = 'pdf'
                            st.session_state.active_version = 'ai'
                            st.session_state.final_description = st.session_state.processed_pdf_text
                            st.success("AI processed PDF selected for recommendations!")
            
            # Clear PDF state if file is removed
            if not uploaded_file and st.session_state.active_document == 'pdf':
                st.session_state.raw_pdf_text = None
                st.session_state.processed_pdf_text = None
                st.session_state.current_pdf_name = None
                st.session_state.active_document = None
                st.session_state.active_version = None
                st.session_state.final_description = None
        
        # Display current active selection status with more detail
        if st.session_state.active_document:
            st.info(f"Currently using {st.session_state.active_version} version of {st.session_state.active_document} input for recommendations")
            with st.expander("View current selection"):
                st.text(st.session_state.final_description)
        
        # Set job_description based on final selection
        job_description = st.session_state.final_description or ""

        # Priority selection if both title and description are available
        priority = None
        if job_title and job_description:
            priority = st.radio(
                "Which should be prioritized for matching?",
                ["Job Title", "Job Description"],
                help="Select which criteria should be used for finding similar jobs"
            )
        else:
            priority = "Job Title" if job_title else "Job Description" if job_description else None
        
        # Location processing with caching
        location_name = st.text_input("Postal Code (Singapore preferably)", "")
        location = None
        
        if location_name:
            # Check if location is already cached
            if location_name in st.session_state.location_cache:
                cached_data = st.session_state.location_cache[location_name]
                location = cached_data['coords']
                st.success(f"Location found (cached): {cached_data['address']}")
                max_distance_km = st.number_input(
                    "Maximum distance in kilometers", 
                    min_value=1.0, 
                    max_value=100.0, 
                    value=10.0
                )
            else:
                try:
                    with st.spinner("Finding location..."):
                        location_data = geolocator.geocode(location_name)
                        if location_data:
                            location = (location_data.latitude, location_data.longitude)
                            # Cache the location data
                            st.session_state.location_cache[location_name] = {
                                'coords': location,
                                'address': location_data.address
                            }
                            st.success(f"Location found: {location_data.address}")
                            max_distance_km = st.number_input(
                                "Maximum distance in kilometers", 
                                min_value=1.0, 
                                max_value=100.0, 
                                value=10.0
                            )
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
            priority=priority,
            within_range_count=5,  # Default value
            outside_range_count=5  # Default value
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
                st.write("**Similarity Score**")
                st.metric("Score", f"{rec['final_score']:.3f}")
            
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
    
    # Add results configuration just before generate recommendations button
    st.subheader("Results Configuration", divider="gray")
    col1, col2 = st.columns(2)
    with col1:
        within_range_count = st.number_input(
            "Number of results within range",
            min_value=1,
            max_value=50,
            value=5,
            help="How many job recommendations to show within your specified distance"
        )
    with col2:
        outside_range_count = st.number_input(
            "Number of results outside range",
            min_value=1,
            max_value=50,
            value=5,
            help="How many job recommendations to show outside your specified distance"
        )
    
    # Update preferences with the selected counts
    preferences.within_range_count = within_range_count
    preferences.outside_range_count = outside_range_count
    
    # Generate recommendations button
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            recommendations = get_graph_based_recommendations(preferences=preferences)
            display_streamlit_recommendations(recommendations, preferences)

if __name__ == "__main__":
    main()