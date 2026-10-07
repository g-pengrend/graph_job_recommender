import os
import re
import ast
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn.functional as F
import ollama
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from loguru import logger
from PyPDF2 import PdfReader

from resource_loader import load_global_resources
from preloaded_preferences import JOB_PREFERENCES, LOCATIONS

CURRENT_DIR = Path(__file__).resolve().parent
STATIC_DIR = CURRENT_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Singapore Spatial Job Radar API", version="2.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Geocoder
geolocator = Nominatim(user_agent="spatial_job_recommender_api_v1")

from contextlib import asynccontextmanager

# Global resources state
RESOURCES: Optional[Dict[str, Any]] = None

def get_resources():
    global RESOURCES
    if RESOURCES is None:
        logger.info("Initializing ML models, GraphSAGE embeddings, and spatial indices...")
        RESOURCES = load_global_resources()
        logger.success("Server ready! All resources active.")
    return RESOURCES

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_resources()
    yield

app = FastAPI(title="Singapore Spatial Job Radar API", version="2.0.0", lifespan=lifespan)

# ----------------- Helper Functions ----------------- #

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_job_description_embedding(text: str, tokenizer, model, max_chunk_length: int = 512, overlap: int = 50, device: str = 'cpu') -> np.ndarray:
    if isinstance(max_chunk_length, str):
        device = max_chunk_length
        max_chunk_length = 512
    max_chunk_length = int(max_chunk_length)
    tokens_per_chunk = max_chunk_length - 2
    words = text.split()
    chunks = [' '.join(words[i:i+tokens_per_chunk]) for i in range(0, max(1, len(words)), max(1, tokens_per_chunk - overlap))]
    
    embeddings = []
    for chunk in chunks:
        encoded = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_chunk_length).to(device)
        with torch.no_grad():
            out = model(**encoded)
        sent_emb = mean_pooling(out, encoded['attention_mask'])
        sent_emb = F.normalize(sent_emb, p=2, dim=1)
        embeddings.append(sent_emb.cpu().numpy())

    if embeddings:
        return np.mean(embeddings, axis=0).squeeze()
    return np.zeros((384,))

def get_job_title_embedding(text: str, tokenizer, model, device: str = 'cpu') -> np.ndarray:
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**encoded)
    sent_emb = mean_pooling(out, encoded['attention_mask'])
    sent_emb = F.normalize(sent_emb, p=2, dim=1)
    return sent_emb.cpu().numpy().squeeze()

def calculate_blend_weights(count: int) -> List[float]:
    base_count = 5
    weights = [1.0] * base_count
    remaining = count - base_count
    if remaining > 0:
        groups = (remaining // base_count) + (1 if remaining % base_count != 0 else 0)
        for i in range(groups):
            sim_weight = max(0.1, 0.9 - (i * 0.1))
            weights.extend([sim_weight] * base_count)
    return weights[:count]

# ----------------- Request Models ----------------- #

class RecommendRequest(BaseModel):
    job_title: Optional[str] = ""
    job_description: Optional[str] = ""
    user_lat: float = 1.3521
    user_lon: float = 103.8198
    max_distance_km: float = 10.0
    priority: str = "Job Title"  # "Job Title" or "Job Description"
    top_k_within: int = 10
    top_k_outside: int = 10

# ----------------- API Endpoints ----------------- #

@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Modern Spatial Recommender API running. static/index.html not found yet."}

@app.get("/api/health")
def health_check():
    res = get_resources()
    return {
        "status": "healthy",
        "total_jobs": len(res['df']),
        "device": res['device']
    }

@app.get("/api/presets")
def get_presets():
    """Returns preset personas and locations for demo clicks"""
    personas = []
    for k, v in JOB_PREFERENCES.items():
        personas.append({
            "id": k,
            "title": v.get("title", ""),
            "description": v.get("description", "").strip()
        })
        
    locations = []
    for k, v in LOCATIONS.items():
        locations.append({
            "id": k,
            "name": v[0],
            "lat": v[1][0],
            "lon": v[1][1]
        })
        
    return {
        "personas": personas,
        "locations": locations
    }

@app.get("/api/heatmap")
def get_heatmap():
    """Returns the pre-parsed coordinates of all 25,610 jobs for 60fps client rendering"""
    res = get_resources()
    return {"points": res['heatmap_coords']}

@app.get("/api/geocode")
def geocode_address(q: str = Query(..., description="Singapore Postal Code or Address")):
    """Geocode Singapore postal codes or locations"""
    try:
        search_query = f"{q}, Singapore" if "singapore" not in q.lower() else q
        location = geolocator.geocode(search_query)
        if location:
            return {
                "success": True,
                "lat": location.latitude,
                "lon": location.longitude,
                "address": location.address
            }
        return {"success": False, "message": "Location not found"}
    except Exception as e:
        logger.error(f"Geocode error: {e}")
        return {"success": False, "message": str(e)}

@app.get("/api/config")
def get_config():
    """Returns map provider keys and active LLM configuration"""
    return {
        "mapbox_token": os.getenv("MAPBOX_ACCESS_TOKEN", "").strip(),
        "stadia_key": os.getenv("STADIA_API_KEY", "").strip(),
        "carto_key": os.getenv("CARTO_API_KEY", "").strip(),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    }

@app.post("/api/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """Extracts text from uploaded PDF resume and parses with Ollama LLM"""
    try:
        pdf_reader = PdfReader(file.file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        cleaned_text = text.strip()
        
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
        llm_summary = None
        llm_success = False
        llm_error = None

        try:
            prompt = """You are an expert technical recruiter and HR career advisor.
Analyze the user's resume and extract their profile into 3 structured aspects:
1. Core Responsibilities & Experience: (Concise bullet points of previous duties)
2. Qualifications & Education: (Degrees, certifications, seniority)
3. Technical & Soft Skills: (Programming languages, frameworks, domain skills)
Format clearly in clean bullet points line by line without filler text."""
            
            resp = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': cleaned_text[:3500]}
                ]
            )
            llm_summary = resp['message']['content'].strip()
            llm_success = True
        except Exception as e:
            logger.warning(f"Ollama parsing fallback ({model_name}): {e}")
            llm_error = str(e)
            
        return {
            "success": True,
            "filename": file.filename,
            "raw_text": cleaned_text,
            "ai_extracted": llm_summary or cleaned_text,
            "word_count": len(cleaned_text.split()),
            "llm_used": model_name,
            "llm_success": llm_success,
            "llm_error": llm_error
        }
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

@app.post("/api/recommend")
def recommend_jobs(req: RecommendRequest):
    """Core Recommendation Engine with Progressive Blending & Commute Partitioning"""
    t0 = time.time()
    res = get_resources()

    df = res['df']
    tokenizer = res['tokenizer']
    model = res['model']
    device = res['device']
    graph_metrics = res['graph_metrics']

    user_coords = (req.user_lat, req.user_lon)

    # 1. Semantic Similarity
    if req.priority == "Job Title" and req.job_title:
        title_emb = get_job_title_embedding(req.job_title, tokenizer, model, device)
        similarities = np.dot(np.array(df['job_title_embedding'].tolist()), title_emb)
    elif req.priority == "Job Description" and req.job_description:
        desc_emb = get_job_description_embedding(req.job_description, tokenizer, model, device=device)
        similarities = np.dot(np.array(df['job_description_embedding'].tolist()), desc_emb)
    else:
        # Default Graph Centrality
        similarities = np.array([
            graph_metrics['pagerank'][f"job_{i}"] * 0.4 +
            graph_metrics['degree'][f"job_{i}"] * 0.4 +
            graph_metrics['core_numbers'][f"job_{i}"] * 0.2
            for i in range(len(df))
        ])

    # 2. Graph Centrality Metric
    graph_scores = np.array([
        graph_metrics['pagerank'][f"job_{i}"] * 0.4 +
        graph_metrics['degree'][f"job_{i}"] * 0.4 +
        graph_metrics['core_numbers'][f"job_{i}"] * 0.2
        for i in range(len(df))
    ])

    # Normalization to [0, 1]
    similarities = (similarities - similarities.min()) / max(1e-9, (similarities.max() - similarities.min()))
    graph_scores = (graph_scores - graph_scores.min()) / max(1e-9, (graph_scores.max() - graph_scores.min()))

    sorted_indices = np.argsort(similarities)[::-1]
    within_range = []
    outside_range = []

    within_weights = calculate_blend_weights(req.top_k_within)
    outside_weights = calculate_blend_weights(req.top_k_outside)

    for idx in sorted_indices:
        if len(within_range) >= req.top_k_within and len(outside_range) >= req.top_k_outside:
            break

        job_row = df.iloc[idx]
        sim_val = float(similarities[idx])
        graph_val = float(graph_scores[idx])

        job_lat = float(job_row['latitude'])
        job_lon = float(job_row['longitude'])
        job_coords = (job_lat, job_lon)

        try:
            dist_km = round(geodesic(user_coords, job_coords).kilometers, 2)
            is_within = dist_km <= req.max_distance_km
        except Exception:
            dist_km = 999.0
            is_within = False

        job_type = job_row['job_type']
        if isinstance(job_type, str):
            try:
                job_type = ast.literal_eval(job_type)
            except Exception:
                job_type = [job_type]
        if not isinstance(job_type, list):
            job_type = [str(job_type)]

        if is_within and len(within_range) < req.top_k_within:
            w = within_weights[len(within_range)]
            final_score = round(float(w * sim_val + (1 - w) * graph_val), 3)
            within_range.append({
                "id": str(job_row['id']),
                "rank": len(within_range) + 1,
                "title": str(job_row['title']),
                "company": str(job_row['company']),
                "address": str(job_row['address']),
                "job_type": job_type,
                "is_remote": bool(job_row['is_remote']),
                "lat": job_lat,
                "lon": job_lon,
                "distance_km": dist_km,
                "final_score": final_score,
                "similarity_score": round(sim_val, 3),
                "graph_score": round(graph_val, 3),
                "sim_weight": round(w, 2),
                "job_url": job_row['job_url'] if pd.notna(job_row['job_url']) else None,
                "job_url_direct": job_row['job_url_direct'] if pd.notna(job_row['job_url_direct']) else None,
                "status": "within"
            })
        elif not is_within and len(outside_range) < req.top_k_outside:
            w = outside_weights[len(outside_range)]
            final_score = round(float(w * sim_val + (1 - w) * graph_val), 3)
            outside_range.append({
                "id": str(job_row['id']),
                "rank": len(outside_range) + 1,
                "title": str(job_row['title']),
                "company": str(job_row['company']),
                "address": str(job_row['address']),
                "job_type": job_type,
                "is_remote": bool(job_row['is_remote']),
                "lat": job_lat,
                "lon": job_lon,
                "distance_km": dist_km,
                "final_score": final_score,
                "similarity_score": round(sim_val, 3),
                "graph_score": round(graph_val, 3),
                "sim_weight": round(w, 2),
                "job_url": job_row['job_url'] if pd.notna(job_row['job_url']) else None,
                "job_url_direct": job_row['job_url_direct'] if pd.notna(job_row['job_url_direct']) else None,
                "status": "outside"
            })

    elapsed_ms = round((time.time() - t0) * 1000, 1)
    return {
        "within_range": within_range,
        "outside_range": outside_range,
        "execution_time_ms": elapsed_ms,
        "total_indexed": len(df)
    }

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
