#!/usr/bin/env python3

"""
Graph Building Script

This script builds a graph representation of job listings data, creating edges between jobs based on
various similarity metrics including company, job type, location, and semantic similarity of titles
and descriptions.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from tqdm import tqdm
import multiprocessing as mp
from itertools import combinations
import time
import torch
import pickle
from annoy import AnnoyIndex
from scipy.spatial import cKDTree
import faiss
from sklearn.cluster import KMeans
import cupy as cp

def calculate_total_potential_edges(df_ready):
    """Calculate the total number of potential edges in a complete graph."""
    n = len(df_ready)
    return (n * (n-1)) // 2

def save_graph_checkpoint(graph, filename):
    """Save the graph to a pickle file checkpoint."""
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Saved checkpoint: {filename}")

def load_graph_checkpoint(filename):
    """Load a graph checkpoint from a pickle file."""
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    return graph
    
def initialize_graph(df_ready):
    """Initialize graph and add nodes with their attributes."""
    print("Initializing graph and adding nodes...")
    graph = nx.Graph()
    
    for idx, row in tqdm(df_ready.iterrows(), total=len(df_ready), desc="Adding nodes"):
        # Only add essential attributes needed for graph construction
        graph.add_node(
            f"job_{idx}",
            job_title_embedding=row['job_title_embedding'],
            job_description_embedding=row['job_description_embedding'],
            company=row['company'],
            job_type_encoding=row['job_type_encoded'],
            is_remote=row['is_remote'],
            lat_long=row['lat_long']
        )
    
    save_graph_checkpoint(graph, 'graph_with_nodes.pkl')
    return graph

def create_company_edges(df_ready, graph):
    """Create edges between jobs from the same company."""
    print("Creating edges between jobs from the same company...")
    company_groups = df_ready.groupby('company').groups
    
    total_company_edges = sum(len(indices) * (len(indices) - 1) // 2 
                            for indices in company_groups.values())
    
    pbar = tqdm(total=total_company_edges, desc="Company edges")
    edge_count = 0
    
    for company, indices in company_groups.items():
        if len(indices) > 1:
            for idx1, idx2 in combinations(indices, 2):
                graph.add_edge(f"job_{idx1}", f"job_{idx2}", 
                             type="same_company")
                edge_count += 1
                pbar.update(1)
    
    pbar.close()
    print(f"Added {edge_count} company edges")
    save_graph_checkpoint(graph, 'graph_with_company_edges.pkl')
    return graph

def create_job_type_edges(df_ready, graph, threshold=0.9, batch_size=1000):
    """Create edges between jobs with similar job types using MinHash LSH for approximate similarity.
    Only keeps the top 10 strongest connections per job."""
    print(f"Creating edges between jobs with similar job types (threshold={threshold})...")
    
    from datasketch import MinHash, MinHashLSH
    from heapq import heappush, heappushpop
    
    # Initialize LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    edge_count = 0
    
    # Create MinHash for each job
    for idx, row in tqdm(df_ready.iterrows(), desc="Creating MinHashes"):
        m = MinHash(num_perm=128)
        job_types = np.where(row['job_type_encoded'] == 1)[0]
        for t in job_types:
            m.update(str(t).encode('utf-8'))
        minhashes[f"job_{idx}"] = m
        lsh.insert(f"job_{idx}", m)
    
    # Store top 10 edges per job
    top_edges = {}
    
    # Query similar jobs in batches
    for idx in tqdm(range(len(df_ready)), desc="Finding similar jobs"):
        job_id = f"job_{idx}"
        query_hash = minhashes[job_id]
        
        # Initialize min heap for this job
        top_edges[job_id] = []
        
        # Find approximate neighbors
        similar_jobs = lsh.query(query_hash)
        
        # Process similar jobs
        for similar_job in similar_jobs:
            if similar_job > job_id:  # Avoid duplicate edges
                # Calculate actual Jaccard similarity
                sim = minhashes[job_id].jaccard(minhashes[similar_job])
                if sim > threshold:
                    # Use negative similarity for max heap behavior
                    if len(top_edges[job_id]) < 10:
                        heappush(top_edges[job_id], (sim, similar_job))
                    else:
                        heappushpop(top_edges[job_id], (sim, similar_job))
        
        # Add edges for top 10 similar jobs
        for sim, similar_job in top_edges[job_id]:
            graph.add_edge(job_id, similar_job,
                          type="job_type_similarity",
                          weight=float(sim))
            edge_count += 1
        
        # Save checkpoint every 1000 jobs
        if idx % 1000 == 0:
            save_graph_checkpoint(graph, 'graph_with_job_type_edges_temp.pkl')
            print(f"Processed {idx} jobs, current edge count: {edge_count}")
    
    print(f"Added {edge_count} job type similarity edges")
    save_graph_checkpoint(graph, 'graph_with_job_type_edges.pkl')
    return graph

def create_location_edges(df_ready, graph, max_distance=10, min_weight=0.4, chunk_size=500, sigma=1.5):
    """Create edges between jobs within geographical proximity using Gaussian decay."""
    print(f"Creating edges between jobs within {max_distance}km of each other...")
    print(f"Using minimum weight threshold of {min_weight}")
    edge_count = 0

    # Extract coordinates
    lat_longs = []
    valid_indices = []
    
    for idx in tqdm(range(len(df_ready)), desc="Extracting coordinates"):
        lat_long = df_ready.iloc[idx]['lat_long']
        if lat_long is not None:
            lat_longs.append(lat_long)
            valid_indices.append(idx)
    
    print(f"Processing {len(valid_indices)} locations")
    
    lat_longs = cp.array(lat_longs)
    n_points = len(lat_longs)
    
    # Build KD-tree for efficient spatial querying
    tree = cKDTree(cp.asnumpy(lat_longs))
    
    # Process in chunks
    for i in tqdm(list(range(0, n_points, chunk_size)), desc="Processing location proximity"):
        chunk_end = min(i + chunk_size, n_points)
        chunk_points = lat_longs[i:chunk_end]
        
        # Find nearby points within max_distance
        chunk_points_cpu = cp.asnumpy(chunk_points)
        nearby_points = tree.query_ball_point(chunk_points_cpu, max_distance/111.32)
        
        for j, neighbors in enumerate(nearby_points):
            if not neighbors:
                continue
                
            point1 = chunk_points[j]
            points2 = lat_longs[neighbors]
            
            # Calculate distances using haversine formula
            lat1, lon1 = point1[0], point1[1]
            lat2, lon2 = points2[:, 0], points2[:, 1]
            
            dlat = cp.radians(lat2 - lat1)
            dlon = cp.radians(lon2 - lon1)
            lat1, lat2 = cp.radians(lat1), cp.radians(lat2)
            
            a = cp.sin(dlat/2)**2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon/2)**2
            distances = 2 * 6371 * cp.arcsin(cp.sqrt(a))  # Earth radius in km
            
            # Calculate weights with Gaussian decay
            weights = cp.exp(-(distances**2)/(2*sigma**2))
            
            # Apply distance and weight thresholds
            weights = cp.where((distances <= max_distance) & (weights >= min_weight), weights, 0)
            
            weights_cpu = cp.asnumpy(weights)
            actual_idx1 = valid_indices[i+j]
            
            # Add edges only for significant weights and avoid duplicates
            for k, neighbor_idx in enumerate(neighbors):
                if (neighbor_idx > i+j and  # Only process upper triangle
                    weights_cpu[k] > min_weight):
                    actual_idx2 = valid_indices[neighbor_idx]
                    
                    # Add edge with weight
                    graph.add_edge(f"job_{actual_idx1}", f"job_{actual_idx2}",
                                 type="location_proximity", 
                                 weight=float(weights_cpu[k]))
                    edge_count += 1
            
            # Print progress every 5000 edges
            if edge_count % 5000 == 0:
                print(f"Created {edge_count} edges so far...")

    print(f"Added {edge_count} location proximity edges")
    save_graph_checkpoint(graph, 'graph_with_location_edges.pkl')
    return graph

def create_embedding_edges(df_ready, graph, embedding_type, threshold=0.7, k=10, n_clusters=500):
    """Create edges between jobs with similar embeddings using parallel processing and optimized FAISS search."""
    print(f"Creating edges between jobs with similar {embedding_type} (threshold={threshold})...")
    edge_count = 0
    use_gpu = torch.cuda.is_available()
    
    # Get embeddings and normalize once - embeddings already in array format
    embeddings = np.array(df_ready[embedding_type].tolist()).astype('float32')
    faiss.normalize_L2(embeddings)
    n_jobs = len(df_ready)
    
    # Create optimized IVF index
    d = embeddings.shape[1]  # Embedding dimension
    nlist = min(n_clusters, int(np.sqrt(n_jobs)))  # Number of Voronoi cells
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Train and add vectors
    index.train(embeddings)
    index.add(embeddings)
    
    # Process in batches
    edge_type = "title_similarity" if embedding_type == "job_title_embedding" else "description_similarity"
    batch_size = 1000
    
    for start_idx in tqdm(range(0, n_jobs, batch_size), desc="Finding similar jobs"):
        end_idx = min(start_idx + batch_size, n_jobs)
        batch_embeddings = embeddings[start_idx:end_idx]
        
        # Batch search
        similarities, indices = index.search(batch_embeddings, k)
        
        # Process results
        for i, (sims, nbrs) in enumerate(zip(similarities, indices)):
            global_idx = start_idx + i
            for sim, nbr in zip(sims, nbrs):
                if nbr <= global_idx or sim < threshold or nbr == -1:
                    continue
                graph.add_edge(f"job_{global_idx}", f"job_{nbr}",
                             type=edge_type, weight=float(sim))
                edge_count += 1
    
    print(f"Added {edge_count} {edge_type} edges")
    checkpoint_name = 'graph_with_title_edges.pkl' if edge_type == "title_similarity" else 'graph_with_description_edges.pkl'
    save_graph_checkpoint(graph, checkpoint_name)
    return graph

def build_complete_graph(df_ready, resume_from=None):
    """Build the complete graph with all edge types."""
    start_time = time.time()
    
    if resume_from is None:
        graph = initialize_graph(df_ready)
    else:
        checkpoint_files = {
            'nodes': 'graph_with_nodes.pkl',
            'company': 'graph_with_company_edges.pkl',
            'job_type': 'graph_with_job_type_edges.pkl',
            'location': 'graph_with_location_edges.pkl',
            'title': 'graph_with_title_edges.pkl',
            'description': 'graph_with_description_edges.pkl'
        }
        graph = load_graph_checkpoint(checkpoint_files[resume_from])
    
    steps = ['company', 'job_type', 'location', 'title', 'description']
    start_idx = steps.index(resume_from) + 1 if resume_from in steps else 0
    
    for step in steps[start_idx:]:
        step_start = time.time()
        if step == 'company':
            graph = create_company_edges(df_ready, graph)
        elif step == 'job_type':
            graph = create_job_type_edges(df_ready, graph)
        elif step == 'location':
            graph = create_location_edges(
                    df_ready, 
                    graph,
                    max_distance=10,      # Only connect very close jobs
                    min_weight=0.5,      # Only strong connections
                    sigma=1,            # Sharp distance decay
                    chunk_size=1000
                )
        elif step == 'title':
            graph = create_embedding_edges(df_ready, graph, 'job_title_embedding', threshold=0.7, k=10)
        elif step == 'description':
            graph = create_embedding_edges(df_ready, graph, 'job_description_embedding', threshold=0.7, k=10)
        step_end = time.time()
        print(f"{step} step took {(step_end - step_start)/60:.2f} minutes")
    
    total_time = time.time() - start_time
    print(f"\nTotal graph construction time: {total_time/60:.2f} minutes")
    print("\nGraph construction complete!")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    
    save_graph_checkpoint(graph, 'final_complete_graph.pkl')
    print("Final graph saved to 'final_complete_graph.pkl'")
    
    return graph 

def main():
    """Main function to load data and build the graph."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Build job listing graph with optional checkpoint resume.')
    parser.add_argument('--resume-from', type=str, choices=['nodes', 'company', 'job_type', 'location', 'title', 'description'],
                        help='Resume graph building from a specific checkpoint')
    
    args = parser.parse_args()
    
    print("Loading pickled dataframe...")
    df_ready = pd.read_pickle('final_graph_model_training.pkl')
    
    print("\nDataframe column dtypes:")
    print(df_ready.dtypes)

    print("\nSample checks:")
    print(f"lat_long type: {type(df_ready['lat_long'].iloc[0])}")
    print(f"job_title_embedding type: {type(df_ready['job_title_embedding'].iloc[0])}")
    print(f"job_description_embedding type: {type(df_ready['job_description_embedding'].iloc[0])}")

    graph = build_complete_graph(df_ready, resume_from=args.resume_from)
    return graph

if __name__ == "__main__":
    main()
