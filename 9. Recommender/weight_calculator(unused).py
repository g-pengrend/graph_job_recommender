from typing import Dict
from loguru import logger

def calculate_weights(
    has_title: bool,
    has_description: bool,
    has_location: bool,
    title_importance: str,
    description_importance: str,
    location_importance: str
) -> Dict[str, float]:
    """Calculate weights based on available inputs and user importance levels"""
    
    # First, get base weights without location component
    if has_title and has_description:
        # All semantic inputs available
        weights = {
            'title_similarity': 0.35,
            'description_similarity': 0.35,
            'pagerank': 0.15,
            'degree': 0.10,
            'core_numbers': 0.05
        }
    elif has_title or has_description:
        # Only one semantic input
        semantic_weight = 0.50
        weights = {
            'title_similarity': semantic_weight if has_title else 0.0,
            'description_similarity': semantic_weight if has_description else 0.0,
            'pagerank': 0.25,
            'degree': 0.20,
            'core_numbers': 0.05
        }
    else:
        # No semantic inputs
        weights = {
            'pagerank': 0.45,
            'degree': 0.45,
            'core_numbers': 0.10
        }

    # Apply importance multipliers only for title and description
    importance_multipliers = {
        "Very important": 2.0,
        "Important": 1.0,
        "Not important": 0.5,
        "Not important at all": 0.25
    }

    # Adjust weights based on importance levels
    if has_title:
        weights['title_similarity'] *= importance_multipliers[title_importance]
    if has_description:
        weights['description_similarity'] *= importance_multipliers[description_importance]

    # Calculate total weight of user preferences
    preference_weight = sum([
        weights.get('title_similarity', 0),
        weights.get('description_similarity', 0)
    ])

    # Redistribute remaining weight to graph metrics
    remaining_weight = 1.0 - preference_weight
    if remaining_weight > 0:
        graph_metrics = ['pagerank', 'degree', 'core_numbers']
        graph_weights_sum = sum(weights.get(metric, 0) for metric in graph_metrics)
        
        if graph_weights_sum > 0:
            scale_factor = (graph_weights_sum + remaining_weight) / graph_weights_sum
            for metric in graph_metrics:
                if metric in weights:
                    weights[metric] *= scale_factor

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    logger.info("\nFinal calculated weights:")
    for k, v in weights.items():
        logger.info(f"- {k}: {v:.3f}")
    
    return weights