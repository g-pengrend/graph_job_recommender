from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from weight_calculator import calculate_weights

@dataclass
class UserPreferences:
    """User preferences for job recommendations"""
    location: Optional[Tuple[float, float]] = None
    location_name: Optional[str] = None
    job_title: Optional[str] = None
    job_description: Optional[str] = None
    max_distance_km: float = 10.0
    weights: dict = None
    title_importance: str = "Important"
    description_importance: str = "Important"
    location_importance: str = "Important"
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = calculate_weights(
                has_title=bool(self.job_title),
                has_description=bool(self.job_description),
                has_location=bool(self.location),
                title_importance=self.title_importance,
                description_importance=self.description_importance,
                location_importance=self.location_importance
            )