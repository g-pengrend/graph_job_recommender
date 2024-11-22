from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class UserPreferences:
    """User preferences for job recommendations"""
    location: Optional[Tuple[float, float]] = None
    location_name: Optional[str] = None
    job_title: Optional[str] = None
    job_description: Optional[str] = None
    max_distance_km: float = 10.0
    priority: Optional[str] = None  # "Job Title" or "Job Description"
    within_range_count: int = 5     # Number of results to show within range
    outside_range_count: int = 5    # Number of results to show outside range