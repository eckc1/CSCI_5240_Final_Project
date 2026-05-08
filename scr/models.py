from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class UserConstraints(BaseModel):
    budget: Optional[float] = None
    min_protein_g: Optional[float] = None
    max_calories: Optional[float] = None
    meals_breakfast: int = 0
    meals_lunch: int = 0
    meals_dinner: int = 0
    preferred_items: List[str] = Field(default_factory=list)
    avoid_items: List[str] = Field(default_factory=list)
    dietary_tags_required: List[str] = Field(default_factory=list)
    low_added_sugar: bool = False

class Plan(BaseModel):
    goal: str
    steps: List[str]
    assumptions: Dict[str, Any] = Field(default_factory=dict)
