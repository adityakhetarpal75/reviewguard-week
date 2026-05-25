from pydantic import BaseModel, Field
from typing import Literal


class ReviewVerdict(BaseModel):
    label: Literal["fake", "genuine", "suspicious"] = Field(
        description="fake = confirmed fake, genuine = confirmed real, suspicious = uncertain"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="confidence in the label from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="brief explanation, max 2 sentences"
    )
    flagged_for_human: bool = Field(
        description="true if confidence below 0.7 or signals conflict"
    )