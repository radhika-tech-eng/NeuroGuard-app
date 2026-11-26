from typing import Literal
from pydantic import BaseModel, Field

Annotation = Literal['Possible seizure', 'Noise detected', 'Needs review']
ReviewStatus = Literal['Reviewed', 'Needs follow-up']


class ReviewCreate(BaseModel):
    prediction_id: str = Field(min_length=1, max_length=200)
    reviewer_name: str = Field(min_length=1, max_length=100)
    annotation: Annotation
    notes: str = Field(default='', max_length=2000)
    status: ReviewStatus = 'Reviewed'


class ReviewUpdate(BaseModel):
    annotation: Annotation | None = None
    notes: str | None = Field(default=None, max_length=2000)
    status: ReviewStatus | None = None
