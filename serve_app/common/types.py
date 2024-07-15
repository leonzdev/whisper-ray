from typing import List, Optional
from pydantic import BaseModel
from .constants import FORMAT_JSON

class TranscribeInput(BaseModel):
    file: bytes
    model: str = "base"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: str = FORMAT_JSON
    temperature: float = 0.0
    timestamp_granularities: List[str] = []

class TranslateInput(BaseModel):
    file: bytes
    model: str = "base"
    prompt: Optional[str] = None
    response_format: str = FORMAT_JSON
    temperature: float = 0.0
