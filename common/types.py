from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from .constants import FORMAT_JSON
from abc import ABC, abstractmethod

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

class AbstractWhisperWorker(ABC):
    @abstractmethod
    async def transcribe(self, input: TranscribeInput) -> Union[Dict[str, Any], str]:
        pass

    @abstractmethod
    async def translate(self, input: TranslateInput) -> Dict[str, Any]:
        pass