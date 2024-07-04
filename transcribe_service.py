import ray
from ray import serve
from typing import Dict, Any, Optional, Union, List, Iterable
from pydantic import BaseModel
from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionOptions, Segment
import asyncio
import io

# Define input schema compatible with OpenAI's API
class TranscriptionInput(BaseModel):
    file: bytes
    model: str = "base"
    prompt: Optional[str] = None
    temperature: float = 0.0
    language: Optional[str] = None
    format: str = "json"

# Initialize Faster-Whisper model

@serve.deployment
class TranscriptionService:
    def __init__(self):
        self.model = WhisperModel("base", device="cpu")

    
    async def transcribe(self, input: TranscriptionInput) -> Union[Dict[str, Any], str]:
        # Process the audio file from memory
        audio_data = io.BytesIO(input.file)

        # Set up transcription options
        options = {
            "task": "transcribe",
            "temperature": [input.temperature],
            "initial_prompt": input.prompt,
        }

        segments_generator, info = self.model.transcribe(audio_data, **options)
        segments = []
        for s in segments_generator:
            await asyncio.sleep(0)
            segments.append(s)
        if input.format == "verbose_json":
            result = {
                "text": " ".join([segment.text for segment in segments]),
                "language": info.language,
                "duration": info.duration,
                "segments": [
                    {
                        "id": segment.id,
                        "seek": segment.seek,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "tokens": segment.tokens,
                        "temperature": segment.temperature,
                        "avg_logprob": segment.avg_logprob,
                        "compression_ratio": segment.compression_ratio,
                        "no_speech_prob": segment.no_speech_prob,
                        "words": segment.words if hasattr(segment, 'words') else None
                    } for segment in segments
                ]
            }
        else:  # Default to simple json format
            result = {
                "text": " ".join([segment.text for segment in segments])
            }
        
        return result
