import os
import asyncio
import io
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List, Iterable

from fastapi import FastAPI, UploadFile, File, Form
from ray import serve
from ray.serve.handle import DeploymentHandle
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class APIIngress:
    def __init__(self, transcribe_service: DeploymentHandle) -> None:
        self.transcribe_service = transcribe_service

    @app.post("/transcribe")
    async def transcribe(self, file: UploadFile = File(...), model: str = Form(...), prompt: str = Form(None), temperature: float = Form(0.0), language: str = Form(None), format: str = Form("json")):
        # Read the file content
        file_content = await file.read()
        input_data = TranscribeInput(
            file=file_content,
            model=model,
            prompt=prompt,
            temperature=temperature,
            language=language,
            format=format
        )
        # Call the transcription service
        result = await self.transcribe_service.transcribe.remote(input_data)
        return result



# Define input schema compatible with OpenAI's API
class TranscribeInput(BaseModel):
    file: bytes
    model: str = "base"
    prompt: Optional[str] = None
    temperature: float = 0.0
    language: Optional[str] = None
    format: str = "json"

# Initialize Faster-Whisper model

@serve.deployment()
class TranscribeService:

    def __init__(self, model_name: str, device: str):
        from faster_whisper import WhisperModel
        from faster_whisper.transcribe import TranscriptionOptions, Segment
        self.model = WhisperModel(model_name, device)
    
    async def transcribe(self, input: TranscribeInput) -> Union[Dict[str, Any], str]:
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

whisper_ray = APIIngress.bind(TranscribeService.bind(os.getenv("MODEL_NAME"), os.getenv("MODEL_DEVICE")))
