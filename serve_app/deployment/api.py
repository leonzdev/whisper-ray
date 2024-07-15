from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.datastructures import FormData
from ray import serve
from ray.serve.handle import DeploymentHandle
from ..common.constants import FORMAT_JSON, SEGMENT
from ..common.types import TranscribeInput

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class APIIngress:
    def __init__(self, whisper_model_service: DeploymentHandle) -> None:
        self.transcribe_service = whisper_model_service

    @staticmethod
    def parse_timestamp_granularities(form_data: FormData) -> List[str]:
        timestamp_granularities: List[str] = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            return [SEGMENT]
        return timestamp_granularities

    @app.post("/v1/audio/transcriptions")
    async def transcribe(self, 
        request: Request,
        file: UploadFile = File(...), 
        model: str = Form(...), 
        language: str = Form(None), 
        prompt: str = Form(None), 
        response_format: str = Form(FORMAT_JSON),
        temperature: float = Form(0.0),
        # timestamp_granularities: This parameter needs to be parsed from request directly
    ):
        form_data = await request.form()
        file_content = await file.read()
        timestamp_granularities: List[str] = self.parse_timestamp_granularities(form_data)
        input_data = TranscribeInput(
            file=file_content,
            model=model,
            prompt=prompt,
            temperature=temperature,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities
        )
        # Call the transcription service
        try:
            result = await self.transcribe_service.transcribe.remote(input_data)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
