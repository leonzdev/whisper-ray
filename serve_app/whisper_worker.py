from typing import Any, Dict
from common.types import AbstractWhisperWorker, TranscribeInput, TranslateInput
from ray.serve.handle import DeploymentHandle

class RemoteWhisperWorker(AbstractWhisperWorker):
    def __init__(self, whisper_model_service: DeploymentHandle):
        self.whisper_model_service = whisper_model_service

    async def transcribe(self, input: TranscribeInput) -> Dict[str, Any] | str:
        return await self.whisper_model_service.transcribe.remote(input)
    
    async def translate(self, input: TranslateInput) -> Dict[str, Any]:
        return await self.whisper_model_service.translate.remote(input)
