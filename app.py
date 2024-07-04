from fastapi import FastAPI, UploadFile, File, Form
from ray import serve
from ray.serve.handle import DeploymentHandle
from transcribe_service import TranscriptionService, TranscriptionInput
app = FastAPI()

@serve.deployment(route_prefix="/task")
@serve.ingress(app)
class APIIngress:
    def __init__(self, transcription_service: DeploymentHandle) -> None:
        self.transcription_service = transcription_service

    @app.post("/transcribe")
    async def transcribe(self, file: UploadFile = File(...), model: str = Form(...), prompt: str = Form(None), temperature: float = Form(0.0), language: str = Form(None), format: str = Form("json")):
        # Read the file content
        file_content = await file.read()
        input_data = TranscriptionInput(
            file=file_content,
            model=model,
            prompt=prompt,
            temperature=temperature,
            language=language,
            format=format
        )
        # Call the transcription service
        result = await self.transcription_service.transcribe.remote(input_data)
        return result

# Deploy the API
serve.run(APIIngress.bind(TranscriptionService.bind()))
