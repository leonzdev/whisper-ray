from enum import Enum
from typing import List, Union, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.datastructures import FormData
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.exceptions import BackPressureError
from ..common.constants import FORMAT_JSON, SEGMENT, FORMAT_VERBOSE
from ..common.types import TranscribeInput, TranslateInput, ModelServiceConfig

app = FastAPI()

class Task(Enum):
    transcribe = "transcribe"
    translate = "translate"

@serve.deployment
@serve.ingress(app)
class APIIngress:
    def __init__(self, preferred_app_name, preferred_dep_name, backup_app_name, backup_dep_name) -> None:
        self.model_services_config = [
            ModelServiceConfig(app_name=preferred_app_name, deployment_name=preferred_dep_name),
            ModelServiceConfig(app_name=backup_app_name, deployment_name=backup_dep_name)
        ]

    @staticmethod
    def parse_timestamp_granularities(form_data: FormData) -> List[str]:
        timestamp_granularities: List[str] = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            return [SEGMENT]
        return timestamp_granularities

    def get_preferred_model(self) -> DeploymentHandle:
        return self.get_model_by_index(0)
    def get_backup_model(self) -> DeploymentHandle:
        return self.get_model_by_index(1)
    def get_model_by_index(self, index: int) -> DeploymentHandle:
        return serve.get_deployment_handle(
            app_name=self.model_services_config[index].app_name,
            deployment_name=self.model_services_config[index].deployment_name
        )

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
        # see: https://github.com/tiangolo/fastapi/issues/842
        # see: https://github.com/tiangolo/fastapi/issues/3532
        # see: https://github.com/tiangolo/fastapi/discussions/8741
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
        return await self.run_task(input_data, response_format, Task.transcribe)

    @app.post("/v1/audio/translations")
    async def translate(self, 
        file: UploadFile = File(...), 
        model: str = Form(...), 
        prompt: str = Form(None), 
        response_format: str = Form(FORMAT_JSON),
        temperature: float = Form(0.0),
    ):
        file_content = await file.read()
        input_data = TranslateInput(
            file=file_content,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature
        )
        return await self.run_task(input_data, response_format, Task.translate)

    async def run_task(self, input_data: Union[TranslateInput, TranscribeInput], response_format: str, task: Task) -> Any:
        result = None
        try:
            preferred_model = self.get_preferred_model()
            # min max_queued_requests is 1 
            # I *think* there is a bug that max_queued_requests=1 actually requires 2 queued requests to create backpressure
            # When using the __call__ method (implicitly) to create backpressure it takes 3 requests
            preferred_model.ping.remote()
            preferred_model.ping.remote()
            result = await self.call_model(preferred_model, input_data, task)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except BackPressureError as e:
            result = await self.call_model(self.get_backup_model(), input_data, task)
        if response_format in [FORMAT_JSON, FORMAT_VERBOSE]:
            return JSONResponse(result)
        return PlainTextResponse(result)

    @staticmethod
    async def call_model(model, input_data, task):
        if task == Task.transcribe:
            return await model.transcribe.remote(input_data)
        elif task == Task.translate:
            return await model.translate.remote(input_data)
        else:
            raise HTTPException(status_code=500, detail="invalid internal task {}".format(task))
