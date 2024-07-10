import os
# Deploy the API
from deployment.api import APIIngress
from deployment.transcribe import TranscribeService

whisper_ray = APIIngress.bind(TranscribeService.bind(os.getenv("MODEL_NAME"), os.getenv("MODEL_DEVICE")))
