import os
# Deploy the API
from .deployment.api import APIIngress
from .deployment.model import WhisperModelService

whisper_ray = APIIngress.bind(WhisperModelService.bind(
    os.getenv("MODEL_NAME"), 
    os.getenv("MODEL_DEVICE"), 
    int(os.getenv("MODEL_CPU_THREADS")) if os.getenv("MODEL_CPU_THREADS") else 0
))
