import os
from core.whisper_worker import WhisperWorker
from core.api import APIIngress, app
from .whisper_worker import RemoteWhisperWorker
from ray import serve

APIIngressDeployment = serve.deployment(serve.ingress(app)(APIIngress))
WhisperModelDeployment = serve.deployment(WhisperWorker)

remote_whisper_worker = RemoteWhisperWorker(WhisperModelDeployment.bind(
    os.getenv("MODEL_NAME"), 
    os.getenv("MODEL_DEVICE"), 
    int(os.getenv("MODEL_CPU_THREADS")) if os.getenv("MODEL_CPU_THREADS") else 0,
    os.getenv("MODEL_GPU_FLASH_ATTENTION") == "true"
))
whisper_ray = APIIngressDeployment.bind(remote_whisper_worker)
