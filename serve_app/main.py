import os
# Deploy the API
from .deployment.api import APIIngress
from .deployment.whisper_model import WhisperModelService
from .common.types import ModelServiceConfig
# model_name, device, cpu_threads, flash_attention
gpu_model = WhisperModelService.options(name=os.getenv("DEP_NAME")).bind(
    os.getenv("MODEL_NAME"), 
    "cuda",
    0,
    os.getenv("MODEL_GPU_FLASH_ATTENTION") == "true"
)
cpu_model = WhisperModelService.options(name=os.getenv("DEP_NAME")).bind(
    os.getenv("MODEL_NAME"),
    "cpu",
    int(os.getenv("MODEL_CPU_THREADS")) if os.getenv("MODEL_CPU_THREADS") else 0,
    False
)
whisper_ray = APIIngress.options(name="whisper_api").bind(
    os.getenv("GPU_APP_NAME"), os.getenv("GPU_DEP_NAME"),
    os.getenv("CPU_APP_NAME"), os.getenv("CPU_DEP_NAME"),
)
