import os
from core.api import APIIngress, app
from core.whisper_worker import WhisperWorker
from .utils import make_fastapi_class_based_view

model = WhisperWorker(os.getenv("MODEL_NAME"), 
    os.getenv("MODEL_DEVICE"), 
    int(os.getenv("MODEL_CPU_THREADS")) if os.getenv("MODEL_CPU_THREADS") else 0,
    os.getenv("MODEL_GPU_FLASH_ATTENTION") == "true"
)
master_controller = APIIngress(model)
make_fastapi_class_based_view(app, APIIngress, master_controller)

@app.on_event("startup")
async def startup_event():
    print("Starting up the application...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down the application...")
