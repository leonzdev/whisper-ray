# whisper-ray
Dual-mode Whisper API server
# Features
* APIs compatible with OpenAI's hosted audio transcribe and translate APIs
* Dual-mode: run as either a FastAPI app or Ray Serve app

# How-to
* Run as a FastAPI app: `scripts/dev_fastapi.sh`
* Run as a Ray Serve app: `serve run dev-config.yaml`
