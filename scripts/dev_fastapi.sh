#!/usr/bin/env sh
export MODEL_NAME=tiny
export MODEL_DEVICE=cpu
export MODEL_CPU_THREADS="4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fastapi dev ${SCRIPT_DIR}/../fastapi_app/main.py