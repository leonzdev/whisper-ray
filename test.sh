#!/usr/bin/env sh
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@${1}" \
     -F "model=base" \
     -F "prompt=" \
     -F "temperature=0.0" \
     -F "language=" \
     -F "format=verbose_json"
