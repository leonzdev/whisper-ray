#!/usr/bin/env sh
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@${1}" \
     -F "model=tiny" \
     -F "prompt=" \
     -F "temperature=0.0" \
     -F "language=" \
     -F "response_format=verbose_json" \
     -F "timestamp_granularities[]=word" \
     -F "timestamp_granularities[]=segment"
