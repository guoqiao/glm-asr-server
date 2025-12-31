#!/bin/bash

set -ueo pipefail

file=${1:-data/audio.mp3}

curl -s -X POST "http://localhost:8000/v1/audio/transcriptions" \
    -H "Content-Type: multipart/form-data" \
    -F "language=zh" \
    -F "response_format=json" \
    -F "file=@${file}" | jq