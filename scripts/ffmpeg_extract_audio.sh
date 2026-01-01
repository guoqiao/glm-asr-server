#!/bin/bash

set -ueo pipefail

# extract decent audio from video/audio for transcription

video="${1:-video.mp4}"
# same path, but with .m4a extension
audio="${video%.*}.m4a"

ffmpeg -i ${video} -vn -ar 16000 -ac 1 -b:a 48k -c:a aac ${audio}
