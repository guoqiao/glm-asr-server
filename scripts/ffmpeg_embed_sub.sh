#!/bin/bash

set -ueo pipefail

# Embed a subtitle file into a video file.

sub="${1:-sub.zh.vtt}"
lang="${2:-eng}"
video_in="${3:-input.mp4}"
video_out="${4:-output.mp4}"

ffmpeg -i ${video_in} -i ${sub} -c copy -c:s mov_text -metadata:s:s:0 language=${lang} ${video_out}
