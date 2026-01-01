#!/bin/bash

# use yt-dlp to get decent audio from youtube for transcription
# the audio quality doesn't need to be best/high, as long as it's good enough for transcription
# keep the final audio file size as small as possible

set -ueo pipefail

# Default URL
url=${1:-"https://www.youtube.com/watch?v=TepmKvnzjIg"}

# Config to avoid bot detection
# For audio-only, the 'android' client often fails to return formats without PO Token.
# 'ios' or 'web' (default) are often better for audio, but 'web' has more bot detection.
# We will try 'ios' which is a good middle ground, or fallback to default if that fails.
# However, 'android' generally works for video but is picky about audio-only formats.

# Let's try to be less restrictive for audio since we don't need 4K video.
# We will use the default client but with User-Agent spoofing, which is usually enough for audio.
# If that fails, we can try 'ios'.

UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

echo "Downloading audio for: ${url}..."

# -f "ba[ext=m4a]/ba[ext=webm]/ba": Select best audio-only source, prefer m4a/webm
# -x --audio-format mp3: Extract audio and convert to mp3
# --audio-quality 64K: Variable bitrate target ~64kbps (VBR 9-ish in ffmpeg terms usually, but yt-dlp maps it).
#                      64k is sufficient for speech recognition.
# --output: Save with title and id
yt-dlp \
    --user-agent "${UA}" \
    -f "ba" \
    -x --audio-format mp3 \
    --audio-quality 64K \
    --output "%(id)s.%(ext)s" \
    "${url}"

echo "Done."
