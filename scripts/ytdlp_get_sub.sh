#!/bin/bash

set -ueo pipefail

# Default URL
url=${1:-"https://www.youtube.com/watch?v=TepmKvnzjIg"}

# Config for "No Cookies" mode
# Using Android client emulation is currently one of the best ways to bypass bot detection without cookies.
# https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
UA="com.google.android.youtube/19.22.36 (Linux; U; Android 13) gzip"

# Extractor arguments:
# - player_client=android: Use the Android API which is less restricted.
# - player_skip=webpage,configs: Skip downloading the HTML webpage and internal configs to reduce requests and avoid some scraping blocks.
EXTRACTOR_ARGS="youtube:player_client=android;player_skip=webpage,configs"

echo "Retrieving subtitle list for: ${url}..."
echo "========================================"

# List subs
yt-dlp \
    --extractor-args "${EXTRACTOR_ARGS}" \
    --user-agent "${UA}" \
    --list-subs \
    "${url}"

echo "========================================"
echo "Enter the language codes you wish to download (comma-separated)."
echo "Examples: 'en', 'en,ja', 'all', 'en.*' (regex)"
read -p "Selection > " selection

if [ -z "$selection" ]; then
    echo "No selection provided. Exiting."
    exit 0
fi

# Output directory logic
ts=$(date +%m%d%H%M%S)
out=${DATA_DIR:-data}/${ts}
mkdir -p "$out"

echo "Downloading subtitles for languages: ${selection} into ${out}"

# Download subs
yt-dlp \
    --extractor-args "${EXTRACTOR_ARGS}" \
    --user-agent "${UA}" \
    --skip-download \
    --write-subs \
    --sub-langs "${selection}" \
    --convert-subs "srt" \
    -P "${out}" --output "sub.%(ext)s" \
    "${url}"

echo "Done. Files in ${out}:"
find "${out}" -type f
