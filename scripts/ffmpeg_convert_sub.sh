#!/bin/bash

set -ueo pipefail

i=${1:-sub.vtt}
o=${2:-sub.srt}

ffmpeg -i ${i} ${o}

