#!/bin/bash
# Script to extract heavy video/audio features
# Example usage: bash extract_features.sh

python Data/preprocessors/extract_audio_features.py
python Data/preprocessors/extract_video_features.py
