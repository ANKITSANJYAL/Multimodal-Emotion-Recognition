"""
RAVDESS Dataset Preprocessing Script
Parses video filenames, extracts audio, and generates metadata.csv
"""

import os
import pandas as pd
from pathlib import Path
import subprocess
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Emotion labels mapping (3rd identifier in filename)
EMOTION_DICT = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def parse_filename(filename):
    """
    Parse RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.mp4
    Example: 01-01-06-01-02-01-12.mp4
    
    Returns: emotion_label, actor_id, emotion_id, emotion_code
    
    Filename identifiers:
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    - Vocal channel (01 = speech, 02 = song)
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    - Emotional intensity (01 = normal, 02 = strong)
    - Statement (01 = "Kids...", 02 = "Dogs...")
    - Repetition (01 = 1st, 02 = 2nd)
    - Actor (01 to 24)
    """
    parts = filename.replace('.mp4', '').split('-')
    if len(parts) != 7:
        return None, None, None, None
    
    emotion_code = parts[2]  # 3rd position (0-indexed)
    actor_id = int(parts[6])  # 7th position
    emotion_label = EMOTION_DICT.get(emotion_code, None)
    
    # Direct conversion: '01' -> 0, '02' -> 1, ..., '08' -> 7
    emotion_id = int(emotion_code) - 1 if emotion_label else None
    
    return emotion_label, actor_id, emotion_id, emotion_code

def extract_audio_from_video(video_path, audio_output_path):
    """
    Extract audio from video using ffmpeg
    """
    if os.path.exists(audio_output_path):
        return True
    
    try:
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-loglevel', 'quiet',
            str(audio_output_path)
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    videos_dir = data_dir / 'videos'
    audio_extract_dir = data_dir / 'audio_extracted'
    audio_extract_dir.mkdir(exist_ok=True)
    
    # Collect all video files
    print("Scanning video files...")
    video_files = []
    for actor_dir in sorted(videos_dir.glob('Actor_*')):
        for video_file in actor_dir.glob('*.mp4'):
            video_files.append(video_file)
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    metadata = []
    print("Processing videos and extracting audio...")
    
    for video_path in tqdm(video_files, desc="Processing"):
        filename = video_path.name
        emotion_label, actor_id, emotion_id, emotion_code = parse_filename(filename)
        
        if emotion_label is None or actor_id is None:
            continue
        
        # Create audio output path
        actor_audio_dir = audio_extract_dir / f'Actor_{actor_id:02d}'
        actor_audio_dir.mkdir(exist_ok=True)
        audio_path = actor_audio_dir / filename.replace('.mp4', '.wav')
        
        # Extract audio
        success = extract_audio_from_video(video_path, audio_path)
        
        if success:
            # Determine split (subject-independent)
            # Train: Actors 1-20, Val/Test: Actors 21-24
            if actor_id <= 20:
                split = 'train'
            else:
                split = 'test'  # We'll split test into val/test later if needed
            
            metadata.append({
                'video_path': str(video_path.relative_to(base_dir)),
                'audio_path': str(audio_path.relative_to(base_dir)),
                'emotion': emotion_label,
                'emotion_id': emotion_id,  # Now correctly 0-indexed (0-7)
                'actor_id': actor_id,
                'filename': filename,
                'split': split
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(metadata)
    
    # Further split test into val and test (50-50 from actors 21-24)
    test_df = df[df['split'] == 'test'].copy()
    test_actors = test_df['actor_id'].unique()
    val_actors = test_actors[:2]  # Actors 21, 22
    
    df.loc[df['actor_id'].isin(val_actors), 'split'] = 'val'
    
    # Save metadata
    metadata_path = data_dir / 'metadata.csv'
    df.to_csv(metadata_path, index=False)
    
    print(f"\n{'='*50}")
    print("Preprocessing Complete!")
    print(f"{'='*50}")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(df[df['split'] == 'train'])}")
    print(f"Val samples: {len(df[df['split'] == 'val'])}")
    print(f"Test samples: {len(df[df['split'] == 'test'])}")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts().sort_index())
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
