"""
RAVDESS Trimodal Dataset
PyTorch Dataset for Audio + Video + Text (ASR transcripts)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, AutoTokenizer
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class VideoMAETransform:
    """Transform for VideoMAE preprocessing"""
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __call__(self, frames):
        """
        Args:
            frames: List of numpy arrays (H, W, C)
        Returns:
            Tensor of shape (num_frames, C, H, W)
        """
        return torch.stack([self.transform(frame) for frame in frames])


class RAVDESSTrimodalDataset(Dataset):
    """
    RAVDESS Trimodal Dataset
    Returns: audio, video frames, text tokens, label
    """
    
    def __init__(
        self,
        metadata_csv,
        base_dir,
        split='train',
        num_frames=16,
        sample_rate=16000,
        audio_duration=3.0,
        augment=False,
        audio_noise_prob=0.5,
        video_drop_prob=0.5,
        use_whisper_transcription=False
    ):
        """
        Args:
            metadata_csv: Path to metadata.csv
            base_dir: Base directory of the project
            split: 'train', 'val', or 'test'
            num_frames: Number of frames to sample from video
            sample_rate: Audio sample rate
            audio_duration: Duration of audio in seconds
            augment: Whether to apply augmentation
            audio_noise_prob: Probability of adding audio noise
            video_drop_prob: Probability of dropping video frames
            use_whisper_transcription: Use Whisper for transcription (slow, set False for simple placeholder)
        """
        self.base_dir = Path(base_dir)
        self.split = split
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.audio_length = int(sample_rate * audio_duration)
        self.augment = augment and (split == 'train')
        self.audio_noise_prob = audio_noise_prob
        self.video_drop_prob = video_drop_prob
        self.use_whisper = use_whisper_transcription
        
        # Load metadata
        df = pd.read_csv(metadata_csv)
        self.data = df[df['split'] == split].reset_index(drop=True)
        
        # Initialize processors
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.video_transform = VideoMAETransform(image_size=224)
        
        # Emotion to ID mapping
        self.emotion_to_id = {
            'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
            'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
        }
        
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def __len__(self):
        return len(self.data)
    
    def load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            # Load audio
            waveform, sr = sf.read(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            
            # Pad or trim to fixed length
            if len(waveform) < self.audio_length:
                waveform = np.pad(waveform, (0, self.audio_length - len(waveform)))
            else:
                waveform = waveform[:self.audio_length]
            
            # Apply noise augmentation
            if self.augment and np.random.rand() < self.audio_noise_prob:
                noise = np.random.randn(len(waveform)) * 0.005
                waveform = waveform + noise
            
            return waveform
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return np.zeros(self.audio_length)
    
    def load_video(self, video_path):
        """Load and preprocess video frames"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                # Return black frames if video fails to load
                frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
            else:
                # Sample frames uniformly
                indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Apply video drop augmentation (black out frames)
            if self.augment and np.random.rand() < self.video_drop_prob:
                drop_mask = np.random.rand(len(frames)) < 0.3  # Drop 30% of frames
                frames = [np.zeros_like(f) if drop else f for f, drop in zip(frames, drop_mask)]
            
            return frames
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
    
    def get_text_transcript(self, audio_path):
        """
        Get text transcript from audio using Whisper ASR
        Uses the tiny model for speed, can upgrade to base/small for accuracy
        """
        if self.use_whisper:
            try:
                import whisper
                # Load model once (cached after first load)
                if not hasattr(self, '_whisper_model'):
                    self._whisper_model = whisper.load_model("tiny", device="cpu")
                
                result = self._whisper_model.transcribe(
                    str(audio_path),
                    language='en',
                    fp16=False
                )
                return result["text"].strip()
            except Exception as e:
                # Fallback if Whisper fails
                print(f"Whisper transcription failed for {audio_path}: {e}")
                return self._get_fallback_text(audio_path)
        else:
            # Use statement-based text (more reliable than emotion-based)
            return self._get_statement_text(audio_path)
    
    def _get_statement_text(self, audio_path):
        """
        Extract actual statement text from filename
        RAVDESS has two statements:
        - Statement 01: "Kids are talking by the door"
        - Statement 02: "Dogs are sitting by the door"
        """
        filename = audio_path.stem
        parts = filename.split('-')
        
        if len(parts) >= 5:
            statement_code = parts[4]
            if statement_code == '01':
                return "Kids are talking by the door"
            elif statement_code == '02':
                return "Dogs are sitting by the door"
        
        # Fallback
        return "Someone is speaking"
    
    def _get_fallback_text(self, audio_path):
        """Fallback text based on statement"""
        return self._get_statement_text(audio_path)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        row = self.data.iloc[idx]
        
        # Load modalities
        audio_path = self.base_dir / row['audio_path']
        video_path = self.base_dir / row['video_path']
        
        # Audio
        audio_waveform = self.load_audio(audio_path)
        audio_inputs = self.audio_processor(
            audio_waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Video
        video_frames = self.load_video(video_path)
        video_tensor = self.video_transform(video_frames)  # (T, C, H, W)
        
        # Text
        text = self.get_text_transcript(audio_path)
        text_inputs = self.text_tokenizer(
            text,
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )
        
        # Label
        emotion = row['emotion']
        label = self.emotion_to_id[emotion]
        
        return {
            'audio_input_values': audio_inputs.input_values.squeeze(0),
            'video_pixel_values': video_tensor,
            'text_input_ids': text_inputs.input_ids.squeeze(0),
            'text_attention_mask': text_inputs.attention_mask.squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    metadata_csv,
    base_dir,
    batch_size=16,
    num_workers=4,
    augment_train=True
):
    """
    Create train, val, test dataloaders
    """
    train_dataset = RAVDESSTrimodalDataset(
        metadata_csv=metadata_csv,
        base_dir=base_dir,
        split='train',
        augment=augment_train
    )
    
    val_dataset = RAVDESSTrimodalDataset(
        metadata_csv=metadata_csv,
        base_dir=base_dir,
        split='val',
        augment=False
    )
    
    test_dataset = RAVDESSTrimodalDataset(
        metadata_csv=metadata_csv,
        base_dir=base_dir,
        split='test',
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
