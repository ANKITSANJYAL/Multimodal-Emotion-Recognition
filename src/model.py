"""
Trimodal Transformer Architecture
Audio (Wav2Vec2) + Video (VideoMAE) + Text (DistilRoBERTa)
Late Fusion with MLP Classifier
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    AutoModel,
    VideoMAEModel,
    VideoMAEConfig
)
from typing import Optional


class TrimodalClassifier(nn.Module):
    """
    Trimodal Emotion Recognition Model
    - Audio: Wav2Vec2 (frozen feature extractor, trainable projection)
    - Video: VideoMAE 
    - Text: DistilRoBERTa
    - Fusion: Concatenation + MLP
    """
    
    def __init__(
        self,
        num_classes=8,
        audio_model_name="facebook/wav2vec2-base-960h",
        video_model_name="MCG-NJU/videomae-base",
        text_model_name="distilroberta-base",
        hidden_dim=512,
        dropout=0.3,
        freeze_audio_encoder=True,
        freeze_video_encoder=False,
        freeze_text_encoder=False,
        use_gradient_checkpointing=False
    ):
        """
        Args:
            num_classes: Number of emotion classes (8 for RAVDESS)
            audio_model_name: HuggingFace model for audio
            video_model_name: HuggingFace model for video
            text_model_name: HuggingFace model for text
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
            freeze_audio_encoder: Freeze Wav2Vec2 feature extractor
            freeze_video_encoder: Freeze VideoMAE encoder
            freeze_text_encoder: Freeze text encoder
            use_gradient_checkpointing: Use gradient checkpointing to save memory
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # =============== Audio Branch (Wav2Vec2) ===============
        print(f"Loading audio model: {audio_model_name}")
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        
        if freeze_audio_encoder:
            # Freeze feature extractor
            self.audio_encoder.feature_extractor._freeze_parameters()
            print("Frozen Wav2Vec2 feature extractor")
        
        if use_gradient_checkpointing:
            self.audio_encoder.gradient_checkpointing_enable()
        
        self.audio_dim = self.audio_encoder.config.hidden_size  # 768
        
        # =============== Video Branch (VideoMAE) ===============
        print(f"Loading video model: {video_model_name}")
        self.video_encoder = VideoMAEModel.from_pretrained(video_model_name)
        
        if freeze_video_encoder:
            for param in self.video_encoder.parameters():
                param.requires_grad = False
            print("Frozen VideoMAE encoder")
        
        if use_gradient_checkpointing:
            self.video_encoder.gradient_checkpointing_enable()
        
        self.video_dim = self.video_encoder.config.hidden_size  # 768
        
        # =============== Text Branch (DistilRoBERTa) ===============
        print(f"Loading text model: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("Frozen text encoder")
        
        if use_gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
        
        self.text_dim = self.text_encoder.config.hidden_size  # 768
        
        # =============== Fusion Layer ===============
        self.fusion_dim = self.audio_dim + self.video_dim + self.text_dim  # 2304
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"Model initialized with fusion dim: {self.fusion_dim}")
        print(f"Total parameters: {self.count_parameters():,}")
        print(f"Trainable parameters: {self.count_parameters(trainable_only=True):,}")
    
    def count_parameters(self, trainable_only=False):
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        audio_input_values: torch.Tensor,
        video_pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass
        
        Args:
            audio_input_values: (B, T) audio waveform
            video_pixel_values: (B, num_frames, C, H, W) video frames
            text_input_ids: (B, seq_len) text token ids
            text_attention_mask: (B, seq_len) text attention mask
            labels: (B,) emotion labels
        
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        batch_size = audio_input_values.size(0)
        
        # =============== Audio Encoding ===============
        audio_outputs = self.audio_encoder(audio_input_values)
        audio_features = audio_outputs.last_hidden_state  # (B, T, 768)
        audio_pooled = audio_features.mean(dim=1)  # (B, 768)
        
        # =============== Video Encoding ===============
        # VideoMAE expects (B, C, T, H, W)
        B, T, C, H, W = video_pixel_values.shape
        video_input = video_pixel_values.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        video_outputs = self.video_encoder(video_input)
        video_features = video_outputs.last_hidden_state  # (B, num_patches, 768)
        video_pooled = video_features.mean(dim=1)  # (B, 768)
        
        # =============== Text Encoding ===============
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state  # (B, seq_len, 768)
        text_pooled = text_features[:, 0, :]  # Use [CLS] token (B, 768)
        
        # =============== Fusion ===============
        fused_features = torch.cat([audio_pooled, video_pooled, text_pooled], dim=1)  # (B, 2304)
        
        # =============== Classification ===============
        logits = self.classifier(fused_features)  # (B, num_classes)
        
        # =============== Loss Calculation ===============
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def get_feature_representations(
        self,
        audio_input_values: torch.Tensor,
        video_pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor
    ):
        """
        Extract feature representations without classification
        Useful for visualization and analysis
        """
        with torch.no_grad():
            # Audio
            audio_outputs = self.audio_encoder(audio_input_values)
            audio_pooled = audio_outputs.last_hidden_state.mean(dim=1)
            
            # Video
            B, T, C, H, W = video_pixel_values.shape
            video_input = video_pixel_values.permute(0, 2, 1, 3, 4)
            video_outputs = self.video_encoder(video_input)
            video_pooled = video_outputs.last_hidden_state.mean(dim=1)
            
            # Text
            text_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_pooled = text_outputs.last_hidden_state[:, 0, :]
            
            # Fused
            fused_features = torch.cat([audio_pooled, video_pooled, text_pooled], dim=1)
        
        return {
            'audio_features': audio_pooled,
            'video_features': video_pooled,
            'text_features': text_pooled,
            'fused_features': fused_features
        }


if __name__ == '__main__':
    # Test model instantiation
    print("Testing model...")
    model = TrimodalClassifier(
        num_classes=8,
        freeze_audio_encoder=True,
        use_gradient_checkpointing=False
    )
    
    # Create dummy inputs
    batch_size = 2
    audio_input = torch.randn(batch_size, 48000)  # 3 seconds at 16kHz
    video_input = torch.randn(batch_size, 16, 3, 224, 224)  # 16 frames
    text_input_ids = torch.randint(0, 1000, (batch_size, 32))
    text_attention_mask = torch.ones(batch_size, 32)
    labels = torch.randint(0, 8, (batch_size,))
    
    # Forward pass
    outputs = model(
        audio_input_values=audio_input,
        video_pixel_values=video_input,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        labels=labels
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print("Model test passed!")
