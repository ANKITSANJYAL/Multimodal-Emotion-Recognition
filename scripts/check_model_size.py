import torch
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from modules.affect_diff_module import AffectDiffModule

def get_model_size():
    print("Calculating Model Size and Parameters...")
    
    # Configuration matches config.yaml
    model = AffectDiffModule(
        text_dim=300,
        audio_dim=74,
        video_dim=35,
        latent_dim=256,
        num_classes=6,
        diffusion_steps=1000,
        lr=1e-4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    print("-" * 50)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Estimated Model Size:  {size_all_mb:.2f} MB")
    print("-" * 50)
    
    # Breakdown by component
    print("\nComponent Breakdown:")
    components = {
        "Bottleneck (VAE + Encoders)": model.bottleneck,
        "Causal Graph": model.bottleneck.causal_graph,
        "Classifier": model.classifier,
        "UNet1D (Generative)": model.unet
    }
    
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:<30} : {params:,} params")

if __name__ == "__main__":
    get_model_size()
