import torch
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from modules.affect_diff_module import AffectDiffModule

def test_structure():
    print("Initializing AffectDiffModule (DeepMind Level Architecture)...")
    
    # Configuration matches config.yaml
    model = AffectDiffModule(
        text_dim=300,
        audio_dim=74,
        video_dim=35,
        latent_dim=256,
        num_classes=6,
        diffusion_steps=100, # Small for testing
        lr=1e-4
    )
    
    batch_size = 4
    seq_len = 50
    
    # Dummy data
    batch = {
        'text': torch.randn(batch_size, seq_len, 300),
        'audio': torch.randn(batch_size, seq_len, 74),
        'vision': torch.randn(batch_size, seq_len, 35),
        'labels': torch.randint(0, 6, (batch_size,))
    }
    
    print("\n[Test 1] Standard Forward Pass (Inference)")
    logits = model(batch['text'], batch['audio'], batch['vision'])
    print(f"Logits shape: {logits.shape}") 
    assert logits.shape == (batch_size, 6)
    
    print("\n[Test 2] Shared Step (Training Logic)")
    loss = model.shared_step(batch, 0, stage="train")
    print(f"Total Loss: {loss.item():.4f}")
    assert not torch.isnan(loss)
    
    print("\n[Test 3] Causal Influence Logging Check")
    z_permuted, mu, logvar, adj_matrix = model.bottleneck(batch['text'], batch['audio'], batch['vision'])
    weights = model.bottleneck.causal_graph.get_causal_weights(adj_matrix)
    print("Causal Weights:", weights)
    assert 'Text_Influence' in weights

    print("\nStructure Validated Successfully!")

if __name__ == "__main__":
    test_structure()
