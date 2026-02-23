import matplotlib.pyplot as plt
import numpy as np
import wandb

class AffectiveVisualizer:
    """
    Generates plots to visualize multimodal dissonance and counterfactual shifts
    for Weights & Biases logging.
    """
    def __init__(self, class_names=["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]):
        self.class_names = class_names

    def plot_counterfactual_shift(self, p_factual, p_counterfactual, modality_ablated="Audio"):
        """
        Creates a side-by-side bar chart showing how the emotion prediction
        shifted when a modality was counterfactually hallucinated.
        """
        # Convert to numpy arrays
        p_fac = p_factual.cpu().numpy()[0] # Take first item in batch
        p_cf = p_counterfactual.cpu().numpy()[0]

        x = np.arange(len(self.class_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, p_fac, width, label='Factual (All Modalities)', color='#1f77b4')
        rects2 = ax.bar(x + width/2, p_cf, width, label=f'Counterfactual (No {modality_ablated})', color='#ff7f0e')

        ax.set_ylabel('Probability')
        ax.set_title(f'Affective Shift upon {modality_ablated} Ablation & Generative Healing')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()

        # Convert matplotlib figure to a W&B Image
        wandb_img = wandb.Image(fig)
        plt.close(fig) # Prevent memory leaks in training loop

        return wandb_img

    def plot_causal_graph_weights(self, weights_dict, step):
        """
        Logs the normalized importance of Text vs Audio vs Video over time.
        """
        wandb.log({
            "Causal_Graph/Text_Influence": weights_dict['Text_Influence'],
            "Causal_Graph/Audio_Influence": weights_dict['Audio_Influence'],
            "Causal_Graph/Video_Influence": weights_dict['Video_Influence'],
            "global_step": step
        })
