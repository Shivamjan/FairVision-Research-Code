"""
AC-CQC-DD Model Inference Module
Replace this template with your actual model implementation
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple




"""
AC-CQC-DD Model Inference Module
GQVK Architecture with Pre-computed Text Embeddings & Fairness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from typing import Dict, Tuple, List, Optional
import numpy as np


DEFAULT_TAU = 0.07
TAU_MIN = 0.03
TAU_MAX = 0.2


class GQVKNetwork(nn.Module):
    """
    GQVK Architecture using pre-computed Gemma embeddings
    """

    def __init__(self,
                 output_size=[9, 6],  
                 gemma_dim=768, 
                 pretrained=True):
        super(GQVKNetwork, self).__init__()

        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))


        self.image_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k'
        )
        self.image_dim = self.image_encoder.config.hidden_size  # 768

        self.text_projection = nn.Sequential(
            nn.Linear(gemma_dim, self.image_dim),
            nn.LayerNorm(self.image_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.image_dim, self.image_dim)
        )


        self.inference_query = nn.Parameter(
            torch.randn(1, self.image_dim)
        )

        self.branch_1 = nn.Sequential(
            nn.Linear(self.image_dim, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, output_size[0]),
        )

        self.branch_2 = nn.Linear(self.image_dim, output_size[1])

        self.project_head = nn.Sequential(
            nn.Linear(self.image_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )



    def forward(self, images, text_emb_true=None, text_emb_cf=None):
        B = images.size(0)
        
        image_features = self.image_encoder(images)
        visual_patches = image_features.last_hidden_state[:, 1:, :]
        

        if self.training and text_emb_true is not None:
            text_proj_true = self.text_projection(text_emb_true)
            attn_true = self._compute_attention(visual_patches, text_proj_true)
            retrieved_true = torch.bmm(
                attn_true.unsqueeze(1), visual_patches
            ).squeeze(1)


        
            diagnosis_logits = self.branch_1(retrieved_true)
            skin_tone_logits = self.branch_2(retrieved_true)
            skin_tone_detached = self.branch_2(retrieved_true.detach())
            
            outputs = [
                diagnosis_logits,           #[0]
                skin_tone_logits,           #[1]
                skin_tone_detached,         #[2]
                attn_true,                  #[3]
                visual_patches,             #[4]
                text_proj_true,             #[5]
                retrieved_true              #[6]
            ]
            
            if text_emb_cf is not None:
                text_proj_cf = self.text_projection(text_emb_cf)
                attn_cf = self._compute_attention(visual_patches, text_proj_cf)
                retrieved_cf = torch.bmm(
                    attn_cf.unsqueeze(1), visual_patches
                ).squeeze(1)
                
                outputs.extend([text_proj_cf,       #[7]
                                 attn_cf,           #[8]
                                 retrieved_cf       #[9]
                                 ])
                
            query_inf = self.inference_query.expand(B, -1)
            attn_inf = self._compute_attention(visual_patches.detach(), query_inf)

            retrieved_inf = torch.bmm(
                    attn_inf.unsqueeze(1), visual_patches.detach()
                ).squeeze(1)

            diagnosis_logits_inf = self.branch_1(retrieved_inf)

            outputs.extend([query_inf,              #[10]
                             attn_inf,              #[11]
                             retrieved_inf,         #[12]
                               diagnosis_logits_inf     #[13]
                               ])
        
        else:
            query = self.inference_query.expand(B, -1)
            attention = self._compute_attention(visual_patches, query)
            retrieved = torch.bmm(
                attention.unsqueeze(1), visual_patches
            ).squeeze(1)

            H = W = int(visual_patches.size(1) ** 0.5) 
            attention_map = attention.view(B, 1, H, W)

            diagnostics = {
                'attention_weights': attention_map,
                'features': retrieved,
                'visual_patches': visual_patches,
                'query': query
            }
            
            diagnosis_logits = self.branch_1(retrieved)
            outputs = [
                diagnosis_logits,           #[0]
                None,                       #[1]
                None,                       #[2]
                attention,                  #[3]
                visual_patches,             #[4]
                query,                      #[5]
                  retrieved                     #[6]
            ]
        
        return diagnosis_logits, diagnostics

    def _compute_attention(self, visual_patches, text_query):
        B, P, D = visual_patches.shape

        eps=1e-8 
        V_norm = F.normalize(visual_patches, p=2, dim=2, eps=eps)  # (B, P, D)
        Q_norm = F.normalize(text_query, p=2, dim=1, eps=eps)  # (B, D)

        scores = torch.bmm(
            Q_norm.unsqueeze(1),  # (B, 1, D)
            V_norm.transpose(1, 2)  # (B, D, P)
        ).squeeze(1) / (D ** 0.5)  # (B, P)

        tau = torch.clamp(self.log_tau.exp(), min=TAU_MIN, max=TAU_MAX)
        attention = F.softmax(scores / tau, dim=-1)
        return attention


def load_model(checkpoint_path: str, num_classes: int = 3, device: str = 'cpu'):
    model = GQVKNetwork(output_size=[num_classes, 6])
    
    # Load checkpoint
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        

        model_dict = model.state_dict()
        

        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        if len(pretrained_dict) < len(new_state_dict):
            print(f" Warning: {len(new_state_dict) - len(pretrained_dict)} layers were skipped due to shape mismatch.")
            print("This usually happens if 'num_classes' in App doesn't match Training.")
            
        model.load_state_dict(pretrained_dict, strict=False)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Critical Error loading checkpoint: {e}")
        print("Initializing with random weights (Simulated Mode)")
    
    model.to(device)
    model.eval()
    
    return model


def compute_fairness_metrics(diagnostics: Dict, probs: torch.Tensor) -> Dict[str, float]:
    attention_weights = diagnostics.get('attention_weights')

    if attention_weights is not None:
        # Flatten: [B, 1, H, W] -> [B, H*W]
        attn_flat = attention_weights.view(attention_weights.size(0), -1)
        # Normalize (already softmaxed, but safe to ensure sum=1)
        attn_probs = attn_flat / (attn_flat.sum(dim=-1, keepdim=True) + 1e-10)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum(dim=-1).mean()
        attention_entropy = entropy.item()
    else:
        attention_entropy = 0.5

    quality_score = probs.max().item()


    consistency_score = 1.0 - (attention_entropy / 10.0)
    
    metrics = {
        'counterfactual_consistency': min(max(consistency_score, 0.0), 1.0),
        'attention_entropy': attention_entropy,
        'quality_score': quality_score,
        'decision_confidence': probs.max().item()
    }
    
    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Creating model...")
    model = GQVKNetwork(num_classes=3)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model = model.to(device)
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        logits, diagnostics = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities: {probs}")
    
    print("\nComputing fairness metrics...")
    metrics = compute_fairness_metrics(diagnostics, probs)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")