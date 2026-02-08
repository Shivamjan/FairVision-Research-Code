"""
new_loss.py
GQVK with Pre-computed Text Embeddings (Gemma-300M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import numpy as np
import json 


# Configurations
DEFAULT_TAU = 0.07
TAU_MIN = 0.03
TAU_MAX = 0.2

class GQVKNetwork(nn.Module):
    """
    GQVK Architecture using pre-computed Gemma embeddings
    """

    def __init__(self,
                 output_size=[9, 6],  # [num_diseases, num_skin_tones] #output, size not confirmed yet
                 gemma_dim=768,  # Gemma-300M embedding dimension
                 pretrained=True):
        super(GQVKNetwork, self).__init__()

        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

        # ============================================
        # Image Encoder (Visual Keys/Values)
        # ============================================
        self.image_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k'
        )
        self.image_dim = self.image_encoder.config.hidden_size  # 768
        #self.a = 197 #The number of tokens (196 image patches + 1 [CLS] token)

        # ============================================
        #  Text Projection (Gemma -> ViT dimension)
        # ============================================
        # Gemma embeddings might have different dimension than ViT
        # Project to match ViT's 768-dim space
        self.text_projection = nn.Sequential(
            nn.Linear(gemma_dim, self.image_dim),
            nn.LayerNorm(self.image_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.image_dim, self.image_dim)
        )

        # NEW: Learnable query for inference
        self.inference_query = nn.Parameter(
            torch.randn(1, self.image_dim)
        )
        # ============================================
        # Classification Heads
        # ============================================
        # Main diagnosis classifier
        self.branch_1 = nn.Sequential(
            nn.Linear(self.image_dim, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, output_size[0]),
        )

        # Skin-tone classifier
        self.branch_2 = nn.Linear(self.image_dim, output_size[1])

        # ============================================
        # Contrastive Projection
        # ============================================
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
        
        # TRAINING: Use text queries
        if self.training and text_emb_true is not None:
            text_proj_true = self.text_projection(text_emb_true)
            attn_true = self._compute_attention(visual_patches, text_proj_true)
            retrieved_true = torch.bmm(
                attn_true.unsqueeze(1), visual_patches
            ).squeeze(1)


            
            # Classification uses text-based features
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
            
            # CF for fairness
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
            
            # Also compute inference query (for learning)
            query_inf = self.inference_query.expand(B, -1)
            attn_inf = self._compute_attention(visual_patches.detach(), query_inf)

            # For the retrieval, we can use the attached patches if we want the
            # classifier to learn, but usually, just detaching attention is enough.
            # However, since we are only using attn_inf for the loss, this is safe.
            retrieved_inf = torch.bmm(
                    attn_inf.unsqueeze(1), visual_patches.detach()
                ).squeeze(1)

            diagnosis_logits_inf = self.branch_1(retrieved_inf)

            outputs.extend([query_inf,              #[10]
                             attn_inf,              #[11]
                             retrieved_inf,         #[12]
                               diagnosis_logits_inf     #[13]
                               ])
        
        # INFERENCE: Use learned query
        else:
            query = self.inference_query.expand(B, -1)
            attention = self._compute_attention(visual_patches, query)
            retrieved = torch.bmm(
                attention.unsqueeze(1), visual_patches
            ).squeeze(1)
            
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
        
        return outputs

    def _compute_attention(self, visual_patches, text_query):
        """
        Compute scaled dot-product attention

        Args:
            visual_patches: (B, P, D) - Visual keys/values
            text_query: (B, D) - Text query

        Returns:
            attention: (B, P) - Attention weights
        """
        B, P, D = visual_patches.shape

        eps=1e-8 #stability factor

        # Normalize embeddings
        V_norm = F.normalize(visual_patches, p=2, dim=2, eps=eps)  # (B, P, D)
        Q_norm = F.normalize(text_query, p=2, dim=1, eps=eps)  # (B, D)

        # Compute scores: <Q, K^T>
        scores = torch.bmm(
            Q_norm.unsqueeze(1),  # (B, 1, D)
            V_norm.transpose(1, 2)  # (B, D, P)
        ).squeeze(1) / (D ** 0.5)  # (B, P)

        # Apply temperature and softmax
        tau = torch.clamp(self.log_tau.exp(), min=TAU_MIN, max=TAU_MAX)
        attention = F.softmax(scores / tau, dim=-1)
        return attention


#  ============================================
# AC-CQC Loss (Compatible with Pre-computed)
# ============================================
def ac_cqc_loss(model_outputs,
                lambda_fair=1.0,
                metric='l1',
                include_contrastive=True,
                device='cuda'):
    """
    AC-CQC loss for dual-path architecture
    
    This loss operates ONLY on the Text Path to enforce fairness
    in the backbone's visual representations.
    
    Args:
        model_outputs: List from model.forward()
        lambda_fair: Fairness loss weight
        metric: 'l1', 'cosine', or 'js_divergence'
        include_contrastive: Whether to add contrastive loss
        device: Device for tensors

    Returns:
        total_loss, loss_dict
    """
    # Unpack outputs

    # These come from the text-guided attention (fairness path)
    attn_text = model_outputs[3]        # (B, 196) - Text attention
    text_proj_true = model_outputs[5]   # (B, 768) - Projected text (true)
    retrieved_text = model_outputs[6]   # (B, 768) - Retrieved features (true)

    # Check if CF was computed
    if len(model_outputs) <= 7:
        raise ValueError(
            "Model must be called with text_emb_cf for fairness loss. "
            "Ensure you're in training mode and passing both text_emb_true and text_emb_cf."
        )
  
    text_proj_cf = model_outputs[7]     # (B, 768) - Projected text (CF)
    attn_cf = model_outputs[8]          # (B, 196) - CF attention
    retrieved_cf = model_outputs[9]     # (B, 768) - Retrieved features (CF)

    attn_inf = model_outputs[11]
    
    B = attn_text.size(0)
    eps = 1e-8

    # ============================================
    # 1. Fairness Loss (Attention Consistency)
    # ============================================
    if metric == 'l1':
        #  Asymmetric: Push CF attention toward true attention
        # This prevents the "true" path from being affected by fairness constraint
        loss_fairness = (attn_text.detach() - attn_cf).abs().mean()
        
        # Alternative symmetric version (both paths contribute):
        # loss_fairness = (attn_text - attn_cf).abs().mean()


    elif metric == 'cosine':
        cos_sim = F.cosine_similarity(attn_text, attn_cf, dim=1)
        loss_fairness = (1 - cos_sim).mean()


    elif metric == 'js_divergence':
        # Jensen-Shannon is more robust for probability distribution
        m = 0.5 * (attn_text + attn_cf)
        # F.kl_div requires Log-Probs as input
        log_m = torch.log(m + eps)
        # KL(P || M)
        kl_true = F.kl_div(
            log_m, attn_text,
            reduction='batchmean', log_target=False
        )
        # KL(Q || M)
        kl_cf = F.kl_div(
            log_m, attn_cf,
            reduction='batchmean', log_target=False
        )
        loss_fairness = 0.5 * (kl_true + kl_cf)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    #==================
    # Contrastive Loss
    # ======================
    device = retrieved_text.device
    loss_contrastive = torch.zeros((), device=device)

    if include_contrastive:
        # Normalize
        V_true_norm = F.normalize(retrieved_text, p=2, dim=1)
        V_cf_norm = F.normalize(retrieved_cf, p=2, dim=1)
        T_true_norm = F.normalize(text_proj_true, p=2, dim=1)
        T_cf_norm = F.normalize(text_proj_cf, p=2, dim=1)

        # Compute similarity matrices
        logits_true = torch.matmul(V_true_norm, T_true_norm.T) / 0.07

         # NOTE: CF logits are intentionally NOT used for contrastive loss.
        # CF text is counterfactual and should not be enforced for semantic alignment. 
        logits_cf = torch.matmul(V_cf_norm, T_cf_norm.T) / 0.07

        # Targets (diagonal)
        targets = torch.arange(B, device=device)

       
        loss_contrastive = F.cross_entropy(logits_true, targets)

        # ============================================
        # L_distill: "Teach the Student" (NEW)
        # ============================================
        # Force the Inference Query (Student) to look at the same thing
        # as the Factual Text Query (Teacher).
        # We detach attn_text_true so we don't degrade the text path
        # to match a bad inference path.
    loss_distill = (attn_text.detach() - attn_inf).abs().mean()

    # ============================================
    # Total Loss
    # ============================================
    total_loss = (lambda_fair * loss_fairness) + loss_contrastive + (0.1 * loss_distill)      #write as explicit parameter

    loss_dict = {
        'fairness': loss_fairness.item(),
        'contrastive': loss_contrastive.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_dict


# ============================================
# Helper: Confusion Loss
# ============================================
class Confusion_Loss(torch.nn.Module):
    '''
    Confusion loss built based on the paper 'Invesgating bias and fairness.....' 
    (https://www.repository.cam.ac.uk/bitstream/handle/1810/309834/XuEtAl-ECCV2020W.pdf?sequence=1&isAllowed=y)
    '''
    def __init__(self):
        super(Confusion_Loss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, output, label):
        # output (bs, out_size). label (bs)
        prediction = self.softmax(output) # (bs, out_size)
        log_prediction = torch.log(prediction + 1e-8)
        loss = -torch.mean(torch.mean(log_prediction, dim=1), dim=0)

        # loss = torch.mean(torch.mean(prediction*log_prediction, dim=1), dim=0)
        return loss
