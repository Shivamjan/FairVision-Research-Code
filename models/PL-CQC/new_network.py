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






class GQVKNetwork(nn.Module):
    """
    GQVK Architecture using pre-computed Gemma embeddings
    """

    def __init__(self,
                 output_size=[9, 6],  # [num_diseases, num_skin_tones] #output, size not confirmed yet
                 gemma_dim=768,  # Gemma-300M embedding dimension
                 pretrained=True):
        super(GQVKNetwork, self).__init__()

        self.num_diseases = output_size[0]      # 9
        self.num_skin_tones = output_size[1]    # 6

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
        # NEW: PATCH-LEVEL LOGIT DECOMPOSITION
        # ============================================
        
        # Step 1: Patch-level disease evidence heads
        # Each patch gets its own disease specific feature
        self.patch_disease_encoder = nn.Sequential(
            nn.Linear(self.image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Patch-level disease logits: l_p^(d)
        self.patch_logit_head = nn.Linear(128, self.num_diseases)
        
        # Step 2: Patch importance network
        # Learns alpha_p^(d) conditioned on query
        self.importance_network = nn.Sequential(
            nn.Linear(self.image_dim * 2, 256),  # [patch_feat; query]
            nn.ReLU(),
            nn.Linear(256, self.num_diseases),
            # nn.Softmax(dim=1)  # Normalize across patches
        )
        
        # ============================================
        # 4. Classification Heads
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
        # 5.Contrastive Projection
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

            # NEW: Compute patch-level logit decomposition
            patch_logits_true, patch_importance_true, diagnosis_logits_lgcqc = \
                self._compute_patch_logits(visual_patches, text_proj_true)
            

            attn_true = self._compute_attention(visual_patches, text_proj_true)
            retrieved_true = torch.bmm(
                attn_true.unsqueeze(1), visual_patches
            ).squeeze(1)
            
            # Classification uses text-based features
            diagnosis_logits_standard = self.branch_1(retrieved_true)
            skin_tone_logits = self.branch_2(retrieved_true)
            skin_tone_detached = self.branch_2(retrieved_true.detach())
            

            outputs = [
                diagnosis_logits_lgcqc,     #[0] NEW: LG-CQC logits
                skin_tone_logits,           #[1]
                skin_tone_detached,         #[2]
                attn_true,                  #[3]
                visual_patches,             #[4]
                text_proj_true,             #[5]
                retrieved_true,             #[6]
                patch_logits_true,          #[7] NEW
                patch_importance_true,      #[8] NEW
            ]
            
            # CF for fairness
            if text_emb_cf is not None:


                text_proj_cf = self.text_projection(text_emb_cf)

                # NEW: CF patch logits---
                patch_logits_cf, patch_importance_cf, _ = \
                    self._compute_patch_logits(visual_patches, text_proj_cf)
                
                attn_cf = self._compute_attention(visual_patches, text_proj_cf)
                retrieved_cf = torch.bmm(
                    attn_cf.unsqueeze(1), visual_patches
                ).squeeze(1)
                
                outputs.extend([
                    text_proj_cf,           #[9]
                    attn_cf,                #[10]
                    retrieved_cf,           #[11]
                    patch_logits_cf,        #[12] NEW
                    patch_importance_cf,    #[13] NEW
                ])
            
            # Also compute inference query (for learning)
            query_inf = self.inference_query.expand(B, -1)

            patch_logits_inf, patch_importance_inf, diagnosis_logits_inf = \
                self._compute_patch_logits(visual_patches, query_inf)
            
            attn_inf = self._compute_attention(visual_patches, query_inf)
            retrieved_inf = torch.bmm(
                attn_inf.unsqueeze(1), visual_patches
            ).squeeze(1)

            diagnosis_logits_inf = self.branch_1(retrieved_inf)

            outputs.extend([
                query_inf,                  #[14]
                attn_inf,                   #[15]
                retrieved_inf,              #[16]
                diagnosis_logits_inf,       #[17]
                patch_logits_inf,           #[18] NEW
                patch_importance_inf,       #[19] NEW
            ])
            
            
        
        # INFERENCE: Use learned query
        else:
            query = self.inference_query.expand(B, -1)

            # NEW::: Use LG-CQC at inference
            patch_logits, patch_importance, diagnosis_logits = \
                self._compute_patch_logits(visual_patches, query)
            
            attention = self._compute_attention(visual_patches, query)
            retrieved = torch.bmm(
                attention.unsqueeze(1), visual_patches
            ).squeeze(1)
            
            outputs = [
                diagnosis_logits,           #[0] LG-CQC logits
                None,                       #[1]
                None,                       #[2]
                attention,                  #[3]
                visual_patches,             #[4]
                query,                      #[5]
                retrieved,                  #[6]
                patch_logits,               #[7] NEW
                patch_importance,           #[8] NEW
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

        # Normalize embeddings
        V_norm = F.normalize(visual_patches, p=2, dim=2)  # (B, P, D)
        Q_norm = F.normalize(text_query, p=2, dim=1)  # (B, D)

        # Compute scores: <Q, K^T>
        scores = torch.bmm(
            Q_norm.unsqueeze(1),  # (B, 1, D)
            V_norm.transpose(1, 2)  # (B, D, P)
        ).squeeze(1) / (D ** 0.5)  # (B, P)

        # Apply temperature and softmax
        tau = torch.clamp(self.log_tau.exp(), min=0.03, max=0.2)
        attention = F.softmax(scores / tau, dim=-1)
        return attention
    

    def _compute_patch_logits(self, visual_patches, query):
        """
        NEW: Compute patch-level disease contributions
        
        Args:
            visual_patches: (B, P, D)
            query: (B, D)
        
        Returns:
            patch_logits: (B, P, C) - Disease evidence per patch
            patch_importance: (B, P, C) - Importance weights α_p^(d)
            final_logits: (B, C) - Aggregated disease logits
        """
        B, P, D = visual_patches.shape
        C = self.num_diseases
        
        # Step 1: Compute patch-level disease evidence l_p^(d)
        patch_features = self.patch_disease_encoder(visual_patches)  # (B, P, 128)
        patch_logits = self.patch_logit_head(patch_features)  # (B, P, C)
        
        # Step 2: Compute patch importance alpha_p^(d)
        # Condition on both patch and query
        query_expanded = query.unsqueeze(1).expand(-1, P, -1)  # (B, P, D)
        combined = torch.cat([visual_patches, query_expanded], dim=-1)  # (B, P, 2D)
        
        # Importance per disease per patch
        importance_logits = self.importance_network(
            combined.view(B * P, -1)
        ).view(B, P, C)  # (B, P, C)
        
        # Normalize across patches for each disease
        patch_importance = F.softmax(importance_logits, dim=1)  # (B, P, C)
        
        # Step 3: Aggregate: z_d = Sum over p: alpha_p^(d) · l_p^(d)
        final_logits = torch.sum(
            patch_importance * patch_logits, dim=1
        )  # (B, C)
        
        return patch_logits, patch_importance, final_logits




def lg_cqc_loss_v2(model_outputs,
                   targets,          # NEW: Requires ground truth labels (B,) or (B, C)
                   lambda_fair=1.0,
                   lambda_contrastive=1.0,
                   confidence_threshold=0.5, # For gating
                   device='cuda'):
    """
    Logit-Grounded CQC Loss (Version 2: Conditional & Relative)
    
    Fixes:
    1. Disease-Conditional: Only normalizes/enforces loss on the active disease class.
    2. Confidence-Gated: Only enforces fairness if model is confident.

    Args:
        model_outputs: List from model.forward()
        targets: Ground truth disease labels (B,) - integers in [0, C-1]
        lambda_fair: Weight for fairness loss
        lambda_contrastive: Weight for contrastive loss
        confidence_threshold: Minimum confidence to enforce fairness (0.5-0.7 recommended)
        device: Device
    
    Returns:
        total_loss, loss_dict
    """
    
    if len(model_outputs) < 14:
        raise ValueError("Model outputs missing CF terms.")
        
    # Unpack
    diagnosis_logits = model_outputs[0]       # (B, C)
    patch_logits_true = model_outputs[7]      # (B, P, C)
    patch_importance_true = model_outputs[8]  # (B, P, C)
    patch_logits_cf = model_outputs[12]
    patch_importance_cf = model_outputs[13]
    
    # Aux
    attn_true = model_outputs[3]
    attn_cf = model_outputs[10]
    text_proj_true = model_outputs[5]
    text_proj_cf = model_outputs[9]
    retrieved_true = model_outputs[6]
    retrieved_cf = model_outputs[11]
    
    B, P, C = patch_logits_true.shape
    
    # =========================================================
    # PART 1: TARGET SELECTION (The "Disease-Conditional" Fix)
    # =========================================================
    
    # Compute raw contributions for all classes first: (B, P, C)
    contrib_true_all = patch_importance_true * patch_logits_true
    contrib_cf_all = patch_importance_cf * patch_logits_cf
    
    # We need to gather only the contributions for the target class y.
    # Assuming targets is (B,) containing the index of the disease.
    # If targets is one-hot (B, C), use targets.argmax(dim=1).
    if targets.dim() > 1:
        target_indices = targets.argmax(dim=1)
    else:
        target_indices = targets
        
    # Shape matching for gather: (B, 1, 1) -> expanded to (B, P, 1)
    target_idx_expanded = target_indices.view(B, 1, 1).expand(-1, P, 1)
    
    # Gather: Select only the 'slices' of the tensor corresponding to the true disease
    # Result: (B, P, 1)
    c_true_y = torch.gather(contrib_true_all, 2, target_idx_expanded)
    c_cf_y   = torch.gather(contrib_cf_all,   2, target_idx_expanded)
    
    # =========================================================
    # PART 2: RELATIVE NORMALIZATION (On Ground Truth Only)
    # =========================================================
    eps = 1e-6
    
    # Normalize per image, only using the mass of the relevant disease
    # Sum over Patch dimension (dim=1).... dim=2 is size 1 now.
    norm_true = c_true_y.abs().sum(dim=1, keepdim=True) + eps       # (B, 1, 1)
    norm_cf   = c_cf_y.abs().sum(dim=1, keepdim=True) + eps        # (B, 1, 1)
    
    # Relative Distributions 
    rel_true = c_true_y / norm_true       # (B, P, 1)
    rel_cf   = c_cf_y   / norm_cf      # (B, P, 1)
    
    # Calculate raw distance per sample (B,)
    # Sum over patches, squeeze class dim
    dist_per_sample = (rel_true - rel_cf).abs().sum(dim=1).squeeze(1) 
    
    # =========================================================
    # PART 3: CONFIDENCE GATING (Correct & Defensible)
    # =========================================================

    # logits: (B, C) from classifier head
    # y_true: (B,) ground-truth disease indices

    raw_logit_y = diagnosis_logits.gather(1, target_indices.view(-1, 1)).squeeze()  # (B,)
    
    # Compute confidence (sigmoid of target logit)
    confidence = torch.sigmoid(raw_logit_y)  # (B,)


    # Create gate (1 if confident, 0 otherwise)
    if confidence_threshold > 0:
        #gate = (confidence > confidence_threshold).float().detach()     #Hard gating
        gate = torch.clamp(
                        (confidence - confidence_threshold) / (1 - confidence_threshold),
                        min=0.0
                    ).detach()          #soft gate
    else:
        gate = torch.ones_like(confidence)
   
    avg_gate_weight = gate.mean()
    num_gated = gate.sum()

    if num_gated > 0:
        loss_plc = (dist_per_sample * gate).sum() / num_gated
    else:
        # Fallback: if no samples are confident, use small penalty
        loss_plc = torch.tensor(0.0, device=device)
    
    # ==============================
    # Aux Losses
    # ==============================
    loss_attn = (attn_true.detach() - attn_cf).abs().mean()
    
    V_true_norm = F.normalize(retrieved_true, p=2, dim=1)
    T_true_norm = F.normalize(text_proj_true, p=2, dim=1)
    
    logits_contrastive = torch.matmul(V_true_norm, T_true_norm.T) / 0.07
    targets_contrastive = torch.arange(B, device=device)
    # loss_contrastive = F.cross_entropy(global_logits, targets_contrastive)
    loss_contrastive = torch.tensor(0.0, device=device)
    
    total_loss = (
        lambda_fair * loss_plc + 
        0.1 * loss_attn + 
        lambda_contrastive * loss_contrastive
    )
    
    loss_dict = {
        'lg_cqc_conditional': loss_plc.item(),
        'contrastive': loss_contrastive.item(),
        'total': total_loss.item(),
        'avg_confidence': confidence.mean().item(),      # NEW: Monitor
        'num_gated_samples': num_gated.item(),           # NEW: Track
        'gate_ratio': (num_gated / B).item(),            # NEW: Percentage
        'avg_gate_weight': avg_gate_weight.item(),
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
