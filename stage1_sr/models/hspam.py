# stage1_sr/models/hspam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1_sr.models.etm import LocalityExplorationBlock


class HighSimilarityPassAttention(nn.Module):
    """
    High-Similarity-Pass Attention (HSPA).
    
    PURPOSE: Standard softmax attention gives non-zero weight to ALL 
    positions, even ones with very low similarity to the query. This 
    "dilutes" the attention signal with irrelevant noise — a serious 
    problem for cultural heritage images with aged/worn textures where 
    many patches look superficially similar but shouldn't be matched.
    
    SOLUTION: Apply a SOFT THRESHOLD — truncate attention weights below 
    a learned threshold to exactly zero. This creates a SPARSE attention 
    distribution that only propagates information from genuinely 
    high-similarity locations. 
    
    Think of it as: standard attention = listening to everyone in a crowd;
    HSPA = only listening to people who are clearly talking to you.
    """
    def __init__(self, num_feat, threshold=0.5):
        super().__init__()
        self.num_feat = num_feat
        self.query_conv = nn.Conv2d(num_feat, num_feat // 8, 1)
        self.key_conv   = nn.Conv2d(num_feat, num_feat // 8, 1)
        self.value_conv = nn.Conv2d(num_feat, num_feat, 1)
        self.out_conv   = nn.Conv2d(num_feat, num_feat, 1)
        
        # Learnable threshold — starts at 0.5, adjusts during training
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.gamma = nn.Parameter(torch.zeros(1))  # blend weight

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # number of spatial positions
        
        q = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # [B, N, C//8]
        k = self.key_conv(x).view(B, -1, N)                       # [B, C//8, N]
        v = self.value_conv(x).view(B, C, N)                      # [B, C, N]
        
        # Cosine similarity matrix [B, N, N]
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k.permute(0, 2, 1), dim=-1)
        similarity = torch.bmm(q_norm, k_norm.permute(0, 2, 1))
        
        # Soft threshold: zero out low-similarity connections
        # This is the key innovation — sparse attention
        mask = (similarity > self.threshold).float()
        similarity = similarity * mask
        
        # Normalize (avoid division by zero)
        row_sum = similarity.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = similarity / row_sum
        
        # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, N]
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        
        # Blend with input (gamma starts at 0, gradually increases during training)
        return self.gamma * out + x


class HSPAM(nn.Module):
    """
    Full High-Similarity-Pass Attention Module (HSPAM).
    
    Equations 7 and 8:
    - Equation 7: FHSPAM = HSPAM(FMACM)
    - Equation 8: Ffuse = FETM + α × FHSPAM
    
    Structure: HSPA branch (global, sparse attention) + LEB (local detail 
    restoration) fused via residual connection.
    """
    def __init__(self, num_feat=64):
        super().__init__()
        self.hspa = HighSimilarityPassAttention(num_feat)
        self.leb = LocalityExplorationBlock(num_feat)
        self.fuse = nn.Conv2d(num_feat * 2, num_feat, 1)

    def forward(self, x_macm):
        # Equation 7
        hspa_out = self.hspa(x_macm)    # global sparse attention
        leb_out  = self.leb(x_macm)     # local detail restoration
        # Fuse both branches
        fused = self.fuse(torch.cat([hspa_out, leb_out], dim=1))
        return fused + x_macm  # residual skip