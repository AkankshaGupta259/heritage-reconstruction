# stage1_sr/models/etm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EfficientMultiHeadAttention(nn.Module):
    """
    Efficient Multi-Head Attention (EMHA).
    
    PURPOSE: Standard self-attention is O(N²) in image size — too slow for 
    high-resolution images. This version operates along the CHANNEL dimension 
    instead of the spatial dimension, making it O(C²) which is much cheaper.
    
    The paper uses SCALED COSINE ATTENTION (not dot-product) to be robust 
    to repeated textures (common in cultural heritage objects like tiled 
    patterns). Cosine similarity is direction-based, not magnitude-based, 
    so it doesn't get confused by repeating intensities.
    
    Equations 5 and 6 in the paper.
    """
    def __init__(self, num_feat, num_heads=4):
        super().__init__()
        assert num_feat % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1) * math.log(10))
        
        self.qkv = nn.Linear(num_feat, num_feat * 3)
        self.proj = nn.Linear(num_feat, num_feat)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape: treat each channel as a "token"
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = [t.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                   for t in qkv]
        
        # Scaled cosine attention (robust to repeated textures)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        scale = torch.clamp(self.scale, max=math.log(100)).exp()
        attn = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, C)
        out = self.proj(out)
        return out.permute(0, 2, 1).view(B, C, H, W)


class LocalityExplorationBlock(nn.Module):
    """
    Locality Exploration Block (LEB) — used inside HSPAM.
    
    PURPOSE: Restore fine local details that the global attention 
    mechanism may have smoothed out. Acts as a local residual refinement.
    """
    def __init__(self, num_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class EfficientTransformerBlock(nn.Module):
    """
    One Efficient Transformer Block (ETB).
    Equation 5: Em = LN(EMHA(Ein)) + Ein
    Equation 6: Eout = LN(MLP(Em)) + Em
    
    Uses POST-LayerNorm (applied after attention, not before) for better 
    numerical stability with cultural heritage high-resolution textures.
    """
    def __init__(self, num_feat, mlp_ratio=4):
        super().__init__()
        self.attn = EfficientMultiHeadAttention(num_feat)
        self.norm1 = nn.LayerNorm(num_feat)
        
        mlp_hidden = num_feat * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(num_feat, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_feat)
        )
        self.norm2 = nn.LayerNorm(num_feat)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Equation 5: attention + post-LN + residual
        attn_out = self.attn(x)
        attn_flat = attn_out.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        x_flat = x.flatten(2).permute(0, 2, 1)
        em = self.norm1(attn_flat + x_flat)
        
        # Equation 6: MLP + post-LN + residual
        eout = self.norm2(self.mlp(em) + em)
        return eout.permute(0, 2, 1).view(B, C, H, W)


class ETM(nn.Module):
    """
    Full Efficient Transformer Module (ETM).
    Equation 4: FETM = φm(φm-1(...(φ1(FMACM)))) + FMACM
    
    Contains 1 Transformer block (as stated in paper's experimental setup).
    Adapts channel count: input 32 → output 64 via 3×3 conv, as described 
    in the paper's architecture details.
    """
    def __init__(self, in_feat=64, num_feat=64, num_block=1):
        super().__init__()
        self.channel_adapt = nn.Conv2d(in_feat, num_feat, 3, padding=1)
        self.blocks = nn.ModuleList(
            [EfficientTransformerBlock(num_feat) for _ in range(num_block)]
        )

    def forward(self, x_macm):
        # Equation 4
        feat = self.channel_adapt(x_macm)
        for block in self.blocks:
            feat = block(feat)
        return feat + x_macm  # skip connection over entire ETM