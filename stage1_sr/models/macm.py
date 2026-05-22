# stage1_sr/models/macm.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESA(nn.Module):
    """
    Enhanced Spatial Attention (ESA) module.
    
    PURPOSE: After extracting multi-scale features with convolutions, 
    we need to tell the network WHERE to focus — which spatial regions 
    are most important (edges, corners, fine textures). ESA does this 
    by creating a spatial attention map.
    
    HOW IT WORKS:
    1. Compress channels → create a spatial summary
    2. Apply pooling to capture spatial statistics
    3. Generate an attention mask (values 0–1 for each pixel)
    4. Multiply original features by the mask → highlighted important regions
    """
    def __init__(self, num_feat, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = num_feat // 4  # bottleneck channels
        self.conv1 = conv(num_feat, f, 1)
        self.conv_f = conv(f, f, 1)
        self.conv_max = conv(f, f, 3, padding=1)
        self.conv2 = conv(f, f, 3, stride=2, padding=0)
        self.conv3 = conv(f, f, 3, padding=1)
        self.conv3_ = conv(f, f, 3, padding=1)
        self.conv4 = conv(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m  # element-wise mask applied to input


#### Step 2.3 — Implement the Multi-Scale Adjustable Convolution Block (MACB)


class MACB(nn.Module):
    """
    Multi-Scale Adjustable Convolution Block (MACB).
    One building block inside MACM. Figure 4 in the paper.
    
    PURPOSE: Extract features at TWO different scales simultaneously.
    
    HOW IT WORKS (Equations 1 and 2):
    - Branch 1: DWConv 3×3, dilation=1 → captures FINE textures (small patterns)
    - Branch 2: DWConv 5×5, dilation=2 → captures BROAD structures (larger patterns)
    - Residual branch: passes input through to preserve original information
    - Concatenate all three → feed through ESA for spatial attention weighting
    
    "DWConv" = Depthwise Separable Convolution: processes each channel 
    independently, which is much more computationally efficient than 
    regular convolution.
    """
    def __init__(self, num_feat=64):
        super(MACB, self).__init__()
        # 1×1 conv to halve channels before branching (efficiency)
        self.split_conv = nn.Conv2d(num_feat, num_feat // 2, 1)
        
        # Branch 1: fine texture (small kernel, no dilation)
        self.dw_conv1 = nn.Sequential(
            nn.Conv2d(num_feat // 2, num_feat // 2, 3,
                      padding=1, dilation=1, groups=num_feat // 2),
            nn.Conv2d(num_feat // 2, num_feat // 2, 1)
        )
        # Branch 2: broader structure (larger kernel + dilation)
        self.dw_conv2 = nn.Sequential(
            nn.Conv2d(num_feat // 2, num_feat // 2, 5,
                      padding=4, dilation=2, groups=num_feat // 2),
            nn.Conv2d(num_feat // 2, num_feat // 2, 1)
        )
        # Residual branch: cross-scale supplement
        self.dw_res = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, padding=1, groups=num_feat),
            nn.Conv2d(num_feat, num_feat, 1)
        )
        
        # After concat of branch1+branch2+residual → project back to num_feat
        self.fuse_conv = nn.Conv2d(num_feat * 2, num_feat, 1)
        
        # Spatial attention to highlight important regions
        self.esa = ESA(num_feat)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Split input for the two parallel branches
        split = self.act(self.split_conv(x))
        
        # Equation 1: Fmulti_branch = Concat(DW1(Fin), DW2(Fin))
        branch1 = self.act(self.dw_conv1(split))
        branch2 = self.act(self.dw_conv2(split))
        multi_branch = torch.cat([branch1, branch2], dim=1)  # [B, C, H, W]
        
        # Equation 2: Fout = ESA(Concat(Fmulti_branch, DWres(Fin)))
        res = self.act(self.dw_res(x))
        combined = self.fuse_conv(torch.cat([multi_branch, res], dim=1))
        
        # ESA attention + residual skip
        out = self.esa(combined) + x
        return out

#### Step 2.4 — Implement the Full MACM (Multi-Scale Adjustable CNN Module)


class MACBG(nn.Module):
    """
    Multi-Scale Adjustable Convolution Block GROUP (MACBG).
    A group of MACB blocks connected with dense residual links.
    
    Dense residual links: the input of each MACB block is the 
    sum of ALL previous block outputs. This helps gradients flow 
    backward during training (avoids vanishing gradient problem).
    """
    def __init__(self, num_feat=64, num_block=4):
        super(MACBG, self).__init__()
        self.blocks = nn.ModuleList([MACB(num_feat) for _ in range(num_block)])
        self.fuse = nn.Conv2d(num_feat * (num_block + 1), num_feat, 1)

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            out = block(features[-1])
            features.append(out)
        # Dense connection: concatenate all feature maps
        dense_out = self.fuse(torch.cat(features, dim=1))
        return dense_out + x  # residual skip over the whole group


class MACM(nn.Module):
    """
    Full Multi-Scale Adaptive CNN Module (MACM).
    Equation 3: FMACM = Concat(Fout1, Fout2, ..., FoutN)
    
    Contains 4 MACBG groups (as stated in the paper's experimental setup).
    Output is used by ETM and HSPAM in subsequent stages.
    """
    def __init__(self, num_feat=64, num_group=4, num_block=4):
        super(MACM, self).__init__()
        self.groups = nn.ModuleList(
            [MACBG(num_feat, num_block) for _ in range(num_group)]
        )
        self.fuse = nn.Conv2d(num_feat * num_group, num_feat, 1)

    def forward(self, x):
        group_outputs = []
        feat = x
        for group in self.groups:
            feat = group(feat)
            group_outputs.append(feat)
        # Equation 3
        return self.fuse(torch.cat(group_outputs, dim=1))