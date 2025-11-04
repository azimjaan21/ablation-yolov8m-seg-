# ultralytics/ultralytics/change_model/novel_blocks.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f, Conv

from ultralytics.nn.modules.block import Bottleneck

# ---------- small utils ----------
class ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, bn=True):
        super().__init__()
        if p is None: p = k // 2
        self.add_module("conv", nn.Conv2d(c1, c2, k, s, p, groups=g, bias=not bn))
        if bn: self.add_module("bn", nn.BatchNorm2d(c2))
        if act: self.add_module("act", nn.SiLU(inplace=True))

# ---------- ECA (channel attention) ----------
class ECA(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg(x)                     # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(1,2))  # B,1,C -> temporal conv
        y = self.sigmoid(y.transpose(1,2).unsqueeze(-1))  # B,C,1,1
        return x * y

# ---------- Edge-aware spatial attention (eCBAM-S) ----------
import torch
import torch.nn as nn


class EdgeSpatialAttention(nn.Module):
    def __init__(self, k=7):
        """
        Edge Spatial Attention Module.
        Args:
            k (int): kernel size for spatial attention (default=7).
        """
        super().__init__()
        # Convolution over concatenated [avg_pool, edge_map]
        self.conv = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)

        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)

        # Edge map = difference between average and max
        edge_map = avg_out - max_out  # (B,1,H,W)

        # Concatenate average + edge → (B,2,H,W)
        out = torch.cat([avg_out, edge_map], dim=1)

        # Apply conv + sigmoid attention
        out = self.conv(out)
        attn = self.sigmoid(out)

        # Refine input with attention
        return x * attn


# ---------- eCBAM (channel + edge-aware spatial) ----------
class eCBAM(nn.Module):
    def __init__(self, c, eca_k=3, spa_k=7):
        super().__init__()
        self.ca = ECA(c, k=eca_k)
        self.sa = EdgeSpatialAttention(k=spa_k)
    def forward(self, x):
        return self.sa(self.ca(x))

# ---------- C2f-eCBAM: swap inner blocks with attention-enhanced residuals ----------
class C2f_eCBAM(C2f):
    """
    C2f with edge-aware CBAM on each inner block’s output.
    Very lightweight; great for segmentation boundaries.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, eca_k=3, spa_k=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attn = nn.ModuleList(eCBAM(self.c, eca_k=eca_k, spa_k=spa_k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for i, m in enumerate(self.m):
            yi = m(y[-1])                 # standard C2f inner bottleneck
            yi = self.attn[i](yi)         # add eCBAM
            y.append(yi)
        return self.cv2(torch.cat(y, 1))

###########################################   LKA   ######################################################


# ultralytics/nn/modules/lka_yolo.py

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class LargeKernelAttention(nn.Module):
    """Large Kernel Attention from LKA‑YOLO (Wang et al. 2024)."""
    def __init__(self, c, k1=17, k2=3, dil=3):
        super().__init__()
        # large depthwise conv + small dilated conv + 1×1 conv
        self.dw_large = nn.Conv2d(c, c, k1, padding=k1 // 2, groups=c, bias=False)
        self.dw_dil = nn.Conv2d(c, c, k2, padding=dil, dilation=dil, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.dw_large(x)
        y = self.dw_dil(y)
        y = self.bn(self.pw(y))
        return x * self.act(y)


class ResVANBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, *args, **kwargs):
        super().__init__()
        self.conv_in = Conv(c1, c2, 1, 1, act=True)
        self.lka = LargeKernelAttention(c2)
        self.conv_out = Conv(c2, c2, 3, 1, act=True)
        self.shortcut = shortcut and (c1 == c2)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv_in(x)
        y = self.lka(y)
        y = self.conv_out(y)
        if self.shortcut:
            y = y + x
        return self.act(y)



########################################### LKA LITE ####################################################

class LKALite(nn.Module):
    """
    Large-Kernel Attention (lite):
      DW-3x3 -> DW-7x7 (dilated) -> 1x1 -> sigmoid gate
    """
    def __init__(self, c, dwk=3, dil_k=7, dil=3):
        super().__init__()
        self.dw1 = ConvBNAct(c, c, k=dwk, g=c)                 # DW 3x3
        self.dw2 = nn.Conv2d(c, c, kernel_size=dil_k, padding=dil, dilation=dil, groups=c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.pw  = nn.Conv2d(c, c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.dw1(x)
        H, W = y.shape[2], y.shape[3]
        eff_ksize = self.dw2.kernel_size[0] + (self.dw2.kernel_size[0]-1) * (self.dw2.dilation[0]-1)
        if H < eff_ksize or W < eff_ksize:
            # Fallback to a 3x3 depthwise if too small
            fallback_dw = nn.Conv2d(self.dw2.in_channels, self.dw2.out_channels, 3, padding=1, groups=self.dw2.groups, bias=False).to(x.device)
            y = self.bn2(fallback_dw(y))
        else:
            y = self.bn2(self.dw2(y))
        y = self.bn3(self.pw(y))
        y = self.sig(y)
        if x.shape[2:] != y.shape[2:]:
            y = nn.functional.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x * y



class C2f_LKALite(C2f):
    """
    C2f with LKA-lite gates on inner outputs.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, dwk=3, dil_k=7, dil=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.lka = nn.ModuleList(LKALite(self.c, dwk=dwk, dil_k=dil_k, dil=dil) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for i, m in enumerate(self.m):
            yi = m(y[-1])
            yi = self.lka[i](yi)
            y.append(yi)
        return self.cv2(torch.cat(y, 1))
    




    ############## #1-Idea Novelty MultiBranch #############

import torch
import torch.nn as nn

class MultiBranchLKALite(nn.Module):
    """
    MultiBranchLKALite: Parallel lightweight large-kernel attention 
    with learnable branch weights and pointwise fusion.
    """
    def __init__(self, c, k_list=(5, 7, 9)):
        super().__init__()
        self.num_branches = len(k_list)
        self.branches = nn.ModuleList()
        for k in k_list:
            # Each branch: Depthwise 1xk then kx1 convolution
            branch = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, k//2), groups=c, bias=False),
                nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k//2, 0), groups=c, bias=False),
                nn.BatchNorm2d(c),
            )
            self.branches.append(branch)
        # Learnable branch weights (softmaxed in forward)
        self.branch_weights = nn.Parameter(torch.ones(self.num_branches))
        # 1x1 fusion conv after concatenation
        self.fuse_conv = nn.Conv2d(c*self.num_branches, c, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(c)
        # Final sigmoid gating
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Each branch processes input x
        branch_outputs = [branch(x) for branch in self.branches]   # List of [B, C, H, W]
        # Concatenate along channel axis
        concat = torch.cat(branch_outputs, dim=1)                  # [B, C*num_branches, H, W]
        fused = self.fuse_bn(self.fuse_conv(concat))               # [B, C, H, W]
        # Learnable fusion through softmaxed branch weights (optional, for weighted sum)
        weights = torch.softmax(self.branch_weights, dim=0)        # [num_branches]
        # Weighted sum (optional - comment if using concat+1x1 fusion only)
        # weighted = sum(w * b for w, b in zip(weights, branch_outputs))
        gate = self.sigmoid(fused)                                 # [B, C, H, W]
        # Elementwise gating on input
        out = x * gate
        return out


class C2f_MultiBranchLKALite(C2f):
    """
    C2f with MultiBranchLKALite gates on inner outputs.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k_list=(5,7,9)):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.lka = nn.ModuleList([MultiBranchLKALite(self.c, k_list) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for i, m in enumerate(self.m):
            yi = m(y[-1])
            yi = self.lka[i](yi)
            y.append(yi)
        return self.cv2(torch.cat(y, 1))


    ############## #2-Idea Novelty PAG #############

class ProgressiveLKALite(nn.Module):
    """Enhanced LKA with Progressive Attention Guidance"""
    def __init__(self, c, stages=3, dwk=3, dil_k=7, dil=3):
        super().__init__()
        self.stages = stages
        
        # Multi-stage attention refinement
        self.attention_stages = nn.ModuleList([
            LKALite(c, dwk=dwk, dil_k=dil_k, dil=dil) 
            for _ in range(stages)
        ])
        
        # Progressive fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(stages))
        
    def forward(self, x):
        refined_features = []
        current_x = x
        
        for i, lka in enumerate(self.attention_stages):
            att_x = lka(current_x)
            refined_features.append(att_x)
            current_x = att_x  # Feed refined features to next stage
            
        # Weighted fusion of multi-stage features
        weights = torch.softmax(self.fusion_weights, dim=0)
        output = sum(w * feat for w, feat in zip(weights, refined_features))
        
        return output

class C2f_ProgressiveLKALite(C2f):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, stages=3, dwk=3, dil_k=7, dil=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.lka = nn.ModuleList([ProgressiveLKALite(self.c, stages=stages, dwk=dwk, dil_k=dil_k, dil=dil) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for i, m in enumerate(self.m):
            yi = m(y[-1])
            yi = self.lka[i](yi)
            y.append(yi)
        return self.cv2(torch.cat(y, 1))



import torch
import torch.nn as nn

# ============================================================================
# FREQUENCY DECOMPOSITION MODULE  #3-IDEA
# ============================================================================
class FrequencyDecomposition(nn.Module):
    """Decomposes features into low-frequency and high-frequency components"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.low_pass = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size,
            padding=kernel_size//2, 
            groups=channels,
            bias=False
        )
        nn.init.constant_(self.low_pass.weight, 1.0 / (kernel_size * kernel_size))
        self.low_pass.weight.requires_grad = True
        
    def forward(self, x):
        x_low = self.low_pass(x)
        x_high = x - x_low
        return x_low, x_high


# ============================================================================
# FAD-LKALITE ATTENTION MODULE
# ============================================================================
class FrequencyAdaptiveLKALite(nn.Module):
    """Frequency-Adaptive Dynamic LKALite for segmentation"""
    def __init__(self, channels, dwk=3, dil_k=7, dil=3):
        super().__init__()
        
        # Standard LKALite components
        self.dw_conv = nn.Conv2d(channels, channels, dwk, 1, dwk//2, groups=channels)
        self.dw_d_conv = nn.Conv2d(channels, channels, dil_k, 1, 
                                     (dil_k//2)*dil, dilation=dil, groups=channels)
        self.pw_conv = nn.Conv2d(channels, channels, 1)
        
        # Frequency decomposition
        self.freq_decomp = FrequencyDecomposition(channels, kernel_size=3)
        
        # Adaptive frequency weight generator
        self.freq_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        # Spatial attention pathway
        attn = self.dw_conv(x)
        attn = self.dw_d_conv(attn)
        
        # Frequency decomposition
        low_freq, high_freq = self.freq_decomp(attn)
        
        # Generate adaptive frequency weights
        freq_weight = self.freq_adapter(x)
        
        # Adaptive frequency fusion
        attn_freq = freq_weight * low_freq + (1 - freq_weight) * high_freq
        
        # Channel mixing
        attn = self.pw_conv(attn_freq)
        attn = self.bn(attn)
        
        return x * torch.sigmoid(attn)


# ============================================================================
# C2f WITH FAD-LKALITE
# ============================================================================
class C2f_FADLKALite(nn.Module):
    """C2f module with FAD-LKALite attention for YOLOv8m-seg"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, dwk=3, dil_k=7, dil=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        self.lka = nn.ModuleList(
            FrequencyAdaptiveLKALite(self.c, dwk=dwk, dil_k=dil_k, dil=dil) 
            for _ in range(n)
        )
    
    def forward(self, x):
        """Forward pass with FAD-LKALite attention"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(self.lka[i](m(y[-1])) for i, m in enumerate(self.m))
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        """Forward with explicit chunking (alternative implementation)"""
        y = list(self.cv1(x).chunk(2, 1))
        for i, m in enumerate(self.m):
            yi = m(y[-1])
            yi = self.lka[i](yi)
            y.append(yi)
        return self.cv2(torch.cat(y, 1))

