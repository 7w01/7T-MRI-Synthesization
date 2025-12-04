import torch
import torch.nn as nn
from typing import Optional

class ConvFFN3D(nn.Module):
    """Convolutional Feed-Forward Network for 3D data"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        kernel_size: int = 3,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size // 2, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        # (B, C, H, W, D) -> (B, H, W, D, C) -> (B, H*W*D, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        # Reshape for 3D conv: (B, H*W*D, C) -> (B, H, W, D, C) -> (B, C, H, W, D)
        x = x.view(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        x = self.dwconv(x)
        # Back to original shape for fc2: (B, C, H, W, D) -> (B, H, W, D, C) -> (B, H*W*D, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, x.shape[1])
        
        x = self.fc2(x)
        x = self.drop(x)
        
        # Reshape back to (B, C, H, W, D)
        x = x.view(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x
