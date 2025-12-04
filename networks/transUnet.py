import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from .convFFN3d import ConvFFN3D


class WindowedTransformerBlock3D(nn.Module):
    """Windowed Transformer Block for 3D data with integrated attention mechanism"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        mlp_ratio: float = 2.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size  # Height, Width, Depth
        self.mlp_ratio = mlp_ratio
        
        self.sa = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )

        # self.ca = nn.MultiheadAttention(
        #     embed_dim=dim,
        #     num_heads=num_heads,
        #     dropout=attn_drop,
        #     bias=qkv_bias,
        #     batch_first=True
        # )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN3D(in_features=dim, hidden_features=mlp_hidden_dim, 
                                act_layer=act_layer, drop=drop)

    def windowed_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, pH, pW, pD = x.shape

        # Pad input to make it divisible by window size
        pad_h = (self.window_size[0] - pH % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - pW % self.window_size[1]) % self.window_size[1]
        pad_d = (self.window_size[2] - pD % self.window_size[2]) % self.window_size[2]
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        
        _, _, pad_pH, pad_pW, pad_pD = x.shape
        window_grid_size = (pad_pH // self.window_size[0], pad_pW // self.window_size[1], pad_pD // self.window_size[2])
        
        x = (x
            .reshape(B, C, window_grid_size[0], self.window_size[0], window_grid_size[1], self.window_size[1], window_grid_size[2], self.window_size[2])
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(-1, C, self.window_size[0] * self.window_size[1] * self.window_size[2])
            .permute(0, 2, 1)
        ) # (B*num_windows, patch_volume, C)

        # self-attention
        sa, _ = self.sa(x, x, x)

        # # cross-attention
        # k_v = x.reshape(B, np.prod(window_grid_size), np.prod(self.window_size), C)
        # k_v = k_v.mean(dim=2)
        # print(x.shape, k_v.shape)
        # ca, _ = self.ca(x, k_v, k_v)

        # x = sa + ca
        x = sa

        x = (x
            .permute(0, 2, 1)
            .reshape(B, window_grid_size[0], window_grid_size[1], window_grid_size[2], -1, self.window_size[0], self.window_size[1], self.window_size[2])
            .permute(0, 4, 1, 5, 2, 6, 3, 7)
            .reshape(B, -1, pad_pH, pad_pW, pad_pD)
        ) # (B, C, pad_pH, pad_pW, pad_pD)

        # Remove padding if added
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = x[:, :, :pH, :pW, :pD].contiguous()

        return x

    def forward(self, x):
        B, C, pH, pW, pD = x.shape

        x_norm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_norm = self.norm1(x_norm.view(B, -1, C)).view(B, pH, pW, pD, C)
        x_norm = x_norm.permute(0, 4, 1, 2, 3).contiguous() 
        
        attn_out = self.windowed_attention(x_norm)
        x = x + attn_out  # Residual connection
        
        x_norm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_norm = self.norm2(x_norm.view(B, -1, C)).view(B, pH, pW, pD, C)
        x_norm = x_norm.permute(0, 4, 1, 2, 3).contiguous()
        
        convffn_out = self.convffn(x_norm)
        x = x + convffn_out  # Residual connection
        
        return x


class TransUNet3D(nn.Module):
    """TransUNet 3D Model with windowed attention and UNet-like structure"""
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        in_chans: int = 1,
        num_classes: int = 1,
        embed_dim: int = 512,
        depths: int = 2,
        window_sizes: Tuple[Tuple[int, int, int], ...] = ((8, 8, 8), (8, 8, 8), (16, 16, 16), (8, 8, 8)),  # Increasing then decreasing
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = len(window_sizes)
        
        # self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        
        self.encoder_layers = nn.ModuleList()
        self.conv3d_blocks = nn.ModuleList()  # Conv3d blocks between attention modules
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                WindowedTransformerBlock3D(dim=embed_dim, window_size=window_sizes[i_layer]) for _ in range(depths)
            ])
            self.encoder_layers.append(layer)
            
        self.decoder_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i_layer in range(self.num_layers - 2, -1, -1):
            layer = nn.ModuleList([
                WindowedTransformerBlock3D(dim=embed_dim, window_size=window_sizes[i_layer]) for _ in range(depths)
            ])
            self.decoder_layers.append(layer)

            skip_conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
            self.skip_convs.append(skip_conv)

        # Final generation layer
        self.generate = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            # nn.ConvTranspose3d(embed_dim, 4, kernel_size=2, stride=2, padding=0),
            # nn.ReLU(),
            nn.Conv3d(8, 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, embed_dim, pH, pW, pD)

        skip_connections = []
        
        # Encoder
        for i_layer in range(self.num_layers):
            for i_block in range(len(self.encoder_layers[i_layer])):
                x = self.encoder_layers[i_layer][i_block](x)
            
            if i_layer < self.num_layers - 1:
                skip_connections.append(x)
                
        # Decoder
        for i_layer in range(self.num_layers - 1):
            if len(skip_connections) > 0:
                skip = skip_connections.pop()
                skip = self.skip_convs[i_layer](skip)
                x = x + skip
            
            for i_block in range(len(self.decoder_layers[i_layer])):
                x = self.decoder_layers[i_layer][i_block](x)
        
        x = self.generate(x)
        
        return x

