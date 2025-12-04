import torch
from torch import nn



class FNO_3D(nn.Module):
    def __init__(self, hidden_channels, modesSpace):
        super().__init__()
        
        self.modesSpace = modesSpace

        self.scale = 0.02

        self.weights  = nn.Parameter(self.scale * torch.rand(hidden_channels, hidden_channels, self.modesSpace, 3, dtype=torch.cfloat))
        
        self.bias = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)

        self.activate = nn.GELU()


    def compl_mul3d(self, x, weights): 
    # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)

        x = torch.einsum("bixyz,iox->boxyz", x, weights[..., 0])
        x = torch.einsum("bixyz,ioy->boxyz", x, weights[..., 1])
        x = torch.einsum("bixyz,ioz->boxyz", x, weights[..., 2])

        # return torch.einsum("bixyz,ioxyz->boxyz", x, weights)
        return x

        
    def forward(self, x):
        B, C, X, Y, Z = x.shape

        bias = self.bias(x)

        x_fft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        del x

        out_ft = torch.zeros(B, C, X, Y, Z, dtype=torch.cfloat, device=x_fft.device)
        
        out_ft[..., :self.modesSpace, :self.modesSpace, :self.modesSpace] = self.compl_mul3d(x_fft[..., :self.modesSpace, :self.modesSpace, :self.modesSpace], self.weights)
        del x_fft

        out_ft = torch.fft.irfftn(out_ft, s=(X, Y, Z))

        return self.activate(out_ft + bias)