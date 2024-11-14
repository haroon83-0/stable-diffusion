import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorms_1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorms_2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.groupnorms_1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.groupnorms_2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residual_layer(residue)


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Group normalization to normalize the input tensor
        self.groupnorm = nn.GroupNorm(32, channels)

        # Self-attention mechanism to capture dependencies across the spatial dimensions
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store the original input tensor for the residual connection
        residue = x

        # Reshape and transpose for attention mechanism
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)  # Shape: (batch_size, h*w, channels)

        # Apply attention
        x = self.attention(x)

        # Reshape back to original dimensions
        x = x.transpose(1, 2).view(b, c, h, w)

        # Add the residual (skip connection)
        return x + residue


class VAE_Decoder(nn.Module):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215

        for module in self:
            x = module(x)

        return x
