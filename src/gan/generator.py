"""
STYLE-SYNTH PyTorch GAN Architecture Modules
Implements ResNet-based Style Generator and 70x70 PatchGAN Discriminator
for high-fidelity Face Style Transfer and Latent Synthesis.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple


class ResidualBlock(nn.Module):
    """
    Residual Bottleneck Block with Reflection Padding and Instance Normalization
    Prevents boundary artifacts and maintains structural facial landmarks.
    """
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """
    9-Block ResNet Generator for high-resolution Face Style Transfer.
    Architecture:
      1. Initial Reflection Padding + 7x7 Conv (RGB -> 64 ch)
      2. Downsampling convolutions with stride 2 (64 -> 128 -> 256 ch)
      3. 8 Residual Blocks with Instance Normalization
      4. Upsampling transposed convolutions / bilinear upsample + Conv (256 -> 128 -> 64 ch)
      5. Output 7x7 Conv + Tanh activation mapping to [-1, 1]
    """
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, n_blocks: int = 8):
        super(ResNetGenerator, self).__init__()
        assert n_blocks >= 0, "n_blocks must be non-negative"
        self.ngf = ngf
        self.n_blocks = n_blocks

        # Initial Conv
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                nn.ReLU(inplace=True)
            ]

        # Residual Bottleneck Blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(int(ngf * mult / 2), affine=True),
                nn.ReLU(inplace=True)
            ]

        # Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale intermediate feature maps for perceptual inspection."""
        features = []
        cur = x
        for i, layer in enumerate(self.model):
            cur = layer(cur)
            if isinstance(layer, (nn.ReLU, ResidualBlock)):
                features.append(cur)
        return features

    def get_summary(self) -> Dict[str, Any]:
        """Returns neural network metadata, parameter count, and receptive field info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_type": "ResNet-9Block-Style-Generator",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers_count": len(self.model),
            "residual_blocks": self.n_blocks,
            "base_features": self.ngf,
            "receptive_field_px": 256,
            "normalization": "InstanceNorm2d",
            "padding_mode": "ReflectionPad2d"
        }


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator for adversarial quality assessment.
    Classifies whether 70x70 overlapping image patches are authentic artistic renders
    or synthetic artifacts.
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        super(PatchGANDiscriminator, self).__init__()
        self.ndf = ndf
        self.n_layers = n_layers

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_adversarial_score(self, x: torch.Tensor) -> float:
        """Computes patch realism confidence score in [0.0, 1.0]."""
        with torch.no_grad():
            patch_logits = self.forward(x)
            prob = torch.sigmoid(patch_logits).mean().item()
        return float(prob)


def build_generator(device: str = "cpu") -> ResNetGenerator:
    """Helper factory to instantiate generator on target device."""
    gen = ResNetGenerator()
    gen.to(device)
    gen.eval()
    return gen
