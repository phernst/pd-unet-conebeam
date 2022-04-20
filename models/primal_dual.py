from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D

from utilities.ct_utils import fdk_reconstruction
from utilities.transforms import ZNorm
from utilities.transforms import SpaceNormalization
from .unet import UNet


class PrePadDualConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[List[int], Tuple[int, int], int]):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        with torch.no_grad():
            self.conv.bias.zero_()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def _angle_padding(self, inp):
        return F.pad(
            inp, (
                0, 0,
                0, 0,
                self.kernel_size[0]//2, self.kernel_size[0]//2
            ), mode='circular')

    def _proj_padding(self, inp):
        return F.pad(
            inp, (
                self.kernel_size[2]//2, self.kernel_size[2]//2,
                self.kernel_size[1]//2, self.kernel_size[1]//2,
                0, 0
            ), mode='replicate')

    def forward(self, x):
        return self.conv(self._proj_padding(self._angle_padding(x)))


class DualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = nn.Sequential(
            PrePadDualConv3D(features+2, 32, 3),
            nn.PReLU(32),
            PrePadDualConv3D(32, 32, 3),
            nn.PReLU(32),
            PrePadDualConv3D(32, features, 3),
        )
        self.diff_weight = nn.Parameter(torch.ones(1, features, 1, 1, 1))

    def forward(self, h: torch.Tensor, f: torch.Tensor, g: torch.Tensor):
        B, _, A, V, U = h.shape
        block_input = torch.cat([h, torch.mean(f, dim=1, keepdim=True), g], 1)
        layers_out = self.layers(block_input)
        repeated_weights = self.diff_weight.repeat(B, 1, A, V, U)
        scaled_layers_out = repeated_weights*layers_out
        return h + scaled_layers_out


class PrimalBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(features+1, 32, 3, padding=1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.PReLU(32),
            nn.Conv3d(32, features, 3, padding=1),
        )
        self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1, 1))

    def forward(self, h: torch.Tensor, f: torch.Tensor):
        B, _, D, H, W = f.shape
        block_input = torch.cat([f, torch.mean(h, dim=1, keepdim=True)], 1)
        layers_out = self.layers(block_input)
        repeated_weights = self.diff_weight.repeat(B, 1, D, H, W)
        scaled_layers_out = repeated_weights*layers_out
        return f + scaled_layers_out


class PrimalUnetBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = UNet(features+1, features, wf=5)
        self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1, 1))

    def forward(self, h, f):
        B, _, D, H, W = f.shape
        block_input = torch.cat([f, torch.mean(h, dim=1, keepdim=True)], 1)
        return f + self.diff_weight.repeat(B, 1, D, H, W)*self.layers(block_input)


class PrimalDualNetwork(nn.Module):
    def __init__(self, radon: ConeBeam,
                 n_primary: int, n_dual: int, n_iterations: int,
                 use_original_block: bool = True,
                 use_original_init: bool = True,
                 norm: Optional[SpaceNormalization] = None):
        super().__init__()
        self.primal_blocks = nn.ModuleList([
            (PrimalBlock if use_original_block else PrimalUnetBlock)(n_primary)
            for _ in range(n_iterations)
        ])
        self.dual_blocks = nn.ModuleList([
            DualBlock(n_dual) for _ in range(n_iterations)
        ])
        self.radon = radon
        self.n_primary = n_primary
        self.n_dual = n_dual
        self.use_original_init = use_original_init
        self.norm = norm or SpaceNormalization(1.0, 1.0)

    def get_primal_dual_diff_weights(self):
        return {
            'primal': [f.diff_weight.mean().item() for f in self.primal_blocks],
            'dual': [f.diff_weight.mean().item() for f in self.dual_blocks]
        }

    def forward(self, sino: torch.Tensor, sparse_reco: torch.Tensor,
                output_stages: bool = False):
        # g, h: z-normed
        # f: per99-normed
        g = sino
        znorm = ZNorm(g)
        g = znorm.normalize(g)
        B, _, U, V, A = g.shape
        h = torch.zeros(B, self.n_dual, U, V, A, device=g.device)
        if self.use_original_init:
            f = torch.zeros(
                B, self.n_primary, sparse_reco.shape[-3],
                sparse_reco.shape[-2], sparse_reco.shape[-1],
                device=g.device)
        else:
            f = sparse_reco.repeat(1, self.n_primary, 1, 1, 1)/self.norm.img

        stages = []
        for primary_block, dual_block in zip(self.primal_blocks, self.dual_blocks):
            h = dual_block(h, znorm.normalize(self.radon.forward(f*self.norm.img)), g)
            f = primary_block(fdk_reconstruction(znorm.unnormalize(h), self.radon, 'hann')/self.norm.img, f)
            stages.append(torch.mean(f, dim=1, keepdim=True))

        if output_stages:
            return torch.mean(f*self.norm.img, dim=1, keepdim=True), stages

        return torch.mean(f*self.norm.img, dim=1, keepdim=True)


class PrimalDualNetworkSino(nn.Module):
    def __init__(self, radon: ConeBeam, radon_end: ConeBeam,
                 n_primary: int, n_dual: int, n_iterations: int,
                 use_original_block: bool = True,
                 use_original_init: bool = True,
                 norm: Optional[SpaceNormalization] = None):
        super().__init__()
        self.primal_blocks = nn.ModuleList([
            (PrimalBlock if use_original_block else PrimalUnetBlock)(n_primary)
            for _ in range(n_iterations)
        ])
        self.dual_blocks = nn.ModuleList([
            DualBlock(n_dual) for _ in range(n_iterations)
        ])
        self.radon = radon
        self.radon_end = radon_end
        self.n_primary = n_primary
        self.n_dual = n_dual
        self.use_original_init = use_original_init
        self.norm = norm or SpaceNormalization(1.0, 1.0)

    def get_primal_dual_diff_weights(self):
        return {
            'primal': [f.diff_weight.mean().item() for f in self.primal_blocks],
            'dual': [f.diff_weight.mean().item() for f in self.dual_blocks]
        }

    def forward(self, sino: torch.Tensor, sparse_reco: torch.Tensor,
                output_stages: bool = False):
        # g, h: z-normed
        # f: per99-normed
        g = sino
        znorm = ZNorm(g)
        g = znorm.normalize(g)
        B, _, P, A = g.shape
        h = torch.zeros(B, self.n_dual, P, A, device=g.device)
        if self.use_original_init:
            f = torch.zeros(B, self.n_primary, self.in_size,
                            self.in_size, device=g.device)
        else:
            f = sparse_reco.repeat(1, self.n_primary, 1, 1)/self.norm.img

        stages = []
        for primary_block, dual_block in zip(self.primal_blocks, self.dual_blocks):
            h = dual_block(h, znorm.normalize(self.radon.forward(f*self.norm.img)), g)
            f = primary_block(fdk_reconstruction(znorm.unnormalize(h), self.radon, 'hann')/self.norm.img, f)
            stages.append(torch.mean(f, dim=1, keepdim=True))

        if output_stages:
            return self.radon_end.forward(torch.mean(f*self.norm.img, dim=1, keepdim=True)), stages

        full_reco = torch.mean(f*self.norm.img, dim=1, keepdim=True)
        full_sino = self.radon_end.forward(full_reco)
        return full_sino, full_reco


def test_conebeam():
    from utilities.ict_system import ArtisQSystem, DetectorBinning
    ct_system = ArtisQSystem(DetectorBinning.BINNING4x4)
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False, dtype=np.float32)
    src_dist = ct_system.carm_span*4/6
    det_dist = ct_system.carm_span*2/6
    det_spacing_v = ct_system.pixel_dims[1]
    radon = ConeBeam(
        det_count_u=ct_system.nb_pixels[0],
        angles=angles,
        src_dist=src_dist,
        det_dist=det_dist,
        det_count_v=ct_system.nb_pixels[1],
        det_spacing_u=ct_system.pixel_dims[0],
        det_spacing_v=det_spacing_v,
        pitch=0.0,
        base_z=0.0,
        volume=Volume3D(128)
    )
    volume = torch.zeros(1, 1, 128, 128, 128, device='cuda', dtype=torch.float32)
    projections = radon.forward(volume)
    print(projections.shape)


if __name__ == '__main__':
    test_conebeam()
