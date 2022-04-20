import numpy as np
import torch
import torch.nn.functional as F
from torch_radon.filtering import FourierFilters
from torch_radon.radon import ConeBeam


def mu2hu(volume: torch.Tensor, mu_water: float = 0.02) -> torch.Tensor:
    return (volume - mu_water)/mu_water * 1000


def hu2mu(volume: torch.Tensor, mu_water: float = 0.02) -> torch.Tensor:
    return (volume * mu_water)/1000 + mu_water


def filter_projections(projections: torch.Tensor, filter_name="ramp"):
    fourier_filters = FourierFilters()
    projections = projections.permute(0, 1, 3, 2, 4)
    proj_shape = projections.shape
    projections = projections.reshape(np.prod(proj_shape[:-2]), proj_shape[-2], proj_shape[-1])
    size = projections.size(-1)
    n_angles = projections.size(-2)

    # Pad projections to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    padded_projections = F.pad(projections.float(), (0, pad, 0, 0))

    proj_fft = torch.fft.rfft(padded_projections, norm='ortho')

    # get filter and apply
    f = fourier_filters.get(padded_size, filter_name, projections.device)[..., 0]
    filtered_proj_fft = proj_fft * f

    # Inverse fft
    filtered_projections = torch.fft.irfft(filtered_proj_fft, norm='ortho')
    filtered_projections = filtered_projections[:, :, :-pad] * (np.pi / (2 * n_angles))

    return filtered_projections.to(dtype=projections.dtype).reshape(proj_shape).permute(0, 1, 3, 2, 4)


def fdk_reconstruction(projections: torch.Tensor,
                       conebeam: ConeBeam,
                       filter_name: str = 'ramp') -> torch.Tensor:
    reco = conebeam.backprojection(filter_projections(projections, filter_name))
    det_spacing_v = conebeam.projection.cfg.det_spacing_v
    src_dist = conebeam.projection.cfg.s_dist
    det_dist = conebeam.projection.cfg.d_dist
    src_det_dist = src_dist + det_dist
    reco *= det_spacing_v/src_det_dist*src_dist
    return reco
