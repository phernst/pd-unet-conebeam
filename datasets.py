from enum import Enum, auto
from os.path import join as pjoin
from typing import List, NamedTuple

from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from torch_radon.radon import ConeBeam
from torch_radon.volumes import Volume3D

from utilities.ct_utils import hu2mu
from utilities.ict_system import default_cone_geometry


class SpaceNormalization(NamedTuple):
    sino: float
    img: float


class DatasetType(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class LungDataset(torch.utils.data.Dataset):
    percentile = hu2mu(1720.43359375)

    def __init__(self, prior_paths: List[str], radon: ConeBeam, training: bool):
        self.prior_paths = prior_paths
        self.radon = radon
        self.training = training

    def __len__(self) -> int:
        """For returning the length of the file list"""
        return len(self.prior_paths)

    def __getitem__(self, idx):
        dataset = torch.load(self.prior_paths[idx])
        gt_vol, _ = dataset['volume'], dataset['voxel_size']
        gt_vol = hu2mu(gt_vol[None, None])
        gt_vol = F.interpolate(gt_vol, size=128)

        if self.training and torch.rand(1) > 0.5:
            gt_vol = gt_vol.flip(-1)
        if self.training and torch.rand(1) > 0.5:
            gt_vol = gt_vol.flip(-2)
        if self.training and torch.rand(1) > 0.5:
            gt_vol = gt_vol.flip(-3)
        if self.training and torch.rand(1) > 0.5:
            rrot = torch.rand(3,).numpy()*40-20
            rrot = Rotation.from_euler('zyx', rrot, degrees=True).as_matrix()
            rscale = torch.diag(torch.rand(3,)*.2-.1+1).numpy()
            rrot_scale = rrot@rscale
            theta = torch.zeros(1, 3, 4, dtype=torch.float, device='cuda')
            theta[0, :, :3] = torch.from_numpy(rrot_scale)
            rtrans = torch.rand(3,)*.2-.1
            theta[0, :, 3] = rtrans
            grid = F.affine_grid(theta, (1, 1, 128, 128, 128), align_corners=False)
            gt_vol = F.grid_sample(gt_vol, grid, align_corners=False)

        inter_proj = self.radon.forward(gt_vol.cuda())

        return inter_proj[0], gt_vol[0]


def test_dataset():
    geom = default_cone_geometry()
    radon = ConeBeam(
        det_count_u=geom.det_count_u,
        angles=np.linspace(0, 2*np.pi, 10, endpoint=False),
        src_dist=geom.src_dist,
        det_dist=geom.det_dist,
        det_count_v=geom.det_count_v,
        det_spacing_u=geom.det_spacing_u,
        det_spacing_v=geom.det_spacing_v,
        pitch=0.,
        base_z=0.,
        volume=Volume3D(128),
    )
    ds_dir = '/mnt/nvme2/lungs/lungs3d/priors/'
    subjects = ["R_111"]
    prior_list = [pjoin(ds_dir, f'{f}.pt') for f in subjects]
    dset = LungDataset(prior_list, radon=radon, training=True)
    element = dset[0]
    print(element[0].shape)
    plt.imshow(element[0][0, 0].cpu().numpy())
    plt.show()


if __name__ == '__main__':
    test_dataset()
