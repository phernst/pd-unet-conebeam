import json
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D
from tqdm import tqdm
from datasets import LungDataset

from utilities.ct_utils import fdk_reconstruction
from utilities.ict_system import default_cone_geometry
from utils import calculate_metrics


def main(*, sparsity: int):
    with open('train_valid_test_lung.json', 'r', encoding='utf-8') as json_file:
        test_subjects = json.load(json_file)['test_subjects']

    out_dir = pjoin('test', f'sparse{sparsity}_fdk')
    os.makedirs(out_dir, exist_ok=True)

    geom = default_cone_geometry()
    theta = np.arange(360)[::sparsity]
    radon = ConeBeam(
        det_count_u=geom.det_count_u,
        angles=np.deg2rad(theta),
        det_count_v=geom.det_count_v,
        det_spacing_u=geom.det_spacing_u,
        det_spacing_v=geom.det_spacing_v,
        src_dist=geom.src_dist,
        det_dist=geom.det_dist,
        pitch=geom.pitch,
        base_z=geom.base_z,
        volume=Volume3D(128),
    )

    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = json_data['ds_dir']
    test_dataset = LungDataset(
        [pjoin(ds_dir, f'{f}.pt') for f in test_subjects],
        radon,
        training=False,
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
    )

    metrics = []
    with torch.inference_mode():
        for batch in tqdm(dataloader_test):
            batch_sparse_sino, batch_fulldata = batch
            batch_prediction = fdk_reconstruction(batch_sparse_sino, radon, 'hann')

            for prediction, fulldata in zip(batch_prediction, batch_fulldata):
                metrics += calculate_metrics(prediction, fulldata)

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, "Results.csv"))


if __name__ == '__main__':
    seed_everything(1701)
    main(sparsity=16)
