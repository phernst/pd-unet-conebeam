import json
import os
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import LungDataset

from train_pd_orig import PrimalDual
from utilities.ct_utils import fdk_reconstruction, mu2hu
from utils import calculate_metrics


def main(run_name: str, save_visuals: bool = False):
    with open('train_valid_test_lung.json', 'r', encoding='utf-8') as json_file:
        test_subjects = json.load(json_file)['test_subjects']

    checkpoint_dir = pjoin('validation', run_name)
    checkpoint_path = sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin('test', run_name)
    os.makedirs(out_dir, exist_ok=True)

    visual_dir = pjoin('visual', run_name)
    os.makedirs(visual_dir, exist_ok=True)

    model = PrimalDual.load_from_checkpoint(
        pjoin(checkpoint_dir, checkpoint_path))
    model.eval()
    model.cuda()

    test_dataset = LungDataset(
        [pjoin(model.hparams.ds_dir, f'{f}.pt') for f in test_subjects],
        model.net.radon,
        training=False,
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=model.hparams.batch_size,
    )

    metrics = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader_test)):
            batch_sparse_sino, batch_fulldata = batch
            batch_sparse_fdk = fdk_reconstruction(batch_sparse_sino, model.net.radon, 'hann')
            batch_prediction = model(
                batch_sparse_sino,
                batch_sparse_fdk,
            )

            for prediction, fulldata in zip(batch_prediction, batch_fulldata):
                metrics += calculate_metrics(prediction, fulldata)

            if save_visuals and batch_idx == 0:
                img = nib.Nifti1Image(mu2hu(batch_fulldata[0, 0].clamp(min=0)).cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, "gt.nii.gz"))
                img = nib.Nifti1Image(mu2hu(batch_sparse_fdk[0, 0].clamp(min=0)).cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, "in.nii.gz"))
                img = nib.Nifti1Image(mu2hu(batch_prediction[0, 0].clamp(min=0)).cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, "prediction.nii.gz"))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, "Results.csv"))


def create_visuals(run_name: str):
    main(run_name, save_visuals=True)


if __name__ == '__main__':
    seed_everything(1701)
    # main('sparse16_pd_orig')
    create_visuals('sparse16_pd_orig')
