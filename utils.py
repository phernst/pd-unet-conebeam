import csv

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch

from datasets import LungDataset
from utilities.ct_utils import mu2hu


def calculate_metrics(prediction: torch.Tensor, groundtruth: torch.Tensor) -> dict[str, float]:
    prediction.transpose_(0, 1)
    groundtruth.transpose_(0, 1)

    norm_scaler = LungDataset.percentile

    psnr_val = [peak_signal_noise_ratio(
        (g/norm_scaler).clamp(0, 1).cpu().numpy(),
        (p/norm_scaler).clamp(0, 1).cpu().numpy(),
        data_range=1.0,
    ) for p, g in zip(prediction, groundtruth)]
    ssim_val = [structural_similarity(
        (p/norm_scaler).clamp(0, 1).cpu().numpy(),
        (g/norm_scaler).clamp(0, 1).cpu().numpy(),
        gaussian_weights=True,
        channel_axis=0,
        data_range=1.0,
    ) for p, g in zip(prediction, groundtruth)]

    # RMSE in HU
    prediction_hu = mu2hu(prediction)
    groundtruth_hu = mu2hu(groundtruth)
    rmse_val = ((prediction_hu - groundtruth_hu)**2).mean(dim=(1, 2, 3)).sqrt()
    return [{
        'ssim': s.item(),
        'psnr': p.item(),
        'rmse': r.item(),
    } for s, p, r in zip(ssim_val, psnr_val, rmse_val)]


def read_csv(path: str, metric: str, reduce: bool = True):
    all_values = []
    with open(path, 'r', encoding='utf-8') as file:
        csvreader = csv.DictReader(file)
        for row in csvreader:
            all_values.append(float(row[metric]))

    if not reduce:
        return all_values

    return np.median(all_values), np.percentile(all_values, 25), \
        np.percentile(all_values, 75)
