import json
from os.path import join as pjoin
import sys

import numpy as np
import torch

from utilities.ct_utils import hu2mu


def main(ds_dir: str):

    with open('train_valid_test_lung.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        train_subjects: str = json_data["train_subjects"]
        valid_subjects: str = json_data["valid_subjects"]
        test_subjects: str = json_data["test_subjects"]
        subjects = train_subjects + valid_subjects + test_subjects
    prior_list = [pjoin(ds_dir, f'{f}.pt') for f in subjects]

    dataset = [
        np.percentile(hu2mu(torch.load(p)['volume']).cpu().numpy(), 99)
        for p in prior_list
    ]
    print(
        np.min(dataset),
        np.mean(dataset),
        np.median(dataset),  # this is used for LungDataset.percentile
        np.max(dataset),
    )


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please specify the dataset directory"
    main(sys.argv[1])
