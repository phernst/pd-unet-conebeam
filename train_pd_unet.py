from argparse import ArgumentParser
import json
from typing import List, Any
import os
from os.path import join as pjoin

import cv2
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
import torch
from torch.utils.data import DataLoader
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D

from datasets import LungDataset
from models.primal_dual import PrimalDualNetwork as Net
from models.primal_dual import PrimalUnetBlock
from utilities.ct_utils import fdk_reconstruction
from utilities.transforms import SpaceNormalization
from utilities.ict_system import default_cone_geometry

seed_everything(1701)


class PrimalDual(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        geom = default_cone_geometry()
        theta = np.arange(360)[::self.hparams.sparsity] \
            if self.hparams.subtype == 'sparse' \
            else np.arange(360)[:self.hparams.sparsity]
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
        self.net = Net(radon, 4, 5, 2,
                       use_original_block=False,
                       use_original_init=False,
                       norm=self.hparams.per99)
        self.loss = torch.nn.L1Loss()
        self.trafo_aug = []  # [AddNoise()]
        self.trafo = []  # [SortSino(), ToTensor()]
        # self.example_input_array = [
        #     torch.empty(8, 1, 363, len(theta)),
        #     torch.empty(8, 1, 256, 256),
        # ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=1e-3,
        #     epochs=self.hparams.max_epochs,
        #     steps_per_epoch=3,
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'step',
            # },
            'monitor': 'val_loss',
        }

    def forward(self, *args, **kwargs):
        sparse_sino, under_reco = args[0], args[1]
        return self.net(sparse_sino, under_reco)

    def training_step(self, *args, **kwargs):
        batch = args[0]
        sparse_sino, fulldata = batch
        prediction = self(
            sparse_sino,
            fdk_reconstruction(sparse_sino, self.net.radon, 'hann'))
        loss = self.loss(prediction, fulldata)
        return loss

    def validation_step(self, *args, **kwargs):
        batch, batch_idx = args[0], args[1]
        sparse_sino, fulldata = batch
        prediction = self(
            sparse_sino,
            fdk_reconstruction(sparse_sino, self.net.radon, 'hann'))
        loss = self.loss(prediction, fulldata)

        if batch_idx % 10 == 0:
            os.makedirs(
                pjoin(self.hparams.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}'),
                exist_ok=True,
            )
            img = fulldata.cpu().numpy()[0][0][64]
            cv2.imwrite(
                pjoin(self.hparams.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}/{batch_idx}_out_gt.png'),
                img/img.max()*255)

            reco = prediction.cpu().numpy()[0][0][64]
            reco[reco < 0] = 0
            cv2.imwrite(
                pjoin(self.hparams.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}/{batch_idx}_out_pred.png'),
                reco/img.max()*255,
            )

        return {'val_loss': loss}

    def create_dataset(self, training: bool) -> LungDataset:
        subjects = self.hparams.train_subjects \
            if training else self.hparams.valid_subjects
        return LungDataset(
            [pjoin(self.hparams.ds_dir, f'{f}.pt') for f in subjects],
            self.net.radon,
            training,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(training=True),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False,
                          num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(training=False),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False,
                          num_workers=0)

    def predict_dataloader(self):
        ...

    def test_dataloader(self):
        ...

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('training', avg_loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ds_dir', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--run_name', type=str)
        parser.add_argument('--per99', type=SpaceNormalization)
        parser.add_argument('--subtype', type=str, default='sparse')
        parser.add_argument('--sparsity', type=int, default=8)
        return parser


def main():
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = json_data['ds_dir']
        valid_dir: str = json_data["valid_dir"]

    with open('train_valid_test_lung.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        train_subjects: str = json_data["train_subjects"]
        valid_subjects: str = json_data["valid_subjects"]

    subtype: str = 'sparse'
    sparsity: int = 4
    run_name: str = f'{subtype}{sparsity}_pd_unet'
    num_epochs: int = 151
    use_amp: bool = True
    # 99th percentile of the gray values in the ct dataset
    per99 = SpaceNormalization(
        sino=4.509766686324887,
        img=0.025394852084501224)

    parser = ArgumentParser()
    parser = PrimalDual.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.lr = 1e-3
    hparams.max_epochs = num_epochs
    hparams.batch_size = 1
    hparams.ds_dir = ds_dir
    hparams.valid_dir = valid_dir
    hparams.run_name = run_name
    hparams.per99 = per99
    hparams.subtype = subtype
    hparams.sparsity = sparsity
    hparams.train_subjects = train_subjects
    hparams.valid_subjects = valid_subjects

    model = PrimalDual(**vars(hparams))
    assert isinstance(model.net.primal_blocks[0], PrimalUnetBlock), \
        "Make sure PrimalDualNetwork uses PrimalUnetBlocks"

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(valid_dir, run_name),
        monitor='val_loss',
        save_last=True,
    )
    lr_callback = LearningRateMonitor()

    logger = TensorBoardLogger('lightning_logs', name=hparams.run_name)
    trainer = Trainer(
        logger=logger,
        precision=16 if use_amp else 32,
        gpus=1,
        callbacks=[checkpoint_callback, lr_callback],
        max_epochs=hparams.max_epochs,
        accumulate_grad_batches=16,
        log_every_n_steps=20,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
