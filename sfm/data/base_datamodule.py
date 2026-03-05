import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from copy import deepcopy


class BaseDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        n_workers: int = 16,
        **ds_kwargs
        ):
        super().__init__()

        # initialize attributes
        self.batch_size = batch_size
        self.n_workers = n_workers

        # train dataset
        train_kwargs = deepcopy(ds_kwargs)
        self.train_dataset = dataset(
            dataset_name='train',
            **train_kwargs,
        )

        # val dataset
        val_kwargs = deepcopy(ds_kwargs)
        val_kwargs['data_params']['corpus_params']['dynamic_mixing'] = False  # disable dynamic mixing
        self.val_dataset = dataset(
            dataset_name='val',
            **val_kwargs,
        )

        # test dataset
        test_kwargs = deepcopy(ds_kwargs)
        test_kwargs['data_params']['corpus_params']['dynamic_mixing'] = False  # disable dynamic mixing
        test_kwargs['data_params']['corpus_params']['audio_time'] = None  # test on full length audio
        self.test_dataset = dataset(
            dataset_name='test',
            **test_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.n_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            collate_fn=collate_fn, 
            persistent_workers=True if self.n_workers > 0 else False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn, 
            persistent_workers=False,
            pin_memory=True
        )

def collate_fn(batch: dict):
    assert isinstance(batch, list), (type(batch), 'expect list')
    return {
        key: torch.stack(
            [sample[key] for sample in batch], dim=0
        ) if isinstance(value, torch.Tensor) else [
            sample[key] for sample in batch
        ] for key, value in batch[0].items()
    }
