import os
import warnings
from os.path import join

import datasets
import numpy as np

if "MOLMO_DATA_DIR" in os.environ:
    DATA_HOME = join(os.environ["MOLMO_DATA_DIR"], "torch_datasets")
else:
    warnings.warn("MOLMO_DATA_DIR is not set, data loading might fail")
    DATA_HOME = None


class Dataset:
    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.get(item, np.random)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, item, rng):
        # `rng` is used to support deterministic data augmentation for tasks that require it.
        # Used to avoid the hazards of relying on the global rng state for determinism
        raise NotImplementedError()


class DeterministicDataset:
    """Dataset wrapper that supports padding and control the random seed based on the epoch"""

    def __init__(self, dataset: Dataset, preprocessor, seed, n_pad=0):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.seed = seed
        self.n_pad = n_pad

    def __len__(self):
        return len(self.dataset) + self.n_pad

    def __getitem__(self, idx):
        return self.get(idx, 0)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, idx, epoch=0):
        rng = np.random.RandomState(self.seed + idx + len(self.dataset)*epoch)
        if idx >= len(self.dataset):
            # Padding example
            item = self.dataset.get(0, rng)
            if "metadata" not in item:
                item["metadata"] = {}
            item["metadata"]["valid"] = False
        else:
            item = self.dataset.get(idx, rng)
        if self.preprocessor:
            item = self.preprocessor(item, rng)
        return item


class DatasetBase(Dataset):
    def __init__(self, split, sample: int=None):
        super().__init__()
        self.split = split
        self.sample = sample
        self.data = self.load()[:self.sample]

    def load(self):
        raise NotImplementedError()

    def __len__(self):
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, item):
        return self.get(item, np.random)

    def get(self, item, rng):
        raise NotImplementedError()


class HfDataset(Dataset):
    PATH = None

    @classmethod
    def download(cls, n_procs=None):
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()

    def __init__(self, split: str, keep_in_memory=True, **kwargs):
        self.split = split
        self.dataset = datasets.load_dataset(
            self.PATH, split=split, keep_in_memory=keep_in_memory, **kwargs)

    def __len__(self):
        return len(self.dataset)
