import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

from src.ml.transform import MinMaxScaler4D
from src.ml.utils import reshape_data


class ChannelsData(torch.utils.data.Dataset):
    def __init__(self, h_freqs: np.ndarray, pilot_matrices: np.ndarray):
        self.h_freqs = h_freqs
        self.pilot_matrices = pilot_matrices
        self.len_dataset = h_freqs.shape[0]

    def __getitem__(self, item):
        channel_res = self.h_freqs[item]
        pilot_matrices = self.pilot_matrices[item]

        item_channel_res = torch.FloatTensor(channel_res)
        item_pilot_matrices = torch.FloatTensor(pilot_matrices)

        return item_channel_res, item_pilot_matrices

    def __len__(self):
        return len(self.h_freqs)


class ChannelDataloader:
    def __init__(
        self,
        h_freqs: np.ndarray,
        pilot_matrices: np.ndarray,
        batch_size: int = 64,
        pin_memory: bool = False,
        num_worker: int = 1,
        n_splits: int = 5,  # Number of folds for cross-validation
        shuffle: bool = True,
    ):
        self.dataset = ChannelsData(h_freqs, pilot_matrices)
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_worker = num_worker
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle)

    def get_fold_dataloaders(self):
        """Returns a list of (train_loader, val_loader) tuples for each fold"""
        fold_loaders = []

        for train_idx, val_idx in self.kfold.split(range(len(self.dataset))):
            # Create subsets for training and validation
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                shuffle=True,
                num_workers=self.num_worker,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                shuffle=False,
                num_workers=self.num_worker,
            )

            fold_loaders.append((train_loader, val_loader))

        return fold_loaders

    # Optional: Keep original split methods for compatibility
    def train_dataloader(self):
        """Returns a single train dataloader with 80% of data"""
        train_size = int(0.8 * len(self.dataset))
        indices = torch.randperm(len(self.dataset)).tolist()
        train_subset = Subset(self.dataset, indices[:train_size])

        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=True,
            num_workers=self.num_worker,
        )

    def test_dataloader(self):
        """Returns a single test dataloader with remaining 20% of data"""
        train_size = int(0.8 * len(self.dataset))
        indices = torch.randperm(len(self.dataset)).tolist()
        test_subset = Subset(self.dataset, indices[train_size:])

        return DataLoader(
            test_subset,
            batch_size=1,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=self.num_worker,
        )


if __name__ == "__main__":
    from src.settings.ml import batch_size, n_splits, num_workers

    pilot_matrices = np.load(
        "data/txant_4_rxant_8_speed_0_samples_1000/pilot_matrices.npy"
    )
    h_freqs = np.load("data/txant_4_rxant_8_speed_0_samples_1000/h_freqs.npy")
    (
        num_samples,
        batch,
        num_rx,
        number_ant_rx,
        num_tx,
        number_and_tx,
        symbols,
        subcarriers,
    ) = pilot_matrices.shape

    pilot_matrices = reshape_data(pilot_matrices)
    h_freqs = reshape_data(h_freqs)

    dataloader = ChannelDataloader(
        pilot_matrices=pilot_matrices,
        h_freqs=h_freqs,
        batch_size=batch_size,
        num_worker=num_workers,
        n_splits=n_splits,
    )
    fold_loaders = dataloader.get_fold_dataloaders()

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"Fold {fold + 1}")
        # Training loop
        for batch in train_loader:
            channels, pilot_matrices = batch
            print(
                f"Training -> Channels shape: {channels.shape}, Pilot matrix shape: {pilot_matrices.shape}"
            )
            break
            # Your training code here

        # Validation loop
        for batch in val_loader:
            channels, pilot_matrices = batch
            print(
                f"Eval -> Channels shape: {channels.shape}, Pilot matrix shape: {pilot_matrices.shape}"
            )
            break
            # Your validation code here
