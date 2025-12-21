import torch
import numpy as np

from torch.utils.data import Dataset
from engine import ConwayGame


class MnistDataset(Dataset):
    def __init__(self, npz_path, seed, conway_steps=1):
        self.data = np.load(npz_path, mmap_mode="r") # do not load everything in RAM
        self.X = self.data["x"]
        self.y = self.data["y"]
        self.seed = seed
        self.conway_steps = conway_steps

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]

        # ConwayGame on-the-fly
        conway = ConwayGame(state=img, seed=self.seed,  size=img.shape[0])
        final_state = conway.generate_sequence(steps=self.conway_steps)[-1]

        # To tensor
        x = torch.from_numpy(final_state).float().unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y
