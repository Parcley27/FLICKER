import h5py
import json
import numpy as np
import torch
import torch.utils.data as data

class TransitDataset(data.Dataset):
    def __init__(self, h5Path, statsPath):
        self.statsPath = statsPath

        self.file = h5py.File(h5Path, "r")

        self.index = []

        for ticID in self.file.keys():
            for observationIndex in self.file[ticID].keys():
                self.index.append((ticID, observationIndex))
        
        self.index = np.array(self.index)

        with open(statsPath) as f:
            stats = json.load(f)
        self.nanMask = np.isnan(stats["std"])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ticID, observationIndex = self.index[index]
        sample = self.file[ticID][observationIndex]

        globalView = torch.tensor(sample["globalView"][()].T, dtype=torch.float32)
        localView = torch.tensor(sample["localView"][()].T, dtype=torch.float32)
        secondaryView = torch.tensor(sample["secondaryView"][()].T, dtype=torch.float32)

        scalars = sample["scalars"][()].copy()

        scalars[self.nanMask] = 0.0
        scalars = torch.tensor(scalars, dtype=torch.float32)

        label = torch.tensor(float(sample["exoplanetLabel"][()]), dtype=torch.float32)

        return {
            "globalView": globalView,
            "localView": localView,
            "secondaryView": secondaryView,
            "scalars": scalars,
            "label": label,

        }
    
if __name__ == "__main__":
    dataset = TransitDataset("data/processed/dataset.h5", "data/processed/scalar_stats.json")
    print(len(dataset))

    sample = dataset[0]

    for key in sample:
        print(f"{key}, {sample[key].shape}, {sample[key].dtype}")

    print(dataset[0]["label"])