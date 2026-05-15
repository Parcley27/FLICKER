
import h5py
import json
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

# splits data into train / validation / test sets
# 80 / 10 / 10 % respectively
def makeSplits(h5Path) -> list[list, list, list]:
    file = h5py.File(h5Path, "r")

    allLabels = []

    for ticID in file.keys():
        for observationIndex in file[ticID].keys():
            allLabels.append(int(file[ticID][observationIndex]["exoplanetLabel"][()]))

    file.close()

    allIndices = list(range(len(allLabels)))

    # uses a random split, but based on a set seed so it's reproducible
    trainValIndices, testIndices, trainValLabels, _ = train_test_split(
        allIndices, allLabels, test_size = 0.1, stratify = allLabels, random_state = 27

    )

    trainIndices, valIndices = train_test_split(
        trainValIndices, test_size = 1/9, stratify = trainValLabels, random_state = 27

    )

    return trainIndices, valIndices, testIndices


class TransitDataset(data.Dataset):
    def __init__(self, h5Path, statsPath, indices = None):
        self.h5Path = h5Path
        self.file = None

        index = []

        with h5py.File(h5Path, "r") as f:
            for ticID in f.keys():
                for observationIndex in f[ticID].keys():
                    index.append((ticID, observationIndex))

        self.index = np.array(index)

        if indices is not None:
            self.index = self.index[indices]

        with open(statsPath) as f:
            stats = json.load(f)

        self.nanMask = np.isnan(stats["std"])

    def __len__(self):
        return len(self.index)

    def _openFile(self):
        if self.file is None:
            self.file = h5py.File(self.h5Path, "r")

    # called like dataset.labelCounts because its a property
    @property
    def labelCounts(self):
        # count positives and negatives across the active split
        with h5py.File(self.h5Path, "r") as f:
            positive = sum(
                int(f[ticID][obsIdx]["exoplanetLabel"][()])
                for ticID, obsIdx in self.index

            )

        negative = len(self.index) - positive

        return {"positive": positive, "negative": negative}

    def __getitem__(self, index):
        self._openFile()

        ticID, observationIndex = self.index[index]
        sample = self.file[ticID][observationIndex]

        globalView = torch.tensor(sample["globalView"][()].T, dtype=torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        localView = torch.tensor(sample["localView"][()].T, dtype=torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        secondaryView = torch.tensor(sample["secondaryView"][()].T, dtype=torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

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

    splits = makeSplits("data/processed/dataset.h5")

    print(len(splits[0]))
    print(len(splits[1]))
    print(len(splits[2]))
