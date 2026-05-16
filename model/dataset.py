
import argparse
import h5py
import json
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from sklearn.model_selection import train_test_split

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

noiseIntensity = 0.01

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
    def __init__(self, h5Path, statsPath, indices = None, augment = False):
        self.h5Path = h5Path
        self.augment = augment
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

    @property
    def sampleWeights(self):
        # each sample gets weight 1/classCount so the sampler draws both classes equally
        counts = self.labelCounts
        weightByLabel = {1: 1.0 / counts["positive"], 0: 1.0 / counts["negative"]}

        weights = []
        with h5py.File(self.h5Path, "r") as f:
            for ticID, obsIdx in self.index:
                label = int(f[ticID][obsIdx]["exoplanetLabel"][()])
                weights.append(weightByLabel[label])

        return weights

    def __getitem__(self, index):
        self._openFile()

        ticID, observationIndex = self.index[index]
        sample = self.file[ticID][observationIndex]

        globalView = torch.tensor(sample["globalView"][()].T, dtype=torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        localView = torch.tensor(sample["localView"][()].T, dtype=torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        secondaryView = torch.tensor(sample["secondaryView"][()].T, dtype=torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        scalars = sample["scalars"][()].copy()

        scalars[self.nanMask] = 0.0
        scalars = torch.tensor(scalars, dtype = torch.float32)

        label = torch.tensor(float(sample["exoplanetLabel"][()]), dtype=torch.float32)

        if self.augment and label.item() == 1.0:
            # add random noise to views
            globalView += torch.randn_like(globalView) * noiseIntensity
            localView += torch.randn_like(localView) * noiseIntensity
            secondaryView += torch.randn_like(secondaryView) * noiseIntensity

            flip = torch.rand(1) < 0.5

            if flip:
                globalView = torch.flip(globalView, dims = [-1])
                localView = torch.flip(localView, dims = [-1])
                secondaryView = torch.flip(secondaryView, dims = [-1])

            scalars += torch.randn_like(scalars) * noiseIntensity

        return {
            "globalView": globalView,
            "localView": localView,
            "secondaryView": secondaryView,
            "scalars": scalars,
            "label": label,

        }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Inspect the TransitDataset")

    parser.add_argument("--data", type = Path, default = defaultDataPath,
        help = "Path to dataset.h5 (default: data/processed/dataset.h5)")
    parser.add_argument("--scalars", type = Path, default = defaultScalarsPath,
        help = "Path to scalar_stats.json (default: data/processed/scalar_stats.json)")
    
    args = parser.parse_args()

    dataset = TransitDataset(args.data, args.scalars)
    print(len(dataset))

    sample = dataset[0]

    for key in sample:
        print(f"{key}, {sample[key].shape}, {sample[key].dtype}")

    print(dataset[0]["label"])

    splits = makeSplits(args.data)

    print(len(splits[0]))
    print(len(splits[1]))
    print(len(splits[2]))
