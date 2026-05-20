
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

numLabels = 5

# uses the pre-assigned split attribute from the HDF5 file 
def makeSplits(h5Path) -> tuple[list, list, list]:
    trainIndices, valIndices, testIndices = [], [], []

    with h5py.File(h5Path, "r") as f:
        i = 0

        for ticID in f.keys():
            for obsIdx in f[ticID].keys():
                split = f[ticID][obsIdx].attrs.get("split", "train")

                if split == "train":
                    trainIndices.append(i)

                elif split == "val":
                    valIndices.append(i)

                else:
                    testIndices.append(i)

                i += 1

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
    def sampleWeights(self) -> list[float]:
        # single pass: collect per-sample labels, then compute weights from counts
        labels = []

        with h5py.File(self.h5Path, "r") as f:
            for ticID, obsIdx in self.index:
                labels.append(int(f[ticID][obsIdx]["label"][()])) # type: ignore

        counts: dict[int, int] = {}

        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        return [1.0 / counts[label] for label in labels]

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

        if self.augment and rawLabel == 0:
            # add random noise to views only
            globalView += torch.randn_like(globalView) * noiseIntensity
            localView += torch.randn_like(localView) * noiseIntensity
            secondaryView += torch.randn_like(secondaryView) * noiseIntensity

            # add noise to scalars, then re-apply NaN mask so missing values stay at 0
            scalars += torch.randn_like(scalars) * noiseIntensity
            scalars[self.nanMask] = 0.0

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
