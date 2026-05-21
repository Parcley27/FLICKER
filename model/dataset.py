# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalSubscript=false

import argparse
import h5py
import json
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path

from config import numClasses, classNames

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

noiseIntensity = 0.01

def remapLabel(rawLabel):
    """Merge N (4) into J (3) — both mean 'not a planet candidate'."""
    return 3 if rawLabel == 4 else rawLabel

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

        # filter out unlabeled samples (label == -1)
        filtered = []

        with h5py.File(h5Path, "r") as f:
            for ticID, obsIdx in self.index:
                if int(f[ticID][obsIdx]["label"][()]) != -1: # type: ignore
                    filtered.append((ticID, obsIdx))

        self.index = np.array(filtered)

        with open(statsPath) as f:
            stats = json.load(f)

        self.nanMask = np.isnan(stats["std"])

    def __len__(self):
        return len(self.index)

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def _openFile(self):
        if self.file is None:
            self.file = h5py.File(self.h5Path, "r")

    @property
    def labelCounts(self) -> list[int]:
        counts = [0] * numClasses

        with h5py.File(self.h5Path, "r") as f:
            for ticID, obsIdx in self.index:
                label = remapLabel(int(f[ticID][obsIdx]["label"][()])) # type: ignore
                counts[label] += 1

        return counts

    @property
    def sampleWeights(self) -> list[float]:
        labels = []

        with h5py.File(self.h5Path, "r") as f:
            for ticID, obsIdx in self.index:
                labels.append(remapLabel(int(f[ticID][obsIdx]["label"][()]))) # type: ignore

        counts = self.labelCounts
        weightByLabel = {i: 1.0 / max(counts[i], 1) for i in range(numClasses)}

        return [weightByLabel[label] for label in labels]

    def __getitem__(self, index):
        self._openFile()

        ticID, observationIndex = self.index[index]
        sample = self.file[ticID][observationIndex]

        globalView = torch.tensor(sample["globalView"][()].T, dtype = torch.float32).nan_to_num(nan = 0.0, posinf = 0.0, neginf = 0.0).clamp(-5.0, 5.0)
        localView = torch.tensor(sample["localView"][()].T, dtype = torch.float32).nan_to_num(nan = 0.0, posinf = 0.0, neginf = 0.0).clamp(-5.0, 5.0)
        secondaryView = torch.tensor(sample["secondaryView"][()].T, dtype = torch.float32).nan_to_num(nan = 0.0, posinf = 0.0, neginf = 0.0).clamp(-5.0, 5.0)
        halfPeriodView = torch.tensor(sample["halfPeriodView"][()].T, dtype = torch.float32).nan_to_num(nan = 0.0, posinf = 0.0, neginf = 0.0).clamp(-5.0, 5.0)

        scalars = sample["scalars"][()]
        scalars = torch.tensor(scalars, dtype = torch.float32)

        label = torch.tensor(remapLabel(int(sample["label"][()])), dtype = torch.long) # type: ignore

        if self.augment:
            # add random noise to views only
            globalView += torch.randn_like(globalView) * noiseIntensity
            localView += torch.randn_like(localView) * noiseIntensity
            secondaryView += torch.randn_like(secondaryView) * noiseIntensity
            halfPeriodView += torch.randn_like(halfPeriodView) * noiseIntensity

            # add noise to scalars, then re-apply NaN mask so missing values stay at 0
            scalars += torch.randn_like(scalars) * noiseIntensity
            scalars[self.nanMask] = 0.0

            flip = torch.rand(1) < 0.5

            if flip:
                globalView = torch.flip(globalView, dims = [-1])
                localView = torch.flip(localView, dims = [-1])
                halfPeriodView = torch.flip(halfPeriodView, dims = [-1])
                # secondary view not flipped because it's centered on secondaryPhase,
                # not phase 0, so flipping would be physically inconsistent

        return {
            "globalView": globalView,
            "localView": localView,
            "secondaryView": secondaryView,
            "halfPeriodView": halfPeriodView,
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
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    for key in sample:
        print(f"  {key}: Shape = {sample[key].shape}, dtype = {sample[key].dtype}")

    print(f"\nFirst sample label: {sample['label']}")

    splits = makeSplits(args.data)

    print(f"\nSplits: Train = {len(splits[0])}, Evaluate = {len(splits[1])}, Test = {len(splits[2])}")

    counts = dataset.labelCounts
    print(f"\nLabel distribution:")

    for i, name in enumerate(classNames):
        print(f"  {name}: {counts[i]}")
