import sys
from pathlib import Path

import torch
import torch.nn as nn

repoRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repoRoot))

from config import globalBins, localBins, secondaryBins

class ConvolutionTower(nn.Module):
    def __init__(self, channelsIn, inputLength, numBlocks = 3):

        super().__init__()

        filterCounts = [16, 32, 64][:numBlocks]

        layers = []
        currentChannels = channelsIn

        for filters in filterCounts:
            layers.extend([
                nn.Conv1d(currentChannels, filters, 5),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),

            ])

            currentChannels = filters

        layers.append(nn.Flatten())

        self.layers = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, channelsIn, inputLength)
            self.outputDimension = self.layers(dummy).numel()

    def forward(self, x):
        return self.layers(x)

class TransitClassifier(nn.Module):
    def __init__(self, scalarDimension = 12, numLabels = 1, dropout = 0.3):
        super().__init__()

        # global view has 4 channels: median, std, transitFlag, hasData
        self.globalTower = ConvolutionTower(4, globalBins, numBlocks = 3)

        # local and secondary views have 2 channels: median, std
        self.localTower = ConvolutionTower(2, localBins, numBlocks = 3)
        self.secondaryTower = ConvolutionTower(2, secondaryBins, numBlocks = 3)

        fullyConnectedInput = scalarDimension + self.globalTower.outputDimension + self.localTower.outputDimension + self.secondaryTower.outputDimension

        self.fullyConnected = nn.Sequential(
            # input layer
            nn.Linear(fullyConnectedInput, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # hidden layer 1
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # hidden layer 2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # output layer - single logit for binary E classification
            nn.Linear(64, numLabels),

        )

    def forward(self, batch):
        globalView = batch["globalView"]
        localView = batch["localView"]
        secondaryView = batch["secondaryView"]
        scalars = batch["scalars"]

        globalFeatures = self.globalTower(globalView)
        localFeatures = self.localTower(localView)
        secondaryFeatures = self.secondaryTower(secondaryView)

        vector = torch.cat([globalFeatures, localFeatures, secondaryFeatures, scalars], dim = 1)

        output = self.fullyConnected(vector)

        # squeeze from (batch, 1) to (batch,) for BCEWithLogitsLoss
        return output.squeeze(1)

if __name__ == "__main__":
    transitClassifier = TransitClassifier()

    dummyGlobalView = torch.zeros(4, 4, globalBins)
    dummyLocalView = torch.zeros(4, 2, localBins)
    dummySecondaryView = torch.zeros(4, 2, secondaryBins)
    dummyScalars = torch.zeros(4, 12)

    dummyBatch = {
        "globalView": dummyGlobalView,
        "localView": dummyLocalView,
        "secondaryView": dummySecondaryView,
        "scalars": dummyScalars,

    }

    output = transitClassifier(dummyBatch)
    print(f"Output shape: {output.shape}")

    assert output.shape == torch.Size([4]), f"Expected (4,), got {output.shape}"
    print("All checks passed")
