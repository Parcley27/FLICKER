import sys
from pathlib import Path

import torch
import torch.nn as nn

repoRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repoRoot))

from config import globalBins, localBins, secondaryBins, halfPeriodBins, oddEvenBins, numClasses

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
    def __init__(self, scalarDimension = 12, numLabels = numClasses, dropout = 0.5):
        super().__init__()

        # global view has 4 channels: median, std, transitFlag, hasData
        self.globalTower = ConvolutionTower(4, globalBins, numBlocks = 3)

        # local, secondary, and half-period views have 2 channels: median, std
        self.localTower = ConvolutionTower(2, localBins, numBlocks = 3)
        self.secondaryTower = ConvolutionTower(2, secondaryBins, numBlocks = 3)
        self.halfPeriodTower = ConvolutionTower(2, halfPeriodBins, numBlocks = 3)

        # odd/even view has 4 channels: odd_median, odd_std, even_median, even_std
        self.oddEvenTower = ConvolutionTower(4, oddEvenBins, numBlocks = 3)

        fullyConnectedInput = scalarDimension + self.globalTower.outputDimension + self.localTower.outputDimension + self.secondaryTower.outputDimension + self.halfPeriodTower.outputDimension + self.oddEvenTower.outputDimension

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

            # output layer - one logit per class (E, S, B, J, N)
            nn.Linear(64, numLabels),

        )

    def forward(self, batch):
        globalView = batch["globalView"]
        localView = batch["localView"]
        secondaryView = batch["secondaryView"]
        halfPeriodView = batch["halfPeriodView"]
        oddEvenView = batch["oddEvenView"]
        scalars = batch["scalars"]

        globalFeatures = self.globalTower(globalView)
        localFeatures = self.localTower(localView)
        secondaryFeatures = self.secondaryTower(secondaryView)
        halfPeriodFeatures = self.halfPeriodTower(halfPeriodView)
        oddEvenFeatures = self.oddEvenTower(oddEvenView)

        vector = torch.cat([globalFeatures, localFeatures, secondaryFeatures, halfPeriodFeatures, oddEvenFeatures, scalars], dim = 1)

        output = self.fullyConnected(vector)

        return output

if __name__ == "__main__":
    transitClassifier = TransitClassifier()

    dummyGlobalView = torch.zeros(4, 4, globalBins)
    dummyLocalView = torch.zeros(4, 2, localBins)
    dummySecondaryView = torch.zeros(4, 2, secondaryBins)
    dummyHalfPeriodView = torch.zeros(4, 2, halfPeriodBins)
    dummyOddEvenView = torch.zeros(4, 4, oddEvenBins)
    dummyScalars = torch.zeros(4, 12)

    dummyBatch = {
        "globalView": dummyGlobalView,
        "localView": dummyLocalView,
        "secondaryView": dummySecondaryView,
        "halfPeriodView": dummyHalfPeriodView,
        "oddEvenView": dummyOddEvenView,
        "scalars": dummyScalars,

    }

    output = transitClassifier(dummyBatch)
    print(f"Output shape: {output.shape}")

    assert output.shape == torch.Size([4, numClasses]), f"Expected (4, {numClasses}), got {output.shape}"
    print("All checks passed")
