import torch
import torch.nn as nn

class ConvolutionTower(nn.Module):
    def __init__(self, channelsIn, inputLength):
        
        super().__init__()

        # layer(in, out, kernel)
        # 16 32 64 is a standard starting point
        # 5 wide to look at 5 adj time steps at once
        self.layers = nn.Sequential(
            # first layer (input)
            nn.Conv1d(channelsIn, 16, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # second (first hidden)
            nn.Conv1d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # third (second hidden)
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # flatten (output layer)
            nn.Flatten(),
        
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channelsIn, inputLength)
            self.outputDim = self.layers(dummy).numel()
        
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # consider global view for example
    # 3 light channels, 201 phase bins
    globalTower = ConvolutionTower(3, 201)

    print(globalTower)
    print(globalTower.outputDim)