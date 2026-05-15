import torch

from model import TransitClassifier
from dataset import TransitDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TransitDataset("data/processed/dataset.h5", "data/processed/scalar_stats.json")

loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)

model = TransitClassifier().to(device)
criteria = torch.nn.BCEWithLogitsLoss()

batch = next(iter(loader))
batch = {key: value.to(device) for key, value in batch.items()}

print("scalars NaN:", batch["scalars"].isnan().any())
print("globalView NaN:", batch["globalView"].isnan().any())
print("label NaN:", batch["label"].isnan().any())

predictions = model(batch)
loss = criteria(predictions, batch["label"])

print(f"Output shape: {predictions.shape}")
print(f"Loss: {loss.item()}")