import torch

from model import TransitClassifier
from dataset import TransitDataset, makeSplits

dataPath = "data/processed/dataset.h5"
scalarsPath = "data/processed/scalar_stats.json"

epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# split the dataset into train / validation / test indices (80 / 10 / 10)
splits = makeSplits(dataPath)

trainingIndices, validationIndices, testIndices = splits[0], splits[1], splits[2]

trainingDataset = TransitDataset(dataPath, scalarsPath, trainingIndices)
validationDataset = TransitDataset(dataPath, scalarsPath, validationIndices)
testDataset = TransitDataset(dataPath, scalarsPath, testIndices)

# shuffling the dataset doesn't do anything for evaluation, but useful for training to avoid overfitting
trainingLoader = torch.utils.data.DataLoader(trainingDataset, batch_size = 64, shuffle = True)
validationLoader = torch.utils.data.DataLoader(validationDataset, batch_size = 64, shuffle = False)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size = 64, shuffle = False)

model = TransitClassifier().to(device)

# BCEWithLogitsLoss expects raw logits (no sigmoid) and handles the binary classification loss
criteria = torch.nn.BCEWithLogitsLoss().to(device)

# "Adam" optimizer adjusts the learning rate per-weight based on gradient history
# weight_decay adds a penalty for large weights to discourage overfitting
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.0001)

for epoch in range(epochs):
    # training
    # model.train() enables dropout so the model is regularised during training
    model.train()

    trainingLoss = 0.0

    for batch in trainingLoader:
        batch = {key: value.to(device) for key, value in batch.items()}

        # clear gradients from the previous batch before computing new ones
        optimizer.zero_grad()

        predictions = model(batch)

        loss = criteria(predictions, batch["label"])

        # backward pass computes how much each weight contributed to the loss
        loss.backward()

        # update weights using the gradients just computed
        optimizer.step()

        trainingLoss += loss.item()

    # average loss across all batches in the epoch
    trainingLoss /= len(trainingLoader)

    # validation
    # model.eval() disables dropout so predictions are deterministic
    model.eval()

    validationLoss = 0.0

    # no_grad skips gradient tracking since we're not updating weights here
    with torch.no_grad():
        for batch in validationLoader:
            batch = {key: value.to(device) for key, value in batch.items()}

            predictions = model(batch)
            loss = criteria(predictions, batch["label"])

            validationLoss += loss.item()

        validationLoss /= len(validationLoader)

    print(f"Epoch {epoch + 1}: Training loss {trainingLoss:.4f} | Validation loss {validationLoss:.4f}")
