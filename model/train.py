import argparse
import torch

from model import TransitClassifier
from dataset import TransitDataset, makeSplits

from pathlib import Path

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

lossThreshold = 100.0


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train the TransitClassifier model")

    parser.add_argument("--data", type = Path, default = defaultDataPath,
        help = "Path to dataset.h5 (default: data/processed/dataset.h5)")
    parser.add_argument("--scalars", type = Path, default = defaultScalarsPath,
        help = "Path to scalar_stats.json (default: data/processed/scalar_stats.json)")
    parser.add_argument("--epochs", type = int, default = 20,
        help = "Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = 8,
        help = "DataLoader worker count for training (default: 8)")
    parser.add_argument("--lr", type = float, default = 1e-4,
        help = "Learning rate (default: 1e-4)")

    return parser.parse_args()


def main():
    args = parseArgs()

    print("Loading data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split the dataset into train / validation / test indices (80 / 10 / 10)
    splits = makeSplits(args.data)

    trainingIndices, validationIndices, testIndices = splits[0], splits[1], splits[2]

    trainingDataset = TransitDataset(args.data, args.scalars, trainingIndices)
    validationDataset = TransitDataset(args.data, args.scalars, validationIndices)
    testDataset = TransitDataset(args.data, args.scalars, testIndices)

    print("Building data loaders...")
    # sampler oversamples the minority class so batches are roughly balanced
    # replaces shuffle=True since the sampler handles randomisation
    trainSampler = torch.utils.data.WeightedRandomSampler(
        weights = trainingDataset.sampleWeights,
        num_samples = len(trainingDataset),
        replacement = True,

    )

    trainingLoader = torch.utils.data.DataLoader(
        trainingDataset, batch_size = args.batch_size, sampler = trainSampler,
        num_workers = args.workers, pin_memory = True, persistent_workers = True,

    )

    validationLoader = torch.utils.data.DataLoader(
        validationDataset, batch_size = args.batch_size, shuffle = False,
        num_workers = max(args.workers // 2, 1), pin_memory = True, persistent_workers = True,

    )

    testLoader = torch.utils.data.DataLoader(
        testDataset, batch_size = args.batch_size, shuffle = False,
        num_workers = max(args.workers // 4, 1), pin_memory = True, persistent_workers = True,

    )

    print("Building model...")
    model = TransitClassifier().to(device)

    # punish the model for predicting false when its actually true to try to deal with data imbalance
    counts = trainingDataset.labelCounts
    positiveHits = counts["positive"]
    negativeHits = counts["negative"]

    # take sqrt of ratio to reduce agressiveness
    positiveMissWeight = torch.tensor((negativeHits / positiveHits) ** 0.5).to(device)

    criteria = torch.nn.BCEWithLogitsLoss(pos_weight = positiveMissWeight).to(device)

    # "Adam" optimizer adjusts the learning rate per-weight based on gradient history
    # weight_decay adds a penalty for large weights to discourage overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0.0001)

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # training
        # model.train() enables dropout so the model is regularised during training
        model.train()

        trainingLoss = 0.0
        batchesSkipped = 0

        for batch in trainingLoader:
            batch = {key: value.to(device) for key, value in batch.items()}

            # clear gradients from the previous batch before computing new ones
            optimizer.zero_grad()

            predictions = model(batch)

            loss = criteria(predictions, batch["label"])

            # skip batches with non-finite or extremely large losses
            if not torch.isfinite(loss) or loss.item() > lossThreshold:
                batchesSkipped += 1

                continue

            # backward pass computes how much each weight contributed to the loss
            loss.backward()

            # maximum gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update weights using the gradients just computed
            optimizer.step()

            trainingLoss += loss.item()

        # average loss across all batches that were actually used
        batchesUsed = len(trainingLoader) - batchesSkipped
        trainingLoss /= max(batchesUsed, 1)

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

        summary = f"Epoch {epoch + 1}: Training loss {trainingLoss:.4f} | Validation loss {validationLoss:.4f}"

        if batchesSkipped > 0:
            summary += f" | {batchesSkipped} batch(es) skipped: extraneous loss"

        print(summary)

if __name__ == "__main__":
    main()
