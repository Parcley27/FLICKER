import argparse
import os
import torch

from pathlib import Path
from sklearn.metrics import average_precision_score

from network import TransitClassifier
from dataset import TransitDataset, makeSplits

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

checkpointPath = repoRoot / "model" / "checkpoints"

lossThreshold = 10.0

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

    # split the dataset using pre-assigned train / val / test attributes from the HDF5 file
    splits = makeSplits(args.data)

    trainingIndices, validationIndices, _ = splits[0], splits[1], splits[2]

    trainingDataset = TransitDataset(args.data, args.scalars, trainingIndices, augment = True)
    validationDataset = TransitDataset(args.data, args.scalars, validationIndices)

    print("Building data loaders...")
    # sampler oversamples minority classes so batches are roughly balanced
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

    print("Building model...")
    model = TransitClassifier().to(device)

    # no pos_weight since the WeightedRandomSampler already handles class imbalance
    criteria = torch.nn.BCEWithLogitsLoss().to(device)

    # Adam adjusts the learning rate per-weight based on gradient history
    # weight_decay adds a penalty for large weights to discourage overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0.0001)

    # reduce learning rate when validation AUC-PR plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = "max", patience = 4, factor = 0.5,

    )

    print("Creating checkpoint directory...")
    os.makedirs(checkpointPath, exist_ok = True)

    bestAuPRc = 0.0

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

            # skip batches with non-finite loss
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
        logits = []
        labels = []

        # no_grad skips gradient tracking since we're not updating weights here
        with torch.no_grad():
            for batch in validationLoader:
                batch = {key: value.to(device) for key, value in batch.items()}

                predictions = model(batch)
                loss = criteria(predictions, batch["label"])

                logits.append(predictions.detach().cpu())
                labels.append(batch["label"].detach().cpu())

                validationLoss += loss.item()

            validationLoss /= len(validationLoader)

        logits = torch.cat(logits)
        labels = torch.cat(labels)

        probabilities = torch.sigmoid(logits).numpy()
        trueLabels = labels.numpy()

        auPRc = average_precision_score(trueLabels, probabilities)

        # step the scheduler based on validation AUC-PR
        scheduler.step(auPRc)

        if auPRc > bestAuPRc:
            bestAuPRc = auPRc

            torch.save(model.state_dict(), checkpointPath / "best.pt")

        currentLR = optimizer.param_groups[0]["lr"]
        summary = f"Epoch {epoch + 1}: Training loss {trainingLoss:.4f} | Validation loss {validationLoss:.4f} | AUC-PR {auPRc:.4f} | Best {bestAuPRc:.4f} | LR {currentLR:.1e}"

        if batchesSkipped > 0:
            summary += f" | {batchesSkipped} batch(es) skipped"

        print(summary)

if __name__ == "__main__":
    main()
