import argparse
import datetime
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
    parser.add_argument("--steps", type = int, default = 20000,
        help = "Total number of gradient steps (default: 20000)")
    parser.add_argument("--val-interval", type = int, default = 500,
        help = "Validate every N steps (default: 500)")
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
    counts = trainingDataset.labelCounts

    # each positive is seen ~2x per sampler epoch, negatives are undersampled
    # this reduces how aggressively the sampler inflates the positive prior
    # compared to num_samples = len(dataset) which draws ~9,960 positives per epoch
    samplerNumSamples = 4 * counts["positive"]

    # sampler oversamples minority classes so batches are roughly balanced
    # replaces shuffle=True since the sampler handles randomisation
    trainSampler = torch.utils.data.WeightedRandomSampler(
        weights = trainingDataset.sampleWeights,
        num_samples = samplerNumSamples,
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

    # sqrt of inverse frequency ratio softens the imbalance correction
    posWeight = torch.tensor([math.sqrt(counts["negative"] / counts["positive"])], dtype = torch.float32).to(device)
    criteria = torch.nn.BCEWithLogitsLoss(pos_weight = posWeight).to(device)

    # Adam adjusts the learning rate per-weight based on gradient history
    # weight_decay adds a penalty for large weights to discourage overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0.0001)

    print("Creating checkpoint directory...")
    os.makedirs(checkpointPath, exist_ok = True)

    bestAuPRc = 0.0
    bestStateDict = None

    step = 0
    intervalLoss = 0.0
    intervalBatchesUsed = 0
    intervalBatchesSkipped = 0

    print(f"Starting training for {args.steps} steps (validating every {args.val_interval} steps)...")

    # model.train() enables dropout so the model is regularised during training
    model.train()

    while step < args.steps:
        for batch in trainingLoader:
            if step >= args.steps:
                break

            batch = {key: value.to(device) for key, value in batch.items()}

            # clear gradients from the previous batch before computing new ones
            optimizer.zero_grad()

            predictions = model(batch)

            loss = criteria(predictions, batch["label"])

            # skip batches with non-finite loss
            if not torch.isfinite(loss) or loss.item() > lossThreshold:
                intervalBatchesSkipped += 1
                step += 1

                continue

            # backward pass computes how much each weight contributed to the loss
            loss.backward()

            # clip maximum gradient norm to stabilise training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update weights using the gradients just computed
            optimizer.step()

            intervalLoss += loss.item()
            intervalBatchesUsed += 1
            step += 1

            if step % args.val_interval == 0 or step >= args.steps:
                avgTrainingLoss = intervalLoss / max(intervalBatchesUsed, 1)

                # validation — model.eval() disables dropout so predictions are deterministic
                model.eval()

                validationLoss = 0.0
                logits = []
                labels = []

                # no_grad skips gradient tracking since we're not updating weights here
                with torch.no_grad():
                    for valBatch in validationLoader:
                        valBatch = {key: value.to(device) for key, value in valBatch.items()}

                        valPredictions = model(valBatch)
                        valLoss = criteria(valPredictions, valBatch["label"])

                        logits.append(valPredictions.detach().cpu())
                        labels.append(valBatch["label"].detach().cpu())

                        validationLoss += valLoss.item()

                validationLoss /= len(validationLoader)

                logits = torch.cat(logits)
                labels = torch.cat(labels)

                probabilities = torch.sigmoid(logits).numpy()
                trueLabels = labels.numpy()

                auPRc = average_precision_score(trueLabels, probabilities)

                if auPRc > bestAuPRc:
                    bestAuPRc = auPRc
                    bestStateDict = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}

                summary = f"Step {step}: Training loss {avgTrainingLoss:.4f} | Validation loss {validationLoss:.4f} | AUC-PR {auPRc:.4f} | Best {bestAuPRc:.4f} | LR {args.lr:.1e}"

                if intervalBatchesSkipped > 0:
                    summary += f" | {intervalBatchesSkipped} batch(es) skipped"

                print(summary)

                # reset interval accumulators and return to training mode
                intervalLoss = 0.0
                intervalBatchesUsed = 0
                intervalBatchesSkipped = 0

                model.train()

    if bestStateDict is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(bestStateDict, checkpointPath / f"best_{timestamp}.pt")
        print(f"Saved best model (AUC-PR {bestAuPRc:.4f}) to checkpoints/best_{timestamp}.pt")
        
    else:
        print("No valid model was found during training.")

if __name__ == "__main__":
    main()
