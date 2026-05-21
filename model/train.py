import argparse
import datetime
import numpy as np
import os
import torch
import torch.nn.functional as F

from pathlib import Path
from sklearn.metrics import average_precision_score

from network import TransitClassifier
from dataset import TransitDataset, makeSplits
from config import defaultSteps, defaultValInterval, defaultBatchSize, defaultWorkers, defaultLR, numClasses

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

checkpointPath = repoRoot / "model" / "checkpoints"

lossThreshold = 10.0
focalGamma = 2.0

def focalLoss(logits, targets, classWeights, gamma = focalGamma):
    perSampleLoss = F.cross_entropy(logits, targets, weight = classWeights, reduction = "none")
    probs = torch.softmax(logits, dim = 1)
    trueClassProbs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focalWeight = (1 - trueClassProbs) ** gamma
    
    return (focalWeight * perSampleLoss).mean()

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train the TransitClassifier model")

    parser.add_argument("--data", type = Path, default = defaultDataPath,
        help = "Path to dataset.h5 (default: data/processed/dataset.h5)")
    parser.add_argument("--scalars", type = Path, default = defaultScalarsPath,
        help = "Path to scalar_stats.json (default: data/processed/scalar_stats.json)")
    parser.add_argument("--steps", type = int, default = defaultSteps,
        help = "Total number of gradient steps (default: 5000)")
    parser.add_argument("--val-interval", type = int, default = defaultValInterval,
        help = "Validate every N steps (default: 500)")
    parser.add_argument("--batch-size", type = int, default = defaultBatchSize,
        help = "Batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = defaultWorkers,
        help = "DataLoader worker count for training (default: 8)")
    parser.add_argument("--lr", type = float, default = defaultLR,
        help = "Learning rate (default: 1e-3)")
    parser.add_argument("--reset-step", type = int, default = 0,
        help = "Reset optimizer state at this step (0 = disabled)")

    return parser.parse_args()

def trainModel(args) -> tuple[dict | None, float | float]:
    print("Loading data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split the dataset using pre-assigned train / val / test attributes from the HDF5 file
    splits = makeSplits(args.data)

    trainingIndices, validationIndices, _ = splits[0], splits[1], splits[2]

    trainingDataset = TransitDataset(args.data, args.scalars, trainingIndices, augment = True)
    validationDataset = TransitDataset(args.data, args.scalars, validationIndices)

    print("Building data loaders...")
    counts = trainingDataset.labelCounts

    trainingLoader = torch.utils.data.DataLoader(
        trainingDataset, batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers, pin_memory = True, persistent_workers = True,

    )

    validationLoader = torch.utils.data.DataLoader(
        validationDataset, batch_size = args.batch_size, shuffle = False,
        num_workers = max(args.workers // 2, 1), pin_memory = True, persistent_workers = True,

    )

    print("Building model...")
    model = TransitClassifier().to(device)

    # class weights from inverse frequency, sqrt-softened to avoid over-correcting
    totalSamples = sum(counts)
    classWeights = torch.tensor(
        [(totalSamples / max(counts[i], 1)) ** 0.75 for i in range(numClasses)],
        dtype = torch.float32,
    ).to(device)

    # Adam adjusts the learning rate per-weight based on gradient history
    # weight_decay adds a penalty for large weights to discourage overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.steps, eta_min = 1e-5)

    print("Creating checkpoint directory...")
    os.makedirs(checkpointPath, exist_ok = True)

    bestScore = 0.0
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

            loss = focalLoss(predictions, batch["label"], classWeights)

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
            scheduler.step()

            intervalLoss += loss.item()
            intervalBatchesUsed += 1
            step += 1

            if args.reset_step > 0 and step == args.reset_step:
                optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0.001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.steps - step, eta_min = 1e-6)
                
                print(f"Optimizer reset at step {step}")

            if step % args.val_interval == 0 or step >= args.steps:
                avgTrainingLoss = intervalLoss / max(intervalBatchesUsed, 1)

                # validation
                # model.eval() disables dropout so predictions are deterministic
                model.eval()

                validationLoss = 0.0
                logits = []
                labels = []

                # no_grad skips gradient tracking since we're not updating weights here
                with torch.no_grad():
                    for valBatch in validationLoader:
                        valBatch = {key: value.to(device) for key, value in valBatch.items()}

                        valPredictions = model(valBatch)
                        valLoss = focalLoss(valPredictions, valBatch["label"], classWeights)

                        logits.append(valPredictions.detach().cpu())
                        labels.append(valBatch["label"].detach().cpu())

                        validationLoss += valLoss.item()

                validationLoss /= len(validationLoader)

                logits = torch.cat(logits)
                labels = torch.cat(labels)

                softmaxProbs = torch.softmax(logits, dim = 1).numpy()
                trueLabels = labels.numpy()

                eAuPRc = average_precision_score((trueLabels == 0).astype(int), softmaxProbs[:, 0])

                if eAuPRc > bestScore:
                    bestScore = eAuPRc
                    bestStateDict = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}

                summary = f"Step {step}: Training loss {avgTrainingLoss:.4f} | Validation loss {validationLoss:.4f} | E AUC-PR {eAuPRc:.4f} | Best {bestScore:.4f} | LR {optimizer.param_groups[0]['lr']:.1e}"

                if intervalBatchesSkipped > 0:
                    summary += f" | {intervalBatchesSkipped} batch(es) skipped"

                print(summary)

                # reset interval accumulators and return to training mode
                intervalLoss = 0.0
                intervalBatchesUsed = 0
                intervalBatchesSkipped = 0

                model.train()
    
    return bestStateDict, bestScore # type: ignore

def main():
    args = parseArgs()

    bestStateDict, bestScore = trainModel(args)

    if bestStateDict is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(bestStateDict, checkpointPath / f"best_{timestamp}.pt")
        print(f"Saved best model (Macro-F1 {bestScore:.4f}) to checkpoints/best_{timestamp}.pt")
        
    else:
        print("No valid model was found during training.")

if __name__ == "__main__":
    main()
