import argparse
import datetime
import numpy as np
import os
import torch

from pathlib import Path

from train import trainModel
from evaluate import runInference, computeMetrics
from network import TransitClassifier
from dataset import TransitDataset, makeSplits

from config import defaultSteps, defaultValInterval, defaultBatchSize, defaultWorkers, defaultLR

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

runsPath = repoRoot / "model" / "runs"

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train and compare multiple models")

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
        help = "Learning rate (default: 1e-4)")
    parser.add_argument("--model-count", type = int, default = 10,
        help = "Number of models to train (default: 5)")
    parser.add_argument("--seed", type = int, default = 27,
        help = "Random seed (default: 27)")

    return parser.parse_args()

def main():
    args = parseArgs()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runDir = runsPath / timestamp
    os.makedirs(runDir, exist_ok = True)

    summaryLines = []

    for i in range(args.model_count):
        seed = args.seed + i

        print(f"\n{'=' * 60}")
        print(f"Training model {i + 1}/{args.model_count} (seed {seed})")
        print(f"{'=' * 60}\n")

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        trainingArgs = argparse.Namespace(
            data = args.data,
            scalars = args.scalars,
            steps = args.steps,
            val_interval = args.val_interval,
            batch_size = args.batch_size,
            workers = args.workers,
            lr = args.lr,

        )

        bestStateDict, bestScore = trainModel(trainingArgs)

        if bestStateDict is not None:
            modelPath = runDir / f"model_{i}.pt"
            torch.save(bestStateDict, modelPath)
            summaryLines.append(f"Model {i}: seed {seed} | Macro-F1 {bestScore:.4f} | {modelPath.name}")
            print(f"Saved model_{i}.pt (Macro-F1 {bestScore:.4f})")

        else:
            summaryLines.append(f"Model {i}: seed {seed} | FAILED")
            print(f"Model {i} failed to produce a valid checkpoint")

    summaryPath = runDir / "summary.txt"
    summaryPath.write_text("\n".join(summaryLines) + "\n")

    # ensemble evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("Evaluating ensemble on test set...")
    print(f"{'=' * 60}\n")

    splits = makeSplits(args.data)
    testIndices = splits[2]
    testDataset = TransitDataset(args.data, args.scalars, testIndices)

    testLoader = torch.utils.data.DataLoader(
        testDataset, batch_size = args.batch_size, shuffle = False,
        num_workers = 2, pin_memory = True, persistent_workers = True,

    )

    allProbabilities = []
    labels = None

    for checkpointFile in sorted(runDir.glob("model_*.pt")):
        model = TransitClassifier().to(device)
        model.load_state_dict(torch.load(checkpointFile, map_location = device, weights_only = True))
        model.eval()

        probabilities, labels = runInference(model, testLoader, device)
        allProbabilities.append(probabilities)

        print(f"{checkpointFile.name} inference complete")
    
    ensembleProbabilities = np.mean(allProbabilities, axis = 0)
    computeMetrics(ensembleProbabilities, labels, runDir, "ensemble")

    print(f"\n{'=' * 60}")
    print(f"Choir done - {args.model_count} models saved to {runDir}")
    print("\n".join(summaryLines))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()