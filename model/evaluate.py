import argparse
import re
import numpy as np
import torch

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score, balanced_accuracy_score,
    classification_report,
)

from network import TransitClassifier
from dataset import TransitDataset, makeSplits
from config import classNames

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

checkpointPath = repoRoot / "model" / "checkpoints"
resultsPath = repoRoot / "model" / "results"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Evaluate the TransitClassifier model on the test set")

    parser.add_argument("--data", type = Path, default = defaultDataPath,
        help = "Path to dataset.h5 (default: data/processed/dataset.h5)")
    parser.add_argument("--scalars", type = Path, default = defaultScalarsPath,
        help = "Path to scalar_stats.json (default: data/processed/scalar_stats.json)")
    parser.add_argument("--checkpoint", type = Path, default = None,
        help = "Path to model checkpoint (.pt file). Defaults to most recent in model/checkpoints/")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = 4,
        help = "DataLoader worker count (default: 4)")

    return parser.parse_args()

def runInference(model, dataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    labels = []

    with torch.no_grad():
        for batch in dataLoader:
            batch = {key: value.to(device) for key, value in batch.items()}

            predictions = model(batch)

            logits.append(predictions.detach().cpu())
            labels.append(batch["label"].detach().cpu())

    logits = torch.cat(logits)
    labels = torch.cat(labels)

    probabilities = torch.softmax(logits, dim = 1).numpy()
    labels = labels.numpy()

    return probabilities, labels

def computeMetrics(probabilities, labels, outputDir, tag = None):
    # probabilities shape: (n_samples, numClasses), labels shape: (n_samples,)
    predictions = np.argmax(probabilities, axis = 1)

    macroF1 = f1_score(labels, predictions, average = "macro", zero_division = 0)
    balancedAcc = balanced_accuracy_score(labels, predictions)

    outputLines = []

    def printAndLog(line = ""):
        print(line)
        outputLines.append(line)

    # summary
    printAndLog(f"Evaluation: {tag}")
    printAndLog(f"Test samples: {len(labels)}")

    for i, name in enumerate(classNames):
        printAndLog(f"  {name}: {int((labels == i).sum())}")

    printAndLog(f"\nMacro-F1: {macroF1:.4f} | Balanced Accuracy: {balancedAcc:.4f}")

    # per-class metrics
    report = classification_report(labels, predictions, target_names = classNames, zero_division = 0)
    printAndLog(f"\n{report}")

    # confusion matrix
    cm = confusion_matrix(labels, predictions, labels = list(range(len(classNames))))

    printAndLog("Confusion Matrix (rows = actual, columns = predicted)")
    header = "        " + "  ".join(f"{name:>5}" for name in classNames)
    printAndLog(header)

    for i, name in enumerate(classNames):
        row = "  ".join(f"{cm[i, j]:>5}" for j in range(len(classNames)))
        printAndLog(f"  {name:>4}  {row}")

    # exoplanet (E) one-vs-rest PR curve
    eBinary = (labels == 0).astype(int)
    eProbs = probabilities[:, 0]
    eAuPRc = average_precision_score(eBinary, eProbs)

    prPrecision, prRecall, _ = precision_recall_curve(eBinary, eProbs)

    printAndLog(f"\nExoplanet (E) one-vs-rest AUC-PR: {eAuPRc:.4f}")

    figure, axis = plt.subplots(figsize = (6, 5))

    axis.plot(prRecall, prPrecision, linewidth = 2)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Exoplanet PR Curve (AUC-PR = {eAuPRc:.4f})")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha = 0.3)

    figure.tight_layout()

    # save results
    outputDir = Path(outputDir)
    outputDir.mkdir(parents = True, exist_ok = True)

    outputFile = outputDir / f"eval_{tag}.txt"

    with open(outputFile, "w") as f:
        f.write("\n".join(outputLines) + "\n")

    prCurvePath = outputDir / f"pr_curve_{tag}.png"
    figure.savefig(prCurvePath, dpi = 150)
    plt.close(figure)

    print(f"\nResults saved to {outputFile}")
    print(f"PR curve saved to {prCurvePath}")

def main():
    args = parseArgs()

    checkpoint = args.checkpoint

    if checkpoint is None:
        # find the most recent best_*.pt checkpoint by timestamp in filename
        candidates = sorted(checkpointPath.glob("best_*.pt"))

        if not candidates:
            print(f"No checkpoints found in {checkpointPath}")
            print("Run train.py first or specify --checkpoint")

            return

        checkpoint = candidates[-1]

    print(f"Using checkpoint {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building model...")
    model = TransitClassifier().to(device)

    model.load_state_dict(torch.load(checkpoint, map_location = device, weights_only = True))

    model.eval()

    print("Building data loader...")
    splits = makeSplits(args.data)

    testIndices = splits[2]

    testDataset = TransitDataset(args.data, args.scalars, testIndices)

    persistWorkers = args.workers > 0

    testLoader = torch.utils.data.DataLoader(
        testDataset, batch_size = args.batch_size, shuffle = False,
        num_workers = args.workers, pin_memory = True, persistent_workers = persistWorkers,

    )

    probabilities, labels = runInference(model, testLoader, device)

    timestampMatch = re.search(r"\d{8}_\d{6}", checkpoint.stem)
    tag = timestampMatch.group() if timestampMatch else "unknown"

    computeMetrics(probabilities, labels, resultsPath, tag)

    

if __name__ == "__main__":
    main()
