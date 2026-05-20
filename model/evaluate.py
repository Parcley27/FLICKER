import argparse
import re
import numpy as np
import torch

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    recall_score, precision_score, average_precision_score,
    precision_recall_curve, confusion_matrix, f1_score,
)

from network import TransitClassifier
from dataset import TransitDataset, makeSplits

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

    logits = []
    labels = []

    with torch.no_grad():
        for batch in testLoader:
            batch = {key: value.to(device) for key, value in batch.items()}

            predictions = model(batch)

            logits.append(predictions.detach().cpu())
            labels.append(batch["label"].detach().cpu())

    logits = torch.cat(logits)
    labels = torch.cat(labels)

    probabilities = torch.sigmoid(logits).numpy()
    labels = labels.numpy()

    auPRc = average_precision_score(labels, probabilities)
    precision = precision_score(labels, (probabilities >= 0.5).astype(int), zero_division = 0)
    recall = recall_score(labels, (probabilities >= 0.5).astype(int), zero_division = 0)

    binaryPredictions = (probabilities >= 0.5).astype(int)
    f1 = f1_score(labels, binaryPredictions, zero_division = 0)

    outputLines = []

    def printAndLog(line = ""):
        print(line)
        outputLines.append(line)

    # summary
    printAndLog(f"Checkpoint: {checkpoint}")
    printAndLog(f"Test samples: {len(labels)}")

    positiveCount = int(labels.sum())
    negativeCount = len(labels) - positiveCount

    printAndLog(f"  Positive: {positiveCount}  Negative: {negativeCount}")
    printAndLog(f"\nAUC-PR: {auPRc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # confusion matrix at 0.5
    cm = confusion_matrix(labels, binaryPredictions)
    tn, fp, fn, tp = cm.ravel()

    printAndLog(f"\nConfusion Matrix (threshold = 0.5)")
    printAndLog(f"                 Predicted Neg  Predicted Pos")
    printAndLog(f"  Actual Neg     {tn:>12}  {fp:>12}")
    printAndLog(f"  Actual Pos     {fn:>12}  {tp:>12}")
    printAndLog()
    printAndLog(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")

    # threshold sweep at 0.05 increments
    printAndLog(f"\nThreshold | Precision |  Recall  |    F1    |   TP |   FP |   FN |   TN")
    printAndLog(f"----------+-----------+----------+----------+------+------+------+------")

    for threshold in np.arange(0.05, 1.0, 0.05):
        threshBinary = (probabilities >= threshold).astype(int)
        threshPrecision = precision_score(labels, threshBinary, zero_division = 0)
        threshRecall = recall_score(labels, threshBinary, zero_division = 0)
        threshF1 = f1_score(labels, threshBinary, zero_division = 0)

        threshCm = confusion_matrix(labels, threshBinary, labels = [0, 1])
        ttn, tfp, tfn, ttp = threshCm.ravel()

        printAndLog(f"  {threshold:.2f}    |  {threshPrecision:.4f}   | {threshRecall:.4f}   | {threshF1:.4f}   | {ttp:>4} | {tfp:>4} | {tfn:>4} | {ttn:>4}")

    # precision-recall curve
    prPrecision, prRecall, _ = precision_recall_curve(labels, probabilities)

    figure, axis = plt.subplots(figsize = (6, 5))

    axis.plot(prRecall, prPrecision, linewidth = 2)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Precision-Recall Curve (AUC-PR = {auPRc:.4f})")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha = 0.3)

    figure.tight_layout()

    # save results
    resultsPath.mkdir(parents = True, exist_ok = True)

    # extract the timestamp from the checkpoint filename (e.g. best_20260520_143022.pt)
    timestampMatch = re.search(r"\d{8}_\d{6}", checkpoint.stem)
    timestamp = timestampMatch.group() if timestampMatch else "unknown"
    outputFile = resultsPath / f"eval_{timestamp}.txt"

    with open(outputFile, "w") as f:
        f.write("\n".join(outputLines) + "\n")

    prCurvePath = resultsPath / f"pr_curve_{timestamp}.png"
    figure.savefig(prCurvePath, dpi = 150)
    plt.close(figure)

    print(f"\nResults saved to {outputFile}")
    print(f"PR curve saved to {prCurvePath}")

if __name__ == "__main__":
    main()
