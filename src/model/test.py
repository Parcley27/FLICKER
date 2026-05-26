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
    confusion_matrix, f1_score, fbeta_score, balanced_accuracy_score,
    classification_report,
    
)

from network import TransitClassifier
from dataset import TransitDataset, makeSplits
from config import classNames, recallBeta

repoRoot = Path(__file__).resolve().parents[2]
defaultDataPath = repoRoot / "src" / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "src" / "data" / "processed" / "scalar_stats.json"

checkpointPath = repoRoot / "src" / "model" / "checkpoints"
resultsPath = repoRoot / "src" / "model" / "results"


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
    parser.add_argument("--e-threshold", type = float, default = None,
        help = "Probability threshold for predicting E (default: use argmax). "
               "Lower values increase recall at the cost of precision. "
               "Pass --e-threshold auto to find the threshold that maximises F2 while achieving >=90%% recall.")
    parser.add_argument("--find-threshold", action = "store_true",
        help = "Scan thresholds and report recall/precision at each; useful for picking --e-threshold.")

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

def findEThreshold(probabilities, labels, targetRecall = 0.90):
    """Scan E thresholds and return the one that maximises F2 at or above targetRecall."""
    eBinary = (labels == 0).astype(int)
    eProbs = probabilities[:, 0]

    thresholds = np.linspace(0.01, 0.99, 99)
    bestThresh = None
    bestF2 = -1.0

    print(f"\nThreshold scan (target recall >= {targetRecall:.0%}):")
    print(f"  {'thresh':>6}  {'recall':>6}  {'prec':>6}  {'F2':>6}")

    for t in thresholds:
        ePreds = (eProbs >= t).astype(int)
        tp = int((ePreds & eBinary).sum())
        fp = int((ePreds & (1 - eBinary)).sum())
        fn = int(((1 - ePreds) & eBinary).sum())

        recall = tp / max(tp + fn, 1)
        prec   = tp / max(tp + fp, 1)
        f2     = (5 * prec * recall) / max(5 * prec + recall, 1e-9)

        if recall >= targetRecall and f2 > bestF2:
            bestF2 = f2
            bestThresh = t

    # print a summary table around the target recall
    for t in thresholds:
        ePreds = (eProbs >= t).astype(int)
        tp = int((ePreds & eBinary).sum())
        fp = int((ePreds & (1 - eBinary)).sum())
        fn = int(((1 - ePreds) & eBinary).sum())
        recall = tp / max(tp + fn, 1)
        prec   = tp / max(tp + fp, 1)
        f2     = (5 * prec * recall) / max(5 * prec + recall, 1e-9)

        if 0.80 <= recall <= 0.97:
            marker = " <-- best" if t == bestThresh else ""
            print(f"  {t:6.2f}  {recall:6.3f}  {prec:6.3f}  {f2:6.3f}{marker}")

    if bestThresh is not None:
        print(f"\nBest threshold for recall>={targetRecall:.0%}: {bestThresh:.2f}  (F2={bestF2:.4f})")
    else:
        print(f"\nNo threshold achieves recall >= {targetRecall:.0%}")

    return bestThresh

def computeMetrics(probabilities, labels, outputDir, tag = None, eThreshold = None):
    # probabilities shape: (n_samples, numClasses), labels shape: (n_samples,)
    if eThreshold is not None:
        # Predict E when P(E) >= eThreshold, else fall back to argmax of all classes.
        eProbs = probabilities[:, 0]
        argmaxPreds = np.argmax(probabilities, axis = 1)
        predictions = np.where(eProbs >= eThreshold, 0, argmaxPreds)
    else:
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
    eF2 = fbeta_score(eBinary, (predictions == 0).astype(int), beta = recallBeta, zero_division = 0)

    prPrecision, prRecall, _ = precision_recall_curve(eBinary, eProbs)

    printAndLog(f"\nExoplanet (E) one-vs-rest AUC-PR: {eAuPRc:.4f}")
    printAndLog(f"Exoplanet (E) F{recallBeta:.0f} score:        {eF2:.4f}")

    # recall / precision table at fixed E thresholds
    printAndLog(f"\nE recall at fixed thresholds:")
    printAndLog(f"  {'thresh':>6}  {'recall':>6}  {'prec':>6}  {'caught':>6}  {'missed':>6}")
    nE = int(eBinary.sum())
    for t in np.arange(0.05, 0.96, 0.05):
        ePreds = (eProbs >= t).astype(int)
        tp = int((ePreds & eBinary).sum())
        fp = int((ePreds & (1 - eBinary)).sum())
        fn = int(((1 - ePreds) & eBinary).sum())
        rec  = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        printAndLog(f"  {t:6.2f}  {rec:6.3f}  {prec:6.3f}  {tp:6d}  {fn:6d}")

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

    eThreshold = None

    if args.find_threshold or args.e_threshold is not None:
        eThreshold = findEThreshold(probabilities, labels)

        if args.e_threshold is not None and args.e_threshold != "auto":
            eThreshold = float(args.e_threshold)

    computeMetrics(probabilities, labels, resultsPath, tag, eThreshold = eThreshold)

    

if __name__ == "__main__":
    main()
