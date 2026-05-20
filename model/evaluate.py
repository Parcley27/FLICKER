import argparse
import datetime
import numpy as np
import torch

from pathlib import Path
from sklearn.metrics import recall_score, precision_score, average_precision_score, confusion_matrix

from network import TransitClassifier
from dataset import TransitDataset, makeSplits

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

checkpointPath = repoRoot / "model" / "checkpoints"
resultsPath = repoRoot / "model" / "results"

labelNames = ["E", "S", "B", "J", "N"]

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Evaluate the TransitClassifier model on the test set")

    parser.add_argument("--data", type = Path, default = defaultDataPath,
        help = "Path to dataset.h5 (default: data/processed/dataset.h5)")
    parser.add_argument("--scalars", type = Path, default = defaultScalarsPath,
        help = "Path to scalar_stats.json (default: data/processed/scalar_stats.json)")
    parser.add_argument("--checkpoint", type = Path, default = None,
        help = "Path to model checkpoint (.pt file). Defaults to model/checkpoints/best.pt")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = 4,
        help = "DataLoader worker count (default: 4)")

    return parser.parse_args()

def main():
    args = parseArgs()

    checkpoint = args.checkpoint

    if checkpoint is None:
        checkpoint = checkpointPath / "best.pt"

        if not checkpoint.exists():
            print(f"No checkpoint found at {checkpoint}")
            print("Run train.py first or specify --checkpoint")

            return

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

    # primary metric: AUC-PR on the exoplanet (E) column
    eProbs = probabilities[:, 0]
    eLabels = labels[:, 0]

    auPRc = average_precision_score(eLabels, eProbs)
    precision = precision_score(eLabels, (eProbs >= 0.5).astype(int), zero_division = 0)
    recall = recall_score(eLabels, (eProbs >= 0.5).astype(int), zero_division = 0)

    outputLines = []

    def printAndLog(line = ""):
        print(line)
        outputLines.append(line)

    printAndLog(f"Checkpoint: {checkpoint}")
    printAndLog(f"Test samples: {len(labels)}")
    printAndLog(f"\nPrimary metric (E class):")
    printAndLog(f"  AUC-PR: {auPRc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # per-class AUC-PR
    printAndLog(f"\nPer-class AUC-PR:")

    for i, name in enumerate(labelNames):
        classAuPR = average_precision_score(labels[:, i], probabilities[:, i])
        printAndLog(f"  {name}: {classAuPR:.4f}")

    # threshold sweep on E column
    printAndLog(f"\nThreshold sweep (E class):")
    printAndLog(f"  Threshold | Precision | Recall")

    for threshold in np.arange(0.1, 1.0, 0.1):
        binaryPredictions = (eProbs >= threshold).astype(int)
        thresholdPrecision = precision_score(eLabels, binaryPredictions, zero_division = 0)
        thresholdRecall = recall_score(eLabels, binaryPredictions, zero_division = 0)

        printAndLog(f"  {threshold:.1f}       |  {thresholdPrecision:.4f}   | {thresholdRecall:.4f}")

    # confusion matrix using argmax predictions
    predictedClasses = np.argmax(probabilities, axis = 1)
    trueClasses = np.argmax(labels, axis = 1)

    confusionMatrix = confusion_matrix(trueClasses, predictedClasses, labels = list(range(len(labelNames))))

    printAndLog(f"\nConfusion matrix (rows = true, columns = predicted):")
    printAndLog(f"{'':>8s} " + " ".join(f"{name:>6s}" for name in labelNames))

    for i, name in enumerate(labelNames):
        row = " ".join(f"{confusionMatrix[i, j]:6d}" for j in range(len(labelNames)))
        printAndLog(f"{name:>8s} {row}")

    # save results
    resultsPath.mkdir(parents = True, exist_ok = True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outputFile = resultsPath / f"eval_{timestamp}.txt"

    with open(outputFile, "w") as f:
        f.write("\n".join(outputLines) + "\n")

    print(f"\nResults saved to {outputFile}")

if __name__ == "__main__":
    main()
