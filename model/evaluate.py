import argparse
import numpy as np
import sys
import torch

from pathlib import Path
from sklearn.metrics import recall_score, precision_score, average_precision_score

from model import TransitClassifier
from dataset import TransitDataset, makeSplits

repoRoot = Path(__file__).resolve().parent.parent
defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"

checkpointPath = repoRoot / "model" / "checkpoints"

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Evaluate the TransitClassifier model on the test set")

    parser.add_argument("--data", type = Path, default = defaultDataPath,
        help = "Path to dataset.h5 (default: data/processed/dataset.h5)")
    parser.add_argument("--scalars", type = Path, default = defaultScalarsPath,
        help = "Path to scalar_stats.json (default: data/processed/scalar_stats.json)")
    parser.add_argument("--checkpoint", type = Path, default = None,
        help = "Path to model checkpoint (.pt file). Defaults to highest AUC-PR checkpoint in model/checkpoints/")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = 4,
        help = "DataLoader worker count (default: 4)")

    return parser.parse_args()

def main():
    args = parseArgs()

    checkpoint = args.checkpoint

    if checkpoint is None:
        files = list(checkpointPath.glob("best_*.pt"))

        if files == []:
            print("No checkpoints found in model/checkpoints/")
        
            sys.exit(1)

        checkpoint = max(files, key = lambda file: float(file.stem.split("_")[1]))

    print(f"Using checkpoint {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building model...")
    model = TransitClassifier().to(device)

    #load state dict with map_location
    model.load_state_dict(torch.load(checkpoint, map_location = device, weights_only = True))

    model.eval()

    print("Building data loader...")
    splits = makeSplits(args.data)

    testIndices = splits[2]

    testDataset = TransitDataset(args.data, args.scalars, testIndices)

    testLoader = torch.utils.data.DataLoader(
        testDataset, batch_size = args.batch_size, shuffle = False,
        num_workers = args.workers, pin_memory = True, persistent_workers = True,
    
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
        precision = precision_score(labels, (probabilities >= 0.5).astype(int))
        recall = recall_score(labels, (probabilities >= 0.5).astype(int))

        print(f"Test auPRc: {auPRc:.4f} | Test precision: {precision:.4f} | Test recall: {recall:.4f}")

        # precision/recall at each threshold to inform operating point choice
        print("\nThreshold | Precision | Recall")
        for threshold in np.arange(0.1, 1.0, 0.1):
            binaryPredictions = (probabilities >= threshold).astype(int)
            thresholdPrecision = precision_score(labels, binaryPredictions, zero_division = 0)
            thresholdRecall = recall_score(labels, binaryPredictions)
            
            print(f"  {threshold:.1f}     |  {thresholdPrecision:.4f}   | {thresholdRecall:.4f}")
        
if __name__ == "__main__":
    main()
