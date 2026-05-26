import argparse
import csv
import datetime
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

scriptDir = Path(__file__).resolve().parent
repoRoot = scriptDir.parents[1]
sys.path.insert(0, str(repoRoot / "src" / "model"))

from network import TransitClassifier
from config import classNames

defaultModelsDir = scriptDir / "models"
defaultOutputRoot = scriptDir / "results"
defaultThreshold = 0.29
expectedChoirSize = 10


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "FLICKER Choir: ensemble exoplanet inference over a folder of preprocessed light curves")

    modeGroup = parser.add_mutually_exclusive_group(required = True)
    modeGroup.add_argument("--baritone", action = "store_true",
        help = "Mean probability vector across all models, then threshold rule")
    modeGroup.add_argument("--soprano", action = "store_true",
        help = "Predict E if any model assigns P_E >= threshold; otherwise use Baritone argmax")

    parser.add_argument("--input", type = Path, required = True,
        help = "Folder containing one or more .h5 files in dataset.h5 schema")
    parser.add_argument("--choir-dir", type = Path, default = defaultModelsDir,
        help = "Folder containing model_*.pt checkpoints (default: FLICKER/Choir/models/)")
    parser.add_argument("--threshold", type = float, default = defaultThreshold,
        help = f"E probability threshold (default: {defaultThreshold})")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Inference batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = 0,
        help = "DataLoader worker count (default: 0, Pi-friendly)")
    parser.add_argument("--device", choices = ["auto", "cpu", "cuda"], default = "auto",
        help = "Compute device (default: auto)")
    parser.add_argument("--output-dir", type = Path, default = None,
        help = "Override output directory. Defaults to FLICKER/Choir/results/<timestamp>/")

    return parser.parse_args()


def resolveDevice(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: --device cuda requested but CUDA is not available; falling back to CPU.")

            return torch.device("cpu")

        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def modelsMissingMessage(choirDir: Path, found: int) -> str:
    return (
        f"ERROR: Found {found} model_*.pt file(s) in {choirDir}, expected {expectedChoirSize}.\n"
        "\n"
        "Train a 10-model ensemble first:\n"
        "  python model/chorus.py --model-count 10\n"
        "\n"
        "Then copy the checkpoints into FLICKER/Choir/models/:\n"
        f"  cp model/runs/<timestamp>/model_*.pt {defaultModelsDir}      (macOS / Linux)\n"
        f"  copy model\\runs\\<timestamp>\\model_*.pt {defaultModelsDir}    (Windows cmd)\n"
    )


class InferenceDataset(torch.utils.data.Dataset):
    """Walks a folder of dataset.h5-schema files and yields per-TCE view tensors for inference.

    Mirrors TransitDataset's view transforms (transpose, nan_to_num, clamp) but skips the
    label filter so unlabeled TCEs are kept, and skips augmentation.
    """

    def __init__(self, inputDir: Path):
        self.inputDir = inputDir

        files = sorted(inputDir.rglob("*.h5"))

        if not files:
            raise FileNotFoundError(f"No .h5 files found under {inputDir}")

        self.index: list[tuple[Path, str, str]] = []

        for filePath in files:
            with h5py.File(filePath, "r") as f:
                for ticID in f.keys():
                    group = f[ticID]

                    for obsIdx in group.keys():  # type: ignore[union-attr]
                        self.index.append((filePath, ticID, obsIdx))

        self._handles: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __del__(self):
        for handle in self._handles.values():
            try:
                handle.close()

            except Exception:
                pass

    def _open(self, filePath: Path) -> h5py.File:
        handle = self._handles.get(filePath)

        if handle is None:
            handle = h5py.File(filePath, "r")
            self._handles[filePath] = handle

        return handle

    def __getitem__(self, index: int) -> dict:
        filePath, ticID, obsIdx = self.index[index]
        sample = self._open(filePath)[ticID][obsIdx]  # type: ignore[index]

        def view(name: str) -> torch.Tensor:
            return torch.tensor(sample[name][()].T, dtype = torch.float32).nan_to_num(nan = 0.0, posinf = 0.0, neginf = 0.0).clamp(-5.0, 5.0)  # type: ignore[index]

        return {
            "globalView": view("globalView"),
            "localView": view("localView"),
            "secondaryView": view("secondaryView"),
            "halfPeriodView": view("halfPeriodView"),
            "oddEvenView": view("oddEvenView"),
            "scalars": torch.tensor(sample["scalars"][()], dtype = torch.float32),  # type: ignore[index]
            "_idx": torch.tensor(index, dtype = torch.long),
        }


def runInference(model: TransitClassifier, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities, sampleIndices). probabilities[i] corresponds to dataset[sampleIndices[i]]."""
    allLogits: list[torch.Tensor] = []
    allIndices: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            sampleIndices = batch.pop("_idx")
            modelInput = {key: value.to(device) for key, value in batch.items()}

            logits = model(modelInput)

            allLogits.append(logits.detach().cpu())
            allIndices.append(sampleIndices)

    probabilities = torch.softmax(torch.cat(allLogits), dim = 1).numpy()
    indices = torch.cat(allIndices).numpy()

    return probabilities, indices


def runAllModels(checkpoints: list[Path], loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference for each checkpoint.

    Returns (perModelProbs, sampleIndices) where perModelProbs has shape (numModels, numSamples, numClasses).
    """
    allPerModelProbs: list[np.ndarray] = []
    indices: np.ndarray | None = None

    for checkpointPath in checkpoints:
        print(f"  - {checkpointPath.name}")
        model = TransitClassifier().to(device)
        model.load_state_dict(torch.load(checkpointPath, map_location = device, weights_only = True))
        model.eval()

        probs, idx = runInference(model, loader, device)
        allPerModelProbs.append(probs)

        if indices is None:
            indices = idx
        else:
            assert np.array_equal(idx, indices), "Inference returned different sample ordering across models"

    return np.stack(allPerModelProbs, axis = 0), indices  # type: ignore[return-value]


def buildPredictions(perModelProbs: np.ndarray, indices: np.ndarray, dataset: InferenceDataset, threshold: float, useSoprano: bool) -> list[dict]:
    """Build per-TCE prediction rows.

    perModelProbs shape: (numModels, numSamples, numClasses)

    CSV columns always include P_E/S/B/J (mean across models) and maxP_E (max P_E across models).
    For Baritone: usedThreshold = threshold forced E despite mean-prob argmax not being E.
    For Soprano:  usedThreshold = soprano vote fired (any model P_E >= threshold).
    """
    meanProbs = perModelProbs.mean(axis = 0)         # (N, 4)
    maxEProbs = perModelProbs[:, :, 0].max(axis = 0) # (N,)
    argmaxOfMean = meanProbs.argmax(axis = 1)

    rows: list[dict] = []

    for i, originalIndex in enumerate(indices):
        filePath, ticID, obsIdx = dataset.index[int(originalIndex)]
        probs = meanProbs[i]
        maxPE = float(maxEProbs[i])

        if useSoprano:
            sopranoVote = maxPE >= threshold
            predicted = "E" if sopranoVote else classNames[int(argmaxOfMean[i])]
            usedThreshold = sopranoVote
        else:
            argmaxClass = int(argmaxOfMean[i])
            forced = probs[0] >= threshold and argmaxClass != 0
            predicted = "E" if probs[0] >= threshold else classNames[argmaxClass]
            usedThreshold = forced

        rows.append({
            "sourceFile": str(filePath.relative_to(dataset.inputDir) if filePath.is_relative_to(dataset.inputDir) else filePath),
            "ticID": ticID,
            "obsIdx": obsIdx,
            "P_E": float(probs[0]),
            "P_S": float(probs[1]),
            "P_B": float(probs[2]),
            "P_J": float(probs[3]),
            "maxP_E": maxPE,
            "predicted": predicted,
            "usedThreshold": usedThreshold,
        })

    return rows


def writeCsv(rows: list[dict], path: Path):
    fieldnames = ["sourceFile", "ticID", "obsIdx", "P_E", "P_S", "P_B", "P_J", "maxP_E", "predicted", "usedThreshold"]

    with open(path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                **row,
                "P_E": f"{row['P_E']:.6f}",
                "P_S": f"{row['P_S']:.6f}",
                "P_B": f"{row['P_B']:.6f}",
                "P_J": f"{row['P_J']:.6f}",
                "maxP_E": f"{row['maxP_E']:.6f}",
                "usedThreshold": "true" if row["usedThreshold"] else "false",
            })


def buildSummary(rows: list[dict], choirDir: Path, numModels: int, threshold: float, useSoprano: bool, inputDir: Path, device: torch.device, started: datetime.datetime) -> str:
    lines: list[str] = []
    total = len(rows)
    modeName = "Soprano" if useSoprano else "Baritone"

    classCounts = {name: 0 for name in classNames}
    thresholdFiredCount = 0
    perFile: dict[str, int] = {}

    for row in rows:
        classCounts[row["predicted"]] += 1
        perFile[row["sourceFile"]] = perFile.get(row["sourceFile"], 0) + 1

        if row["usedThreshold"]:
            thresholdFiredCount += 1

    lines.append(f"FLICKER Choir predictions ({modeName})")
    lines.append(f"Generated:   {started.isoformat(timespec = 'seconds')}")
    lines.append(f"Mode:        {modeName}")
    lines.append(f"Choir dir:   {choirDir}  ({numModels}/{expectedChoirSize} models)")
    lines.append(f"Input:       {inputDir}")
    lines.append(f"Device:      {device}")
    lines.append(f"Threshold:   {threshold}")
    lines.append(f"TCE count:   {total}")
    lines.append(f"File count:  {len(perFile)}")
    lines.append("")
    lines.append("Predicted class distribution:")

    for name in classNames:
        count = classCounts[name]
        pct = (100.0 * count / total) if total else 0.0
        lines.append(f"  {name}: {count:>6}  ({pct:5.1f}%)")

    lines.append("")

    if useSoprano:
        lines.append(f"Soprano votes (any model P_E >= {threshold}): {thresholdFiredCount}")
        lines.append("  usedThreshold=true  -> soprano vote fired; prediction is E")
        lines.append("  usedThreshold=false -> no model voted E; fallback to Baritone argmax")
        lines.append("")
        lines.append("Note: P_E in the CSV is the mean P_E across all models.")
        lines.append("      maxP_E is the decision variable (max P_E across models).")
        lines.append("      The soprano threshold is applied to maxP_E, not to P_E.")
    else:
        lines.append(f"E predictions forced by threshold (mean P_E >= {threshold}, argmax != E): {thresholdFiredCount}")
        lines.append("  usedThreshold=true  -> threshold overrode argmax of mean probs")
        lines.append("  usedThreshold=false -> no override (argmax agreed or threshold not met)")
        lines.append("")
        lines.append("Note: P_E in the CSV is the mean P_E across all models.")
        lines.append("      maxP_E shows the highest single-model P_E (spread indicator).")

    sortKey = "maxP_E" if useSoprano else "P_E"
    topRows = sorted(rows, key = lambda row: row[sortKey], reverse = True)[:20]

    lines.append("")
    lines.append(f"Top 20 highest-{sortKey} TCEs:")
    lines.append(f"  {'rank':>4}  {'P_E':>7}  {'maxP_E':>7}  {'pred':>4}  {'ticID':<14}  {'obsIdx':<8}  sourceFile")

    for rank, row in enumerate(topRows, 1):
        lines.append(f"  {rank:>4}  {row['P_E']:7.4f}  {row['maxP_E']:7.4f}  {row['predicted']:>4}  {row['ticID']:<14}  {row['obsIdx']:<8}  {row['sourceFile']}")

    lines.append("")
    lines.append("Per-source-file TCE counts:")

    for name, count in sorted(perFile.items()):
        lines.append(f"  {count:>6}  {name}")

    return "\n".join(lines) + "\n"


def printTablePreview(rows: list[dict], limit: int = 20):
    if not rows:
        return

    print()
    print(f"First {min(limit, len(rows))} predictions:")
    print(f"  {'ticID':<14}  {'obsIdx':<8}  {'P_E':>7}  {'maxP_E':>7}  {'P_S':>7}  {'P_B':>7}  {'P_J':>7}  {'pred':>4}")

    for row in rows[:limit]:
        print(f"  {row['ticID']:<14}  {row['obsIdx']:<8}  {row['P_E']:7.4f}  {row['maxP_E']:7.4f}  {row['P_S']:7.4f}  {row['P_B']:7.4f}  {row['P_J']:7.4f}  {row['predicted']:>4}")


def main() -> int:
    args = parseArgs()
    started = datetime.datetime.now()
    useSoprano = args.soprano

    if not args.input.exists() or not args.input.is_dir():
        print(f"ERROR: --input must be an existing directory; got {args.input}")

        return 1

    checkpoints = sorted(args.choir_dir.glob("model_*.pt")) if args.choir_dir.exists() else []

    if len(checkpoints) < expectedChoirSize:
        print(modelsMissingMessage(args.choir_dir, len(checkpoints)))

        return 1

    device = resolveDevice(args.device)
    modeName = "Soprano" if useSoprano else "Baritone"

    print(f"Mode:      {modeName}")
    print(f"Device:    {device}")
    print(f"Choir dir: {args.choir_dir}  ({len(checkpoints)}/{expectedChoirSize} models)")
    print(f"Threshold: {args.threshold}")
    print(f"Input:     {args.input}")

    print("Indexing input folder...")

    try:
        dataset = InferenceDataset(args.input)

    except FileNotFoundError as exception:
        print(f"ERROR: {exception}")

        return 1

    print(f"Found {len(dataset)} TCE(s).")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory = (device.type == "cuda"),
        persistent_workers = (args.workers > 0),
    )

    print(f"Running inference over {len(checkpoints)} models...")
    perModelProbs, indices = runAllModels(checkpoints, loader, device)

    print("Building predictions...")
    rows = buildPredictions(perModelProbs, indices, dataset, args.threshold, useSoprano)
    rows.sort(key = lambda row: (row["sourceFile"], row["ticID"], row["obsIdx"]))

    outputDir = args.output_dir or (defaultOutputRoot / started.strftime("%Y%m%d_%H%M%S"))
    outputDir.mkdir(parents = True, exist_ok = True)

    predictionsCsv = outputDir / "predictions.csv"
    summaryTxt = outputDir / "summary.txt"

    writeCsv(rows, predictionsCsv)
    summary = buildSummary(rows, args.choir_dir, len(checkpoints), args.threshold, useSoprano, args.input, device, started)

    with open(summaryTxt, "w", encoding = "utf-8") as f:
        f.write(summary)

    print()
    print(summary, end = "")
    printTablePreview(rows)
    print()
    print(f"Wrote {predictionsCsv}")
    print(f"Wrote {summaryTxt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
