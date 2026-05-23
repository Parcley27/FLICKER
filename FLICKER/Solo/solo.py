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
sys.path.insert(0, str(repoRoot / "model"))

from network import TransitClassifier
from config import classNames

defaultCheckpoint = scriptDir / "best.pt"
defaultOutputRoot = scriptDir / "results"
defaultThreshold = 0.29


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "FLICKER Solo: single-model exoplanet inference over a folder of preprocessed light curves")

    parser.add_argument("--input", type = Path, required = True,
        help = "Folder containing one or more .h5 files in dataset.h5 schema")
    parser.add_argument("--checkpoint", type = Path, default = defaultCheckpoint,
        help = "Path to the Solo model checkpoint (default: FLICKER/Solo/best.pt)")
    parser.add_argument("--threshold", type = float, default = defaultThreshold,
        help = f"E probability threshold for predicting E (default: {defaultThreshold})")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Inference batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = 0,
        help = "DataLoader worker count (default: 0, Pi-friendly)")
    parser.add_argument("--device", choices = ["auto", "cpu", "cuda"], default = "auto",
        help = "Compute device (default: auto)")
    parser.add_argument("--output-dir", type = Path, default = None,
        help = "Override output directory. Defaults to FLICKER/Solo/results/<timestamp>/")

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


def checkpointMissingMessage(path: Path) -> str:
    return (
        f"ERROR: No Solo checkpoint found at {path}.\n"
        "\n"
        "Train a model first:\n"
        "  python model/train.py\n"
        "\n"
        "Then copy your best checkpoint into FLICKER/Solo/:\n"
        f"  cp model/checkpoints/best_<timestamp>.pt {path}      (macOS / Linux)\n"
        f"  copy model\\checkpoints\\best_<timestamp>.pt {path}    (Windows cmd)\n"
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

        # index: list of (filePath, ticID, obsIdx). Read once at init so __len__ works
        # and __getitem__ can stay light. h5py file handles are opened lazily per worker.
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


def buildPredictions(probabilities: np.ndarray, indices: np.ndarray, dataset: InferenceDataset, threshold: float) -> list[dict]:
    rows: list[dict] = []
    argmax = probabilities.argmax(axis = 1)

    for i, originalIndex in enumerate(indices):
        filePath, ticID, obsIdx = dataset.index[int(originalIndex)]
        probs = probabilities[i]
        argmaxClass = int(argmax[i])
        forced = probs[0] >= threshold and argmaxClass != 0
        predicted = "E" if (probs[0] >= threshold) else classNames[argmaxClass]

        rows.append({
            "sourceFile": str(filePath.relative_to(dataset.inputDir) if filePath.is_relative_to(dataset.inputDir) else filePath),
            "ticID": ticID,
            "obsIdx": obsIdx,
            "P_E": float(probs[0]),
            "P_S": float(probs[1]),
            "P_B": float(probs[2]),
            "P_J": float(probs[3]),
            "predicted": predicted,
            "usedThreshold": forced,
        })

    return rows


def writeCsv(rows: list[dict], path: Path):
    fieldnames = ["sourceFile", "ticID", "obsIdx", "P_E", "P_S", "P_B", "P_J", "predicted", "usedThreshold"]

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
                "usedThreshold": "true" if row["usedThreshold"] else "false",
            })


def buildSummary(rows: list[dict], checkpoint: Path, threshold: float, inputDir: Path, device: torch.device, started: datetime.datetime) -> str:
    lines: list[str] = []
    total = len(rows)

    classCounts = {name: 0 for name in classNames}
    forcedCount = 0
    perFile: dict[str, int] = {}

    for row in rows:
        classCounts[row["predicted"]] += 1
        perFile[row["sourceFile"]] = perFile.get(row["sourceFile"], 0) + 1

        if row["usedThreshold"]:
            forcedCount += 1

    lines.append("FLICKER Solo predictions")
    lines.append(f"Generated:   {started.isoformat(timespec = 'seconds')}")
    lines.append(f"Checkpoint:  {checkpoint}")
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
    lines.append(f"E predictions forced by threshold (P_E >= {threshold}, argmax != E): {forcedCount}")

    eRows = sorted((row for row in rows), key = lambda row: row["P_E"], reverse = True)[:20]
    lines.append("")
    lines.append("Top 20 highest-P(E) TCEs:")
    lines.append(f"  {'rank':>4}  {'P_E':>7}  {'pred':>4}  {'ticID':<14}  {'obsIdx':<8}  sourceFile")

    for rank, row in enumerate(eRows, 1):
        lines.append(f"  {rank:>4}  {row['P_E']:7.4f}  {row['predicted']:>4}  {row['ticID']:<14}  {row['obsIdx']:<8}  {row['sourceFile']}")

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
    print(f"  {'ticID':<14}  {'obsIdx':<8}  {'P_E':>7}  {'P_S':>7}  {'P_B':>7}  {'P_J':>7}  {'pred':>4}")

    for row in rows[:limit]:
        print(f"  {row['ticID']:<14}  {row['obsIdx']:<8}  {row['P_E']:7.4f}  {row['P_S']:7.4f}  {row['P_B']:7.4f}  {row['P_J']:7.4f}  {row['predicted']:>4}")


def main() -> int:
    args = parseArgs()
    started = datetime.datetime.now()

    if not args.checkpoint.exists():
        print(checkpointMissingMessage(args.checkpoint))

        return 1

    if not args.input.exists() or not args.input.is_dir():
        print(f"ERROR: --input must be an existing directory; got {args.input}")

        return 1

    device = resolveDevice(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")
    print(f"Input: {args.input}")

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

    print("Loading model...")
    model = TransitClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location = device, weights_only = True))
    model.eval()

    print("Running inference...")
    probabilities, indices = runInference(model, loader, device)

    print("Building predictions...")
    rows = buildPredictions(probabilities, indices, dataset, args.threshold)
    # Preserve original dataset ordering rather than batch arrival order.
    rows.sort(key = lambda row: (row["sourceFile"], row["ticID"], row["obsIdx"]))

    outputDir = args.output_dir or (defaultOutputRoot / started.strftime("%Y%m%d_%H%M%S"))
    outputDir.mkdir(parents = True, exist_ok = True)

    predictionsCsv = outputDir / "predictions.csv"
    summaryTxt = outputDir / "summary.txt"

    writeCsv(rows, predictionsCsv)
    summary = buildSummary(rows, args.checkpoint, args.threshold, args.input, device, started)

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
