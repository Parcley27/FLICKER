# written with Claude Code

import argparse
import csv
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
)

repoRoot = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repoRoot / "model"))

from network import TransitClassifier
from dataset import TransitDataset, makeSplits
from config import classNames, recallBeta

defaultDataPath = repoRoot / "data" / "processed" / "dataset.h5"
defaultScalarsPath = repoRoot / "data" / "processed" / "scalar_stats.json"
defaultSoloCheckpoint = repoRoot / "model" / "checkpoints" / "best_20260522_123141.pt"
defaultChoirDir = repoRoot / "model" / "runs" / "20260522_205646"
defaultOutputRoot = repoRoot / "model" / "evaluation" / "results"

# Thresholds called out explicitly in the paper.
keyThresholds = [0.0105, 0.215, 0.29]
headlineThreshold = 0.29
expectedChoirSize = 10

# Dense grid for the miss-rate-vs-threshold curve. The grid is augmented with the key
# thresholds so each one lands exactly on a sample.
thresholdGrid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 401), np.array(keyThresholds)]))


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Evaluate FLICKER Solo and Choir (Baritone + Soprano) on the test set")

    parser.add_argument("--data", type = Path, default = defaultDataPath,
        help = "Path to dataset.h5")
    parser.add_argument("--scalars", type = Path, default = defaultScalarsPath,
        help = "Path to scalar_stats.json")
    parser.add_argument("--solo-checkpoint", type = Path, default = defaultSoloCheckpoint,
        help = "Path to the Solo model checkpoint (.pt)")
    parser.add_argument("--choir-dir", type = Path, default = defaultChoirDir,
        help = "Path to a model/runs/<timestamp>/ directory containing model_*.pt files")
    parser.add_argument("--no-choir", action = "store_true",
        help = "Skip Choir evaluation even if checkpoints are available")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Batch size for inference")
    parser.add_argument("--workers", type = int, default = 4,
        help = "DataLoader worker count")
    parser.add_argument("--output-dir", type = Path, default = None,
        help = "Override output directory. Defaults to model/evaluation/results/<timestamp>/")

    return parser.parse_args()


def buildLoader(args) -> torch.utils.data.DataLoader:
    splits = makeSplits(args.data)
    testIndices = splits[2]
    testDataset = TransitDataset(args.data, args.scalars, testIndices)
    persistWorkers = args.workers > 0

    return torch.utils.data.DataLoader(
        testDataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory = True,
        persistent_workers = persistWorkers,
    )


def runInference(model, dataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    labels = []

    with torch.no_grad():
        for batch in dataLoader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            logits.append(output.detach().cpu())
            labels.append(batch["label"].detach().cpu())

    logits = torch.cat(logits)
    labels = torch.cat(labels)

    probabilities = torch.softmax(logits, dim = 1).numpy()
    labels = labels.numpy()

    return probabilities, labels


def loadAndInfer(checkpointPath: Path, dataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model = TransitClassifier().to(device)
    model.load_state_dict(torch.load(checkpointPath, map_location = device, weights_only = True))
    model.eval()

    return runInference(model, dataLoader, device)


def applyThresholdRule(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Predict E (class 0) when P(E) >= threshold, otherwise argmax over all classes."""
    eProbs = probabilities[:, 0]
    argmaxPreds = np.argmax(probabilities, axis = 1)

    return np.where(eProbs >= threshold, 0, argmaxPreds)


def sopranoPredictions(individualProbabilities: np.ndarray, baritoneProbabilities: np.ndarray, threshold: float) -> np.ndarray:
    """If ANY of the 10 models has P(E) >= threshold, predict E. Otherwise use Baritone's argmax."""
    # individualProbabilities: (numModels, numSamples, numClasses)
    anyVotesE = (individualProbabilities[:, :, 0] >= threshold).any(axis = 0)
    fallback = np.argmax(baritoneProbabilities, axis = 1)

    return np.where(anyVotesE, 0, fallback)


def binaryEStats(predictions: np.ndarray, eBinary: np.ndarray) -> dict:
    ePreds = (predictions == 0).astype(int)
    tp = int((ePreds & eBinary).sum())
    fp = int((ePreds & (1 - eBinary)).sum())
    fn = int(((1 - ePreds) & eBinary).sum())
    tn = int(((1 - ePreds) & (1 - eBinary)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f2 = (5 * precision * recall) / max(4 * precision + recall, 1e-12)
    missRate = 1.0 - recall

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f2": f2, "missRate": missRate,
    }


def predictionsAtThreshold(method: str, threshold: float,
                            soloProbs: np.ndarray,
                            individualProbs: np.ndarray | None,
                            baritoneProbs: np.ndarray | None) -> np.ndarray:
    if method == "solo":
        return applyThresholdRule(soloProbs, threshold)
    if method == "baritone":
        assert baritoneProbs is not None
        return applyThresholdRule(baritoneProbs, threshold)
    if method == "soprano":
        assert individualProbs is not None and baritoneProbs is not None
        return sopranoPredictions(individualProbs, baritoneProbs, threshold)

    raise ValueError(f"unknown method: {method}")


def methodEScore(method: str, soloProbs, individualProbs, baritoneProbs) -> np.ndarray:
    """Continuous score used for AUC-PR. For Soprano this is max P(E) across models, since
    max is the decision variable the ANY rule actually thresholds."""
    if method == "solo":
        return soloProbs[:, 0]
    if method == "baritone":
        return baritoneProbs[:, 0]
    if method == "soprano":
        return individualProbs[:, :, 0].max(axis = 0)

    raise ValueError(f"unknown method: {method}")


def plotMissRateVsThreshold(perMethodMissRates: dict[str, np.ndarray], outputPath: Path):
    figure, axis = plt.subplots(figsize = (8, 5))

    styles = {
        "solo": {"label": "Solo", "color": "#1f77b4", "linestyle": "-"},
        "baritone": {"label": "Choir Baritone", "color": "#d62728", "linestyle": "--"},
        "soprano": {"label": "Choir Soprano", "color": "#2ca02c", "linestyle": "-."},
    }

    for method, missRates in perMethodMissRates.items():
        style = styles[method]
        axis.plot(thresholdGrid, missRates, linewidth = 2,
                  label = style["label"], color = style["color"], linestyle = style["linestyle"])

    for t in keyThresholds:
        axis.axvline(t, color = "grey", linestyle = ":", alpha = 0.7)
        axis.text(t, 0.97, f"{t}", rotation = 90, va = "top", ha = "right", fontsize = 8, color = "grey")

    axis.set_xlabel("E threshold")
    axis.set_ylabel("Miss rate (1 - recall on E)")
    axis.set_title("E miss rate vs threshold")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha = 0.3)
    axis.legend(loc = "best")

    figure.tight_layout()
    figure.savefig(outputPath, dpi = 150)
    plt.close(figure)


def plotPRCurve(eScore: np.ndarray, eBinary: np.ndarray, method: str, outputPath: Path) -> float:
    aucPR = average_precision_score(eBinary, eScore)
    precision, recall, _ = precision_recall_curve(eBinary, eScore)

    figure, axis = plt.subplots(figsize = (6, 5))
    axis.plot(recall, precision, linewidth = 2)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"{method.capitalize()} E PR curve (AUC-PR = {aucPR:.4f})")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha = 0.3)
    figure.tight_layout()
    figure.savefig(outputPath, dpi = 150)
    plt.close(figure)

    return aucPR


def plotConfusionMatrix(cm: np.ndarray, title: str, outputPath: Path, fmt: str):
    figure, axis = plt.subplots(figsize = (5.5, 4.5))
    image = axis.imshow(cm, cmap = "Blues", aspect = "auto")
    axis.set_xticks(range(len(classNames)))
    axis.set_yticks(range(len(classNames)))
    axis.set_xticklabels(classNames)
    axis.set_yticklabels(classNames)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            axis.text(j, i, format(value, fmt),
                      ha = "center", va = "center",
                      color = "white" if value > threshold else "black",
                      fontsize = 11)

    figure.colorbar(image, ax = axis)
    figure.tight_layout()
    figure.savefig(outputPath, dpi = 150)
    plt.close(figure)


def confusionMatrices(predictions: np.ndarray, labels: np.ndarray):
    cm = confusion_matrix(labels, predictions, labels = list(range(len(classNames))))
    rowSums = cm.sum(axis = 1, keepdims = True)
    rowSums = np.where(rowSums == 0, 1, rowSums)
    cmPercent = 100.0 * cm / rowSums

    return cm, cmPercent


def formatConfusionText(cm: np.ndarray, fmt: str) -> list[str]:
    lines = []
    header = "         " + "  ".join(f"{name:>7}" for name in classNames)
    lines.append(header)

    for i, name in enumerate(classNames):
        row = "  ".join(f"{format(cm[i, j], fmt):>7}" for j in range(len(classNames)))
        lines.append(f"  {name:>4}   {row}")

    return lines


def resolveChoirCheckpoints(choirDir: Path) -> list[Path]:
    if not choirDir.exists():
        return []

    return sorted(choirDir.glob("model_*.pt"))


def main():
    args = parseArgs()

    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outputDir = defaultOutputRoot / timestamp
    else:
        outputDir = args.output_dir

    outputDir.mkdir(parents = True, exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Solo checkpoint -----------------------------------------------------------------
    if not args.solo_checkpoint.exists():
        print(f"ERROR: Solo checkpoint not found at {args.solo_checkpoint}")
        print("       Train one with `python model/train.py`, or pass --solo-checkpoint.")

        return

    # Choir checkpoints ---------------------------------------------------------------
    choirCheckpoints: list[Path] = []
    choirEnabled = not args.no_choir

    if choirEnabled:
        choirCheckpoints = resolveChoirCheckpoints(args.choir_dir)

        if len(choirCheckpoints) == 0:
            print(f"WARNING: no model_*.pt files found in {args.choir_dir}.")
            print("         Choir evaluation will be skipped.")
            print("         Train a 10-model ensemble with:")
            print("           python model/chorus.py --model-count 10")
            print("         Then re-run with --choir-dir model/runs/<new timestamp>/")
            choirEnabled = False

        elif len(choirCheckpoints) < expectedChoirSize:
            print(f"WARNING: only {len(choirCheckpoints)} model_*.pt file(s) found in {args.choir_dir},")
            print(f"         expected {expectedChoirSize} for a full Choir ensemble.")
            print("         Evaluation will proceed with the models available, but Choir results")
            print("         will not match the paper's 10-model spec.")
            print("         To train a complete ensemble:")
            print("           python model/chorus.py --model-count 10")

    # Inference -----------------------------------------------------------------------
    print("Building test loader...")
    testLoader = buildLoader(args)

    print(f"Running Solo inference using {args.solo_checkpoint.name}...")
    soloProbs, labels = loadAndInfer(args.solo_checkpoint, testLoader, device)

    individualProbs = None
    baritoneProbs = None

    if choirEnabled:
        print(f"Running Choir inference over {len(choirCheckpoints)} model(s) from {args.choir_dir.name}...")
        perModelProbs = []

        for checkpointPath in choirCheckpoints:
            print(f"  - {checkpointPath.name}")
            probs, choirLabels = loadAndInfer(checkpointPath, testLoader, device)

            assert np.array_equal(choirLabels, labels), "Choir / Solo label order diverged — dataset ordering bug"

            perModelProbs.append(probs)

        individualProbs = np.stack(perModelProbs, axis = 0)
        baritoneProbs = individualProbs.mean(axis = 0)

    eBinary = (labels == 0).astype(int)

    activeMethods = ["solo"] + (["baritone", "soprano"] if choirEnabled else [])

    # Threshold sweep -----------------------------------------------------------------
    print("Computing threshold sweep...")
    missRatesByMethod: dict[str, np.ndarray] = {}
    sweepRows: list[dict] = []

    for method in activeMethods:
        missRates = np.zeros(len(thresholdGrid), dtype = float)

        for i, t in enumerate(thresholdGrid):
            preds = predictionsAtThreshold(method, float(t), soloProbs, individualProbs, baritoneProbs)
            stats = binaryEStats(preds, eBinary)
            missRates[i] = stats["missRate"]

            sweepRows.append({
                "method": method,
                "threshold": float(t),
                "precision": stats["precision"],
                "recall": stats["recall"],
                "missRate": stats["missRate"],
                "f2": stats["f2"],
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "tn": stats["tn"],
            })

        missRatesByMethod[method] = missRates

    # AUC-PR and PR curves ------------------------------------------------------------
    print("Plotting PR curves...")
    aucByMethod: dict[str, float] = {}

    for method in activeMethods:
        eScore = methodEScore(method, soloProbs, individualProbs, baritoneProbs)
        prPath = outputDir / f"pr_curve_{method}.png"
        aucByMethod[method] = plotPRCurve(eScore, eBinary, method, prPath)

    # Headline metrics + confusion matrices at the operational threshold --------------
    print(f"Computing headline metrics + confusion matrices at threshold {headlineThreshold}...")
    headlineStats: dict[str, dict] = {}
    headlinePredictions: dict[str, np.ndarray] = {}

    for method in activeMethods:
        preds = predictionsAtThreshold(method, headlineThreshold, soloProbs, individualProbs, baritoneProbs)
        headlinePredictions[method] = preds
        headlineStats[method] = binaryEStats(preds, eBinary)

        cmRaw, cmPercent = confusionMatrices(preds, labels)

        plotConfusionMatrix(
            cmRaw,
            f"{method.capitalize()} confusion (counts) @ t={headlineThreshold}",
            outputDir / f"confusion_{method}_raw.png",
            fmt = "d",
        )
        plotConfusionMatrix(
            cmPercent,
            f"{method.capitalize()} confusion (row %) @ t={headlineThreshold}",
            outputDir / f"confusion_{method}_pct.png",
            fmt = ".1f",
        )

    # Miss rate plot ------------------------------------------------------------------
    print("Plotting miss rate vs threshold...")
    plotMissRateVsThreshold(missRatesByMethod, outputDir / "miss_rate_vs_threshold.png")

    # CSV dumps -----------------------------------------------------------------------
    sweepCsv = outputDir / "threshold_sweep.csv"

    with open(sweepCsv, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = ["method", "threshold", "precision", "recall", "missRate", "f2", "tp", "fp", "fn", "tn"])
        writer.writeheader()
        writer.writerows(sweepRows)

    headlineCsv = outputDir / "headline_metrics.csv"

    with open(headlineCsv, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "threshold", "precision", "recall", "f2", "missRate", "tp", "fp", "fn", "tn", "aucPR"])

        for method in activeMethods:
            for t in keyThresholds:
                preds = predictionsAtThreshold(method, t, soloProbs, individualProbs, baritoneProbs)
                stats = binaryEStats(preds, eBinary)
                writer.writerow([
                    method, t,
                    f"{stats['precision']:.6f}", f"{stats['recall']:.6f}",
                    f"{stats['f2']:.6f}", f"{stats['missRate']:.6f}",
                    stats["tp"], stats["fp"], stats["fn"], stats["tn"],
                    f"{aucByMethod[method]:.6f}",
                ])

    # Per-class breakdown at the headline threshold -----------------------------------
    perClassCsv = outputDir / "per_class.csv"

    with open(perClassCsv, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "threshold", "class", "support", "actualPredE", "actualPredS", "actualPredB", "actualPredJ"])

        for method in activeMethods:
            cmRaw, _ = confusionMatrices(headlinePredictions[method], labels)

            for i, name in enumerate(classNames):
                writer.writerow([method, headlineThreshold, name, int(cmRaw[i].sum())] + [int(cmRaw[i, j]) for j in range(len(classNames))])

    # Text summary --------------------------------------------------------------------
    summaryPath = outputDir / "summary.txt"
    summaryLines: list[str] = []

    def line(text: str = ""):
        summaryLines.append(text)
        print(text)

    line("FLICKER evaluation summary")
    line(f"Generated: {datetime.datetime.now().isoformat(timespec = 'seconds')}")
    line(f"Test samples: {len(labels)}")
    classDist = "  ".join(f"{name}={int((labels == i).sum())}" for i, name in enumerate(classNames))
    line(f"Class distribution: {classDist}")
    line()
    line(f"Solo checkpoint: {args.solo_checkpoint}")

    if choirEnabled:
        line(f"Choir directory: {args.choir_dir}  ({len(choirCheckpoints)}/{expectedChoirSize} models)")

        for path in choirCheckpoints:
            line(f"  - {path.name}")

        if len(choirCheckpoints) < expectedChoirSize:
            line()
            line(f"WARNING: Choir is short by {expectedChoirSize - len(choirCheckpoints)} model(s).")
            line("         Train more with: python model/chorus.py --model-count 10")

    else:
        line("Choir: SKIPPED (no checkpoints available).")
        line("Train an ensemble with: python model/chorus.py --model-count 10")

    line()
    line("=" * 72)
    line(f"Headline (threshold = {headlineThreshold})")
    line("=" * 72)
    line(f"  {'method':<10} {'precision':>10} {'recall':>8} {'F2':>8} {'missRate':>10} {'AUC-PR':>8}")

    for method in activeMethods:
        stats = headlineStats[method]
        line(f"  {method:<10} {stats['precision']:>10.4f} {stats['recall']:>8.4f} {stats['f2']:>8.4f} {stats['missRate']:>10.4f} {aucByMethod[method]:>8.4f}")

    line()
    line("Headline counts (E vs rest):")
    line(f"  {'method':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")

    for method in activeMethods:
        stats = headlineStats[method]
        line(f"  {method:<10} {stats['tp']:>5} {stats['fp']:>5} {stats['fn']:>5} {stats['tn']:>5}")

    line()
    line("=" * 72)
    line("Per-threshold breakdown (precision / recall / F2 on E)")
    line("=" * 72)

    for t in keyThresholds:
        line(f"\n  Threshold = {t}")
        line(f"    {'method':<10} {'precision':>10} {'recall':>8} {'F2':>8} {'missRate':>10}")

        for method in activeMethods:
            preds = predictionsAtThreshold(method, t, soloProbs, individualProbs, baritoneProbs)
            stats = binaryEStats(preds, eBinary)
            line(f"    {method:<10} {stats['precision']:>10.4f} {stats['recall']:>8.4f} {stats['f2']:>8.4f} {stats['missRate']:>10.4f}")

    line()
    line("=" * 72)
    line(f"AUC-PR (E one-vs-rest)")
    line("=" * 72)

    for method in activeMethods:
        line(f"  {method:<10} {aucByMethod[method]:.4f}")

    line()
    line("=" * 72)
    line(f"Confusion matrices @ threshold = {headlineThreshold}")
    line("=" * 72)

    for method in activeMethods:
        cmRaw, cmPercent = confusionMatrices(headlinePredictions[method], labels)

        line(f"\n{method.capitalize()} - counts (rows = actual, columns = predicted)")

        for row in formatConfusionText(cmRaw, "d"):
            line(row)

        line(f"\n{method.capitalize()} - row percentages")

        for row in formatConfusionText(cmPercent, ".1f"):
            line(row)

    line()
    line("Outputs:")
    line(f"  {summaryPath}")
    line(f"  {sweepCsv}")
    line(f"  {headlineCsv}")
    line(f"  {perClassCsv}")
    line(f"  {outputDir / 'miss_rate_vs_threshold.png'}")

    for method in activeMethods:
        line(f"  {outputDir / f'pr_curve_{method}.png'}")
        line(f"  {outputDir / f'confusion_{method}_raw.png'}")
        line(f"  {outputDir / f'confusion_{method}_pct.png'}")

    with open(summaryPath, "w", encoding = "utf-8") as f:
        f.write("\n".join(summaryLines) + "\n")


if __name__ == "__main__":
    main()
