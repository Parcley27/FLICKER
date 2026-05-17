# Written with Claude Code

import argparse
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path


repoRoot = Path(__file__).resolve().parent.parent
processedDir = repoRoot / "data" / "processed"
rawDir = repoRoot / "data" / "raw"

datasetPath = processedDir / "dataset.h5"
scalarStatsPath = processedDir / "scalar_stats.json"
tceTablePath = rawDir / "tce_table.csv"

renderDir = repoRoot / "data" / "renders"

labelNames = {0: "Exoplanet", 1: "Single Transit", 2: "Binary", 3: "Junk", 4: "Uncertain", -1: "Unlabeled"}
labelShort = {0: "E", 1: "S", 2: "B", 3: "J", 4: "N", -1: "?"}


def loadTceRow(ticID):
    table = pd.read_csv(tceTablePath, comment = "#")
    table = table.drop(0).reset_index(drop = True)
    table["TIC ID"] = table["TIC ID"].astype(int)

    matches = table[table["TIC ID"] == ticID]

    if matches.empty:
        return None

    return matches.iloc[0]


def denormalizeScalars(scalars):
    with open(scalarStatsPath, "r") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype = np.float32)
    std = np.array(stats["std"], dtype = np.float32)

    # indices 8-10 have NaN stats, leave them as-is
    raw = scalars.copy()

    for i in range(len(mean)):
        if not np.isnan(mean[i]) and not np.isnan(std[i]):
            raw[i] = scalars[i] * std[i] + mean[i]

    return raw


def generateChart(ticID, predictedLabel = None):
    renderDir.mkdir(parents = True, exist_ok = True)

    ticKey = str(ticID)

    with h5py.File(datasetPath, "r") as database:
        if ticKey not in database:
            print(f"TIC {ticID} not found in dataset")
            return

        group = database[ticKey]["0"]

        globalView = group["globalView"][:]
        localView = group["localView"][:]
        secondaryView = group["secondaryView"][:]
        scalars = group["scalars"][:]
        label = int(group["label"][()])
        exoplanetLabel = int(group["exoplanetLabel"][()])
        period = group.attrs["period"]
        epoch = group.attrs["epoch"]

    raw = denormalizeScalars(scalars)

    tceRow = loadTceRow(ticID)

    # phase axes for each view
    globalPhases = np.linspace(-0.5, 0.5, 201)
    duration = raw[1]
    halfWidth = 2 * duration / period
    localPhases = np.linspace(-halfWidth, halfWidth, 61)
    secondaryPhase = raw[11]
    secondaryPhases = np.linspace(secondaryPhase - halfWidth, secondaryPhase + halfWidth, 61)

    # colours
    plotColour = "#4a90d9"
    stdColour = "#4a90d9"
    transitColour = "#d94a4a"

    fig = plt.figure(figsize = (12, 8), facecolor = "#fafafa")

    outer = gridspec.GridSpec(3, 1, height_ratios = [0.08, 1, 1], hspace = 0.35,
                              top = 0.94, bottom = 0.08, left = 0.08, right = 0.96)

    # header info strip
    headerAx = fig.add_subplot(outer[0])
    headerAx.axis("off")

    trueLabelStr = f"{labelNames.get(label, '?')} ({labelShort.get(label, '?')})"

    headerParts = [
        f"TIC {ticID}",
        f"Period: {period:.4f} d",
        f"Duration: {duration:.4f} d",
        f"Depth: {raw[2]:.0f} ppm",
        f"Tmag: {raw[3]:.2f}",
        f"Label: {trueLabelStr}",
    ]

    if tceRow is not None and pd.notna(tceRow.get("SRad")):
        headerParts.insert(5, f"R*: {float(tceRow['SRad']):.3f} R\u2609")

    if predictedLabel is not None:
        predStr = f"{labelNames.get(predictedLabel, predictedLabel)} ({labelShort.get(predictedLabel, predictedLabel)})"
        headerParts.append(f"Predicted: {predStr}")

    headerText = "    \u2502    ".join(headerParts)
    headerAx.text(0.5, 0.5, headerText, transform = headerAx.transAxes,
                  ha = "center", va = "center", fontsize = 9, fontfamily = "monospace",
                  color = "#333333")

    # global view — full width
    globalAx = fig.add_subplot(outer[1])

    globalMedian = globalView[:, 0]
    globalStd = globalView[:, 1]
    transitFlags = globalView[:, 2].astype(bool)

    globalAx.fill_between(globalPhases, globalMedian - globalStd, globalMedian + globalStd,
                          alpha = 0.15, color = stdColour, linewidth = 0)
    globalAx.plot(globalPhases, globalMedian, color = plotColour, linewidth = 3.0)

    # highlight transit region
    transitIndices = np.where(transitFlags)[0]

    if len(transitIndices) > 0:
        tStart = globalPhases[transitIndices[0]]
        tEnd = globalPhases[transitIndices[-1]]
        globalAx.axvspan(tStart, tEnd, alpha = 0.08, color = transitColour)

    globalAx.set_xlabel("Orbital Phase", fontsize = 9)
    globalAx.set_ylabel("Normalised Flux", fontsize = 9)
    globalAx.set_title("Global View", fontsize = 10, fontweight = "bold", loc = "left")
    globalAx.tick_params(labelsize = 8)
    globalAx.set_xlim(-0.5, 0.5)

    # bottom row: local and secondary side by side
    bottomInner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[2], wspace = 0.25)

    # local view
    localAx = fig.add_subplot(bottomInner[0])

    localMedian = localView[:, 0]
    localStd = localView[:, 1]

    localAx.fill_between(localPhases, localMedian - localStd, localMedian + localStd,
                         alpha = 0.15, color = stdColour, linewidth = 0)
    localAx.plot(localPhases, localMedian, color = plotColour, linewidth = 2.2)

    localAx.set_xlabel("Orbital Phase", fontsize = 9)
    localAx.set_ylabel("Normalised Flux", fontsize = 9)
    localAx.set_title("Local View (Transit)", fontsize = 10, fontweight = "bold", loc = "left")
    localAx.tick_params(labelsize = 8)

    # secondary view
    secondaryAx = fig.add_subplot(bottomInner[1])

    secondaryMedian = secondaryView[:, 0]
    secondaryStd = secondaryView[:, 1]

    secondaryAx.fill_between(secondaryPhases, secondaryMedian - secondaryStd, secondaryMedian + secondaryStd,
                             alpha = 0.15, color = stdColour, linewidth = 0)
    secondaryAx.plot(secondaryPhases, secondaryMedian, color = plotColour, linewidth = 2.2)

    secondaryAx.set_xlabel("Orbital Phase", fontsize = 9)
    secondaryAx.set_ylabel("Normalised Flux", fontsize = 9)
    secondaryAx.set_title("Secondary View", fontsize = 10, fontweight = "bold", loc = "left")
    secondaryAx.tick_params(labelsize = 8)

    # style all axes
    for ax in [globalAx, localAx, secondaryAx]:
        ax.set_facecolor("#ffffff")
        ax.grid(True, alpha = 0.2, linewidth = 0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

    outputPath = renderDir / f"{ticID}.png"
    fig.savefig(outputPath, dpi = 150, facecolor = fig.get_facecolor())
    plt.close(fig)

    print(f"Chart saved to {outputPath}")

    return outputPath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate diagnostic chart for a TIC target")
    parser.add_argument("ticID", type = int, help = "TIC ID to visualise")
    parser.add_argument("--predicted", type = int, default = None,
                        help = "Predicted label index (0=E, 1=S, 2=B, 3=J, 4=N)")

    args = parser.parse_args()

    generateChart(args.ticID, args.predicted)
