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
    if not scalarStatsPath.exists():
        # no stats file means normalisation was skipped - scalars are already raw
        return scalars.copy()

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


scalarNames = [
    "Period", "Duration", "Depth", "Tmag", "Stellar Mass", "Stellar Radius",
    "nFolds", "nPoints", "Global Scale", "Local Scale",
    "Secondary Scale", "Secondary Phase",
]

scalarUnits = [
    "d", "d", "ppm", "mag", "M\u2609", "R\u2609",
    "", "", "", "",
    "", "phase",
]


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
        halfPeriodView = group["halfPeriodView"][:]
        scalars = group["scalars"][:]
        label = int(group["label"][()])
        exoplanetLabel = int(group["exoplanetLabel"][()])
        period = group.attrs["period"]
        epoch = group.attrs["epoch"]

    raw = denormalizeScalars(scalars)

    # phase axes for each view
    globalPhases = np.linspace(-0.5, 0.5, 201)
    duration = raw[1]
    halfWidth = 2 * duration / period
    localPhases = np.linspace(-halfWidth, halfWidth, 61)
    secondaryPhase = raw[11]
    secondaryPhases = np.linspace(secondaryPhase - halfWidth, secondaryPhase + halfWidth, 61)
    halfPeriodHalfWidth = 2 * duration / (period / 2)
    halfPeriodPhases = np.linspace(-halfPeriodHalfWidth, halfPeriodHalfWidth, 61)

    # colours
    plotColour = "#4a90d9"
    stdColour = "#4a90d9"
    transitColour = "#d94a4a"

    fig = plt.figure(figsize = (14, 16), facecolor = "#FFFFFF")

    # scalar panel anchored at the top
    scalarSpec = gridspec.GridSpec(1, 1, top = 0.94, bottom = 0.78, left = 0.08, right = 0.96)

    # plot panels independently positioned, shifted down by half a row
    plotOuter = gridspec.GridSpec(3, 1, height_ratios = [1, 1, 1], hspace = 0.35,
                                  top = 0.68, bottom = 0.05, left = 0.08, right = 0.96)

    # scalar info panel
    scalarAx = fig.add_subplot(scalarSpec[0])
    scalarAx.axis("off")

    trueLabelStr = f"{labelNames.get(label, '?')} ({labelShort.get(label, '?')})"
    titleLine = f"TIC {ticID}    |    Label: {trueLabelStr}"

    if predictedLabel is not None:
        predStr = f"{labelNames.get(predictedLabel, predictedLabel)} ({labelShort.get(predictedLabel, predictedLabel)})"
        titleLine += f"    |    Predicted: {predStr}"

    scalarAx.text(0.5, 0.97, titleLine, transform = scalarAx.transAxes,
                  ha = "center", va = "top", fontsize = 22, fontweight = "bold",
                  fontfamily = "monospace", color = "#333333")

    # display all 12 scalars in a 3x4 grid
    nCols = 4

    for idx in range(12):
        col = idx % nCols
        row = idx // nCols
        x = (col + 0.5) / nCols
        y = 0.72 - row * 0.42

        value = raw[idx]
        unit = scalarUnits[idx]

        if np.isnan(value):
            valStr = "N/A"
        elif abs(value) >= 1000:
            valStr = f"{value:.0f}"
        elif abs(value) >= 1:
            valStr = f"{value:.4f}"
        else:
            valStr = f"{value:.6f}"

        if unit:
            valStr += f" {unit}"

        # normalised value in parentheses
        normVal = scalars[idx]
        normStr = f"(norm: {normVal:.3f})" if not np.isnan(normVal) else "(norm: N/A)"

        scalarAx.text(x, y, scalarNames[idx], transform = scalarAx.transAxes,
                      ha = "center", va = "top", fontsize = 14, fontweight = "bold",
                      fontfamily = "monospace", color = "#555555")
        scalarAx.text(x, y - 0.12, valStr, transform = scalarAx.transAxes,
                      ha = "center", va = "top", fontsize = 14,
                      fontfamily = "monospace", color = "#222222")
        scalarAx.text(x, y - 0.24, normStr, transform = scalarAx.transAxes,
                      ha = "center", va = "top", fontsize = 10,
                      fontfamily = "monospace", color = "#888888")

    # global view (full width)
    globalAx = fig.add_subplot(plotOuter[0])

    globalMedian = globalView[:, 0]
    globalStd = np.clip(globalView[:, 1], 0, 0.5)
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
    globalAx.set_title("Global View (201 bins)", fontsize = 10, fontweight = "bold", loc = "left")
    globalAx.tick_params(labelsize = 8)
    globalAx.set_xlim(-0.5, 0.5)

    # middle row: local and secondary side by side
    middleInner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = plotOuter[1], wspace = 0.25)

    # local view
    localAx = fig.add_subplot(middleInner[0])

    localMedian = localView[:, 0]
    localStd = np.clip(localView[:, 1], 0, 0.5)

    localAx.fill_between(localPhases, localMedian - localStd, localMedian + localStd,
                         alpha = 0.15, color = stdColour, linewidth = 0)
    localAx.plot(localPhases, localMedian, color = plotColour, linewidth = 2.2)

    localAx.set_xlabel("Orbital Phase", fontsize = 9)
    localAx.set_ylabel("Normalised Flux", fontsize = 9)
    localAx.set_title("Local View (61 bins)", fontsize = 10, fontweight = "bold", loc = "left")
    localAx.tick_params(labelsize = 8)

    # secondary view
    secondaryAx = fig.add_subplot(middleInner[1])

    secondaryMedian = secondaryView[:, 0]
    secondaryStd = np.clip(secondaryView[:, 1], 0, 0.5)

    secondaryAx.fill_between(secondaryPhases, secondaryMedian - secondaryStd, secondaryMedian + secondaryStd,
                             alpha = 0.15, color = stdColour, linewidth = 0)
    secondaryAx.plot(secondaryPhases, secondaryMedian, color = plotColour, linewidth = 2.2)

    secondaryAx.set_xlabel("Orbital Phase", fontsize = 9)
    secondaryAx.set_ylabel("Normalised Flux", fontsize = 9)
    secondaryAx.set_title("Secondary View (61 bins)", fontsize = 10, fontweight = "bold", loc = "left")
    secondaryAx.tick_params(labelsize = 8)

    # bottom row: half-period view
    halfPeriodAx = fig.add_subplot(plotOuter[2])

    halfPeriodMedian = halfPeriodView[:, 0]
    halfPeriodStd = np.clip(halfPeriodView[:, 1], 0, 0.5)

    halfPeriodAx.fill_between(halfPeriodPhases, halfPeriodMedian - halfPeriodStd, halfPeriodMedian + halfPeriodStd,
                              alpha = 0.15, color = stdColour, linewidth = 0)
    halfPeriodAx.plot(halfPeriodPhases, halfPeriodMedian, color = plotColour, linewidth = 2.2)

    halfPeriodAx.set_xlabel("Half-Period Phase", fontsize = 9)
    halfPeriodAx.set_ylabel("Normalised Flux", fontsize = 9)
    halfPeriodAx.set_title("Half-Period View (61 bins)", fontsize = 10, fontweight = "bold", loc = "left")
    halfPeriodAx.tick_params(labelsize = 8)

    # style all axes
    for ax in [globalAx, localAx, secondaryAx, halfPeriodAx]:
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
