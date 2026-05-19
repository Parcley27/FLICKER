import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import lightkurve as lk
import logging
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import warnings
import wotan

globalBins = 201
localBins = 61
secondaryBins = 61

localTransitDurations = 4

detrendWindow = 0.5

bitmaskQuality = "default"

repoRoot = Path(__file__).resolve().parent.parent
rawDir = repoRoot / "data" / "raw"
processedDir = repoRoot / "data" / "processed"

lightcurveDir = rawDir / "lightcurves"
tceTable = rawDir / "tce_table.csv"

fetchLog = rawDir / "fetch_log.csv"
processLog = processedDir / "process_log.csv"

defaultOutput = processedDir / "dataset.h5"

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    stream = sys.stdout,

)

# less verbosity from lightkurve and astropy over small data loading issues (should cover <~1 pct)
warnings.filterwarnings("ignore", message=".*tpfmodel.*")
logging.getLogger("lightkurve").setLevel(logging.WARNING)
logging.getLogger("astropy").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Preprocess data for neural network")

    # 16 is probably good for i7 14700kf
    # 8 pcores plus 12 ecores - a few for other stuff
    # 6 for m2 pro mbp
    parser.add_argument("--workers", type = int, default = 4,
        help = "Parallel worker threads (default 4)")
    parser.add_argument("--limit", type = int, default = None,
        help = "Process only the first N raw files")
    parser.add_argument("--resume", action = "store_true",
        help = "Skip raw files already marked 'ok' in the fetch log")
    parser.add_argument("--output", type = Path, default = defaultOutput,
        help = "Store output at a specified path")
    
    args = parser.parse_args()

    return args

def loadLightCurve(ticID, row) -> tuple[np.ndarray, np.ndarray] | None:
    ticPath = lightcurveDir / f"{ticID}"

    fitsList = list(ticPath.rglob("*.fits"))

    if len(fitsList) == 0:
        logger.warning(f"No data found for TIC {ticID}")

        return None

    else:
        logger.info(f"Found {len(fitsList)} files for TIC {ticID}")

        sectors = []

        for fitsFile in fitsList:
            try:
                lightCurve = lk.io.read(fitsFile, quality_bitmask = bitmaskQuality) # type: ignore

                lcTime = lightCurve.time.value
                flux = lightCurve.flux.value

                mask = ~np.isnan(flux)

                lcTime, flux = lcTime[mask], flux[mask]

                sectors.append((lcTime, flux))
            
            except Exception as exception:
                logger.warning(f"TIC {ticID}: skipping {fitsFile.name} ({type(exception).__name__})")

        if len(sectors) == 0:
            logger.warning(f"No valid data for TIC {ticID}")

            return None

        times = np.concatenate([t for t, flux in sectors])
        fluxes = np.concatenate([flux for t, flux in sectors])

        # sort both arrays by time
        sortOrder = np.argsort(times)
        times, fluxes = times[sortOrder], fluxes[sortOrder]

        nFolds = (times[-1] - times[0]) / float(row["Period"])

        if nFolds < 3:
            logger.warning(f"TIC {ticID}: only {nFolds:.1f} folds (< 3), proceeding anyway")

        return (times, fluxes)

def detrend(times, fluxes, row) -> np.ndarray | None:
    period = float(row["Period"])
    epoch  = float(row["Epoch"])
    duration = float(row["Duration"])

    # max number of folds to consider
    nMinimum = np.floor((times[0] - epoch) / period)
    nMaximum = np.ceil((times[-1] - epoch) / period)

    # fold centers
    centres = epoch + np.arange(nMinimum, nMaximum + 1) * period

    # times (column vector) - centers (row vector) == 2d time difference matrix
    # np.any collapses to one boolean per time, true if transit is found
    transitMask = np.any(np.abs(times[:, None] - centres[None, :]) < 0.5 * duration, axis = 1)

    # check if transitmask is all true
    if np.all(transitMask):
        logger.warning("All data points are in transit, skipping...")

        return None
    
    try:
        # wotan settings based on Astronet V2 paper
        flatFlux = wotan.flatten(
            times, fluxes,
            mask = transitMask,
            method = "biweight",
            window_length = detrendWindow,

        )

    except Exception as exception:
        logger.warning(f"Error during detrending: {exception}")

        return None

    return flatFlux

def phaseFold(times, flatFlux, row) -> tuple[np.ndarray, np.ndarray]:
    period = float(row["Period"])
    epoch = float(row["Epoch"])

    # maps each timestamp to 0 ..< 1 based on its position in the expected orbit of the planet
    phases = ((times - epoch) % period) / period
    
    # transit goes at the middle rather than the edge, based off of astronet paper display 
    # numpy bool indexing
    phases[phases > 0.5] -= 1.0
    
    # sort phase and flatflux by ascending phase
    sortOrder = np.argsort(phases)
    phases, flatFlux = phases[sortOrder], flatFlux[sortOrder]

    return (phases, flatFlux)

def refineEpoch(phases, flatFlux, row) -> tuple[np.ndarray, np.ndarray, float]:
    duration = float(row["Duration"])
    period = float(row["Period"])

    # same sliding-window approach used in the secondary search
    # but applied to full phase-folded data to find the actual transit minimum
    windowHalfWidth = 0.5 * duration / period

    cumFlux = np.concatenate([[0.0], np.cumsum(flatFlux)])

    leftIndex = np.searchsorted(phases, phases - windowHalfWidth, side = "left")
    rightIndex = np.searchsorted(phases, phases + windowHalfWidth, side = "right")

    counts = np.maximum(rightIndex - leftIndex, 1)
    windowAverages = (cumFlux[rightIndex] - cumFlux[leftIndex]) / counts

    observedPhase = phases[np.argmin(windowAverages)]

    # shift so the transit lands at phase 0
    phases = phases - observedPhase

    # re-wrap to [-0.5, 0.5]
    phases[phases > 0.5] -= 1.0
    phases[phases < -0.5] += 1.0

    # re-sort by phase
    sortOrder = np.argsort(phases)
    phases, flatFlux = phases[sortOrder], flatFlux[sortOrder]

    return (phases, flatFlux, observedPhase)

def buildViews(phases, flatFlux, row) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float, float]:
    duration = float(row["Duration"])
    period = float(row["Period"])

    # whole orbit "global" view

    # "bin" edges needing n + 1 edges for n bins
    # creates 202 edges from -0.5 to 0.5 inclusive, so 201 bins of width 0.005
    globalBinEdges = np.linspace(-0.5, 0.5, globalBins + 1)

    # finds which bin each phase belongs to and builds an array of those indices
    # -1 shifts to 0 index
    globalBinIndices = np.digitize(phases, globalBinEdges) - 1
    np.clip(globalBinIndices, 0, globalBins - 1, out = globalBinIndices)

    medians, stds, hasData = [], [], []

    # compute median and std of all flatflux points where binindices == i
    for globalBin in range(globalBins):
        # gets all flatflux points in the current bin
        binFlux = flatFlux[globalBinIndices == globalBin]

        if len(binFlux) == 0:
            # flat baseline
            median = 1.0
            std = 0.0
            hasData.append(0.0)

        else:
            median = np.median(binFlux)
            std = np.std(binFlux)
            hasData.append(1.0)

        medians.append(median)
        stds.append(std)

    globalBinCentres = 0.5 * (globalBinEdges[:-1] + globalBinEdges[1:])

    # np bool array
    transitFlags = np.abs(globalBinCentres) < 1.5 * duration / period

    globalView = np.column_stack([medians, stds, transitFlags, hasData])

    # median of flux values for out of transit bins
    # actual "flat" baseline
    # fallback to all bins if duration >= period flags the entire orbit as in-transit
    outOfTransitGlobal = globalView[~transitFlags, 0]
    baseline = np.median(outOfTransitGlobal) if len(outOfTransitGlobal) > 0 else np.median(globalView[:, 0])

    globalView[:, 0] -= baseline

    # normalize -1 ... 1
    globalScaleFactor = np.min(globalView[:, 0])

    if globalScaleFactor < 0:
        globalScaleFactor = min(globalScaleFactor, -1e-6)
        globalView[:, 0] /= -globalScaleFactor # type: ignore
        globalView[:, 1] /= -globalScaleFactor # type: ignore

    # local view around the transit
    # +- 2 transit durations around phase 0
    halfWidth = (localTransitDurations / 2) * duration / period

    localMask = np.abs(phases) < halfWidth

    localPhases = phases[localMask]
    localFlux = flatFlux[localMask]

    localBinEdges = np.linspace(-halfWidth, halfWidth, localBins + 1)

    localBinIndices = np.digitize(localPhases, localBinEdges) - 1
    np.clip(localBinIndices, 0, localBins - 1, out = localBinIndices)

    medians, stds = [], []

    for localBin in range(localBins):
        binFlux = localFlux[localBinIndices == localBin]

        if len(binFlux) == 0:
            median = 1.0
            std = 0.0
        
        else:
            median = np.median(binFlux)
            std = np.std(binFlux)
        
        medians.append(median)
        stds.append(std)

    localView = np.column_stack([medians, stds])
    
    localBinCentres = 0.5 * (localBinEdges[:-1] + localBinEdges[1:])
    localTransitFlags = np.abs(localBinCentres) < 0.5 * duration / period

    outOfTransitLocal = localView[~localTransitFlags, 0]
    baseline = np.median(outOfTransitLocal) if len(outOfTransitLocal) > 0 else np.median(localView[:, 0])

    localView[:, 0] -= baseline

    localScaleFactor = np.min(localView[:, 0])

    if localScaleFactor < 0:
        localScaleFactor = min(localScaleFactor, -1e-6)
        localView[:, 0] /= -localScaleFactor
        localView[:, 1] /= -localScaleFactor

    # secondary view to find the deepest out of transit dip, either secondary elcipse or binary system
    # search only outside +-2 transit durations so cant refind the primary
    outOfTransitMask = np.abs(phases) > 2 * duration / period

    secondaryPhases = phases[outOfTransitMask]
    secondaryFlux = flatFlux[outOfTransitMask]

    if secondaryPhases.size == 0:
        # fallback as oppisite of transit window
        secondaryPhase = 0.5

    else:
        # slide a window across out-of-transit points, compute mean flux at each position
        # the position with the lowest mean is the secondary eclipse candidate
        windowWidth = duration / period
        searchHalfWidth = windowWidth / 2

        # cumulative sum
        # [a, b, c] -> [a, a + b, a + b + c]
        cumFlux = np.concatenate([[0.0], np.cumsum(secondaryFlux)])

        leftIndex = np.searchsorted(secondaryPhases, secondaryPhases - searchHalfWidth, side = "left")
        rightIndex = np.searchsorted(secondaryPhases, secondaryPhases + searchHalfWidth, side = "right")
        
        counts = np.maximum(rightIndex - leftIndex, 1)
        windowAverages = (cumFlux[rightIndex] - cumFlux[leftIndex]) / counts

        minimumIndex = np.argmin(windowAverages)

        secondaryPhase = secondaryPhases[minimumIndex]

    # bin a +- halfWidth window centred on the secondary phase, same resolution as local view
    # wrap around the phase boundary since phase is circular (-0.5 == +0.5)
    secondaryBinEdges = np.linspace(secondaryPhase - halfWidth, secondaryPhase + halfWidth, secondaryBins + 1)

    windowLow = secondaryPhase - halfWidth
    windowHigh = secondaryPhase + halfWidth

    if windowLow < -0.5:
        # left edge wraps past -0.5, pull data from right edge
        wrapMask = phases >= (windowLow + 1.0)
        mainMask = phases <= windowHigh
        secondaryWindowPhases = np.concatenate([phases[wrapMask] - 1.0, phases[mainMask]])
        secondaryWindowFlux = np.concatenate([flatFlux[wrapMask], flatFlux[mainMask]])

    elif windowHigh > 0.5:
        # right edge wraps past +0.5, pull data from left edge
        mainMask = phases >= windowLow
        wrapMask = phases <= (windowHigh - 1.0)
        secondaryWindowPhases = np.concatenate([phases[mainMask], phases[wrapMask] + 1.0])
        secondaryWindowFlux = np.concatenate([flatFlux[mainMask], flatFlux[wrapMask]])

    else:
        secondaryMask = (phases >= windowLow) & (phases <= windowHigh)
        secondaryWindowPhases = phases[secondaryMask]
        secondaryWindowFlux = flatFlux[secondaryMask]

    # sort wrapped data by phase for correct binning
    sortOrder = np.argsort(secondaryWindowPhases)
    secondaryWindowPhases = secondaryWindowPhases[sortOrder]
    secondaryWindowFlux = secondaryWindowFlux[sortOrder]

    secondaryBinIndices = np.digitize(secondaryWindowPhases, secondaryBinEdges) - 1
    np.clip(secondaryBinIndices, 0, secondaryBins - 1, out = secondaryBinIndices)

    medians, stds = [], []

    for secondaryBin in range(secondaryBins):
        binFlux = secondaryWindowFlux[secondaryBinIndices == secondaryBin]

        if len(binFlux) == 0:
            medians.append(1.0)
            stds.append(0.0)

        else:
            medians.append(np.median(binFlux))
            stds.append(np.std(binFlux))

    secondaryView = np.column_stack([medians, stds])

    # normalise: subtract out-of-transit baseline, scale so minimum = -1
    secondaryBinCentres = 0.5 * (secondaryBinEdges[:-1] + secondaryBinEdges[1:])
    secondaryTransitFlags = np.abs(secondaryBinCentres - secondaryPhase) < 0.5 * duration / period

    outOfTransitSecondary = secondaryView[~secondaryTransitFlags, 0]
    secondaryBaseline = np.median(outOfTransitSecondary) if len(outOfTransitSecondary) > 0 else np.median(secondaryView[:, 0])
    secondaryView[:, 0] -= secondaryBaseline

    secondaryScaleFactor = np.min(secondaryView[:, 0])

    if secondaryScaleFactor < 0:
        secondaryScaleFactor = min(secondaryScaleFactor, -1e-6)
        secondaryView[:, 0] /= -secondaryScaleFactor
        secondaryView[:, 1] /= -secondaryScaleFactor

    return (globalView, globalScaleFactor, localView, localScaleFactor, secondaryView, secondaryScaleFactor, secondaryPhase) # type: ignore

def processCurveEvent(args: tuple) -> dict:
    logging.getLogger(__name__).setLevel(logging.ERROR)

    ticID, ticIndex, row = args
    startTime = time.time()

    result = {
        "ticID": ticID,
        "ticIndex": ticIndex,
        "success": False,
        "error": None,
        "skipReason": None,

    }

    try:
        lightCurve = loadLightCurve(ticID, row)

        if lightCurve is None:
            result["skipReason"] = "no valid light curve data"

            return result

        times, fluxes = lightCurve

        flatFlux = detrend(times, fluxes, row)

        if flatFlux is None:
            result["skipReason"] = "detrending failed"

            return result

        validMask = ~np.isnan(flatFlux)
        times = times[validMask]
        flatFlux = flatFlux[validMask]

        phases, flatFlux = phaseFold(times, flatFlux, row)

        phases, flatFlux, _ = refineEpoch(phases, flatFlux, row)

        globalView, globalScaleFactor, localView, localScaleFactor, secondaryView, secondaryScaleFactor, secondaryPhase = buildViews(phases, flatFlux, row)

        period = float(row["Period"])
        duration = float(row["Duration"])
        depth = float(row["Depth"])
        tBandMagnitude = float(row["Tmag"]) if pd.notna(row["Tmag"]) else 0.0
        epoch = float(row["Epoch"])
        split = str(row["Split"])

        # fall back to SRadEst (paper's Bayesian estimate) before defaulting to NaN
        stellarMass = float(row["SMass"]) if pd.notna(row["SMass"]) else np.nan
        stellarRadius = (
            float(row["SRad"]) if pd.notna(row["SRad"])
            else float(row["SRadEst"]) if pd.notna(row["SRadEst"])
            else np.nan
        )

        nPoints = np.log1p(len(times))
        timeBaseline = times[-1] - times[0]
        nFolds = np.log1p(min(np.floor(timeBaseline / period), 100)) / np.log1p(100)

        scalars = np.array([
            period, duration, depth, tBandMagnitude, stellarMass, stellarRadius,
            nFolds, nPoints, globalScaleFactor, localScaleFactor,
            secondaryScaleFactor, secondaryPhase

        ], dtype=np.float32)

        consensusLabel = row["Consensus Label"]
        labelMap = {"E": 0, "S": 1, "B": 2, "J": 3, "N": 4}
        label = np.int8(labelMap[consensusLabel] if pd.notna(consensusLabel) and consensusLabel in labelMap else -1)
        exoplanetLabel = np.int8(1 if label == 0 else 0)

        result.update({
            "success": True,
            "globalView": globalView.astype(np.float32),
            "localView": localView.astype(np.float32),
            "secondaryView": secondaryView.astype(np.float32),
            "scalars": scalars,
            "label": label,
            "exoplanetLabel": exoplanetLabel,
            "period": period,
            "epoch": epoch,
            "split": split,
            "elapsed": time.time() - startTime,

        })

    except Exception as exception:
        result["error"] = str(exception)

    return result

def main():
    args = parseArgs()

    eventsData = pd.read_csv(tceTable, comment = "#")

    # drop the units row (row 0) and reset index
    assert eventsData.iloc[0]["Epoch"] == "BTJD", (
        f"Expected units row with Epoch='BTJD', got '{eventsData.iloc[0]['Epoch']}'. "
        "The CSV format may have changed - check whether the units row still exists."
    )

    eventsData = eventsData.drop(0).reset_index(drop = True)

    # convert tic id row to int
    eventsData["TIC ID"] = eventsData["TIC ID"].astype(int)

    # Recover labels for NaN consensus rows via majority vote across individual vetter columns (L1–L8)
    # exclude ties
    vetterCols = [c for c in eventsData.columns if c.startswith("L")]

    def majorityVote(row):
        # Collect all valid (non-NaN, recognised) votes for this row
        votes = [row[col] for col in vetterCols
                 if pd.notna(row[col]) and row[col] in ["E", "S", "B", "J", "N"]]
        
        if not votes:
            return np.nan
        
        counts = pd.Series(votes).value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            return np.nan
        
        return counts.index[0]

    nanMask = eventsData["Consensus Label"].isna()
    eventsData.loc[nanMask, "Consensus Label"] = eventsData[nanMask].apply(majorityVote, axis=1)

    logger.info(
        f"Majority vote: resolved {nanMask.sum()} NaN consensus labels, "
        f"{eventsData['Consensus Label'].isna().sum()} ties"

    )

    logger.info(f"Loaded {len(eventsData)} transit events from tce_table.csv")

    # load fetch log
    fetchLogData = pd.read_csv(fetchLog)

    logger.info(f"Loaded {len(fetchLogData)} entries from fetch_log.csv")

    fetchLogData = fetchLogData[fetchLogData["status"] == "ok"]
    logger.info(f"{len(fetchLogData)} entries with 'ok' status")

    availableStars = set(fetchLogData["ticID"])

    eventsData = eventsData[eventsData["TIC ID"].isin(availableStars)]
    logger.info(f"{len(eventsData)} star transit events with available lightcurves")

    tceList = []

    for ticID, group in eventsData.groupby("TIC ID"):
        for tceIndex, (_, row) in enumerate(group.iterrows()):
            tceList.append((ticID, tceIndex, row))

    logger.info(f"Built TCE list with {len(tceList)} entries")

    if args.resume and args.output.exists():
        doneKeys = set()

        # filter out ones already in the output file
        with h5py.File(args.output, "r") as database:
            for starGroup in database.values():
                for tceKey in starGroup.keys():
                    doneKeys.add(f"{starGroup.name.lstrip('/')}/{tceKey}")

        logger.info(f"Found {len(doneKeys)} already written, skipping")
        tceList = [(ticID, ticIndex, row) for ticID, ticIndex, row in tceList if f"{ticID}/{ticIndex}" not in doneKeys]

    if args.limit is not None:
        tceList = tceList[:args.limit]
        logger.info(f"Limiting to first {args.limit} entries")
    
    # creates output dir if doesn't exist
    args.output.parent.mkdir(parents = True, exist_ok = True)\
    
    numberProcessed = 0
    numberSkipped = 0
    completed = 0
    totalTCEs = len(tceList)

    labelCounts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, -1: 0}

    mainStartTime = time.time()

    results = []

    with ProcessPoolExecutor(max_workers = args.workers) as executor:
        futures = [executor.submit(processCurveEvent, args) for args in tceList]

        with h5py.File(args.output, "a") as database:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                ticID = result["ticID"]
                ticIndex = result["ticIndex"]

                if result["success"]:
                    group = database.create_group(f"{ticID}/{ticIndex}")

                    group.create_dataset("globalView", data = result["globalView"])
                    group.create_dataset("localView", data = result["localView"])
                    group.create_dataset("secondaryView", data = result["secondaryView"])
                    group.create_dataset("scalars", data = result["scalars"])
                    group.create_dataset("label", data = result["label"])
                    group.create_dataset("exoplanetLabel", data = result["exoplanetLabel"])

                    group.attrs["ticID"] = ticID
                    group.attrs["period"] = result["period"]
                    group.attrs["epoch"] = result["epoch"]
                    group.attrs["split"] = result["split"]

                    numberProcessed += 1
                    labelCounts[int(result["label"])] += 1

                else:
                    numberSkipped += 1

                completed += 1

                try:
                    sys.stdout.write(
                        f"\r  {completed:>6,}/{totalTCEs:,}  ({100 * completed / totalTCEs:5.1f}%) | " 
                        f"# Ok: {numberProcessed:,}  | # Skipped: {numberSkipped} "

                    )

                    sys.stdout.flush()

                except ValueError:
                    pass

    try:
        sys.stdout.write("\n\n")
        sys.stdout.flush()

    except ValueError:
        pass

    # log failure reason summary
    skipReasons = {}
    errorMessages = {}

    for r in results:
        if r["success"]:
            continue

        if r["skipReason"]:
            skipReasons[r["skipReason"]] = skipReasons.get(r["skipReason"], 0) + 1

        if r["error"]:
            errorMessages[r["error"]] = errorMessages.get(r["error"], 0) + 1

    if skipReasons:
        logger.info("Skip reasons:")

        for reason, count in sorted(skipReasons.items(), key = lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")

    if errorMessages:
        logger.info("Errors:")

        for error, count in sorted(errorMessages.items(), key = lambda x: -x[1]):
            logger.info(f"  {error}: {count}")

    # write process log
    processLogFields = ("ticID", "ticIndex", "status", "reason", "elapsed")

    with open(processLog, "w", newline = "") as logFile:
        writer = csv.DictWriter(logFile, fieldnames = processLogFields)
        writer.writeheader()

        for r in results:
            status = "ok" if r["success"] else "error" if r["error"] else "skipped"
            reason = r["error"] or r["skipReason"] or ""
            elapsed = f"{r.get('elapsed', 0):.2f}" if r["success"] else ""

            writer.writerow({
                "ticID": r["ticID"],
                "ticIndex": r["ticIndex"],
                "status": status,
                "reason": reason,
                "elapsed": elapsed,
            })

    logger.info(f"Process log written to {processLog}")

    # normalize scalars
    trainingScalars = []

    with h5py.File(args.output, "r") as database:
        for starGroup in database.values():
            for tceGroup in starGroup.values():
                if tceGroup.attrs["split"] == "train":
                    trainingScalars.append(tceGroup["scalars"][:])

    if len(trainingScalars) == 0:
        logger.warning("No training TCEs found so skipping scalar normalisation")

    else:
        trainingScalars = np.array(trainingScalars)  # (n_train, 12)

        # compute mean/std ignoring NaN (from missing stellar params)
        scalarMean = np.nanmean(trainingScalars, axis = 0)
        scalarStd = np.nanstd(trainingScalars, axis = 0)
        scalarStd[scalarStd == 0] = 1.0  # guard against constant features

        nanCounts = np.isnan(trainingScalars).sum(axis = 0)
        nanFeatures = np.where(nanCounts > 0)[0]

        if len(nanFeatures) > 0:
            logger.info(f"NaN counts per scalar feature: {dict(zip(nanFeatures.tolist(), nanCounts[nanFeatures].tolist()))}")

        scalarStats = {"mean": scalarMean.tolist(), "std": scalarStd.tolist()}
        statsPath = processedDir / "scalar_stats.json"
        statsPath.write_text(json.dumps(scalarStats, indent = 2))
        logger.info(f"Scalar stats written to {statsPath}")

        # replace NaN with training mean then z-score - missing values land at z = 0
        with h5py.File(args.output, "r+") as database:
            for starGroup in database.values():
                for tceGroup in starGroup.values():
                    raw = tceGroup["scalars"][:]

                    nanMask = np.isnan(raw)

                    if np.any(nanMask):
                        raw[nanMask] = scalarMean[nanMask]

                    normalised = ((raw - scalarMean) / scalarStd).astype(np.float32)

                    del tceGroup["scalars"]

                    tceGroup.create_dataset("scalars", data = normalised)

        logger.info("Scalars normalised and written back to HDF5")

    mainEndTime = time.time()

    # summary stats
    labelNames = {0: "Exoplanet", 1: "Single Transit", 2: "Binary System", 3: "Junk", 4: "Not Sure", -1: "Unlabeled"}
    total = numberProcessed + numberSkipped

    print()
    logger.info("* Summary *")
    logger.info(f"Total: {total}, Processed: {numberProcessed} ({numberProcessed / total:.1%}), Skipped: {numberSkipped} ({numberSkipped / total:.1%})")

    for labelIndex, name in labelNames.items():
        count = labelCounts[labelIndex]
        percent = 100 * count / numberProcessed if numberProcessed > 0 else 0.0

        logger.info(f"  {name}: {count} ({percent:.1f}%)")

    outputSizeMB = args.output.stat().st_size / (1024 * 1024)
    outputSizeGB = args.output.stat().st_size / (1024 * 1024 * 1024)

    logger.info(f"Output file size: {outputSizeMB:.2f} MB / {outputSizeGB:.2f} GB")

    logger.info(f"Total processing time: {mainEndTime - mainStartTime:.2f} seconds")
    workerTimes = [r["elapsed"] for r in results if r["success"]]
    avgWorkerTime = sum(workerTimes) / len(workerTimes) if workerTimes else 0.0
    logger.info(f"Averaged {avgWorkerTime:.2f} seconds per TCE (worker time)")

    summaryLines = [
        "* Summary *",
        f"Total: {total}, Processed: {numberProcessed} ({numberProcessed / total:.1%}), Skipped: {numberSkipped} ({numberSkipped / total:.1%})",
   
    ]
    for labelIndex, name in labelNames.items():
        count = labelCounts[labelIndex]
        percent = 100 * count / numberProcessed if numberProcessed > 0 else 0.0
        summaryLines.append(f"  {name}: {count} ({percent:.1f}%)")

    summaryLines.append(f"Output file size: {outputSizeMB:.2f} MB / {outputSizeGB:.2f} GB")
    summaryLines.append(f"Total processing time: {mainEndTime - mainStartTime:.2f} seconds")
    summaryLines.append(f"Averaged {avgWorkerTime:.2f} seconds per TCE (worker time)")

    summaryPath = args.output.parent / "summary.txt"
    
    with open(summaryPath, "w") as f:
        f.write("\n".join(summaryLines) + "\n")

    logger.info(f"Summary written to {summaryPath}")

if __name__ == "__main__":
    main()
