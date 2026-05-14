import argparse
import logging
import h5py
import pandas as pd
import numpy as np
import sys
import lightkurve as lk
import wotan
import time

from pathlib import Path

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

# less verbosity from lightcuve over small data loading issues (should cover <~1 pct)
logging.getLogger("lightkurve").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Preprocess data for neural network")

    parser.add_argument("--delete-fits", action = "store_true",
        help = "Delete raw files after completion")
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
                lightCurve =lk.io.read(fitsFile, quality_bitmask = bitmaskQuality)

                time = lightCurve.time.value
                flux = lightCurve.flux.value

                mask = ~np.isnan(flux)

                time, flux = time[mask], flux[mask]

                sectors.append((time, flux))
            
            except Exception as exception:
                logger.warning(f"Error loading {fitsFile} for TIC {ticID}: {exception}")

        if len(sectors) == 0:
            logger.warning(f"No valid data for TIC {ticID}")

            return None

        times = np.concatenate([time for time, flux in sectors])
        fluxes = np.concatenate([flux for time, flux in sectors])

        # sort both arrays by time
        sortOrder = np.argsort(times)
        times, fluxes = times[sortOrder], fluxes[sortOrder]

        nFolds = (times[-1] - times[0]) / float(row["Period"])

        if nFolds < 3:
            logger.warning(f"Not enough folds for TIC {ticID}, skipping...")

            return None
        
        return (times, fluxes)

def detrend(times, fluxes, row) -> tuple[np.ndarray, np.ndarray] | None:
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
        flatFlux, trend = wotan.flatten(
            times, fluxes, 
            mask = transitMask, 
            method = "biweight",
            window_length = detrendWindow,
            return_trend = True,

        )
    
    except Exception as exception:
        logger.warning(f"Error during detrending: {exception}")

        return None
    
    return (flatFlux, trend)

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

    medians, stds = [], []

    # compute median and std of all flatflux points where binindices == i
    for globalBin in range(globalBins):
        # gets all flatflux points in the current bin
        binFlux = flatFlux[globalBinIndices == globalBin]

        if len(binFlux) == 0:
            # flat baseline
            median = 1.0
            std = 0.0
        
        else:
            median = np.median(binFlux)
            std = np.std(binFlux)
        
        medians.append(median)
        stds.append(std)

    globalBinCentres = 0.5 * (globalBinEdges[:-1] + globalBinEdges[1:])

    # np bool array
    transitFlags = np.abs(globalBinCentres) < 0.5 * duration / period

    globalView = np.column_stack([medians, stds, transitFlags])

    # median of flux values for out of transit bins
    # actual "flat" baseline
    baseline = np.median(globalView[~transitFlags, 0])

    globalView[:, 0] -= baseline

    # normalize -1 ... 1
    globalScaleFactor = np.min(globalView[:, 0])

    if globalScaleFactor < 0:
        # min = -1
        globalView[:, 0] /= -globalScaleFactor

    # local view around the transit
    # +- 2 transit durations around phase 0
    halfWidth = (localTransitDurations / 2) * duration / period

    localMask = np.abs(phases) < halfWidth

    localPhases = phases[localMask]
    localFlux = flatFlux[localMask]

    localBinEdges = np.linspace(-halfWidth, halfWidth, localBins + 1)

    localBinIndices = np.digitize(localPhases, localBinEdges) - 1

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

    baseline = np.median(localView[~localTransitFlags, 0])

    localView[:, 0] -= baseline

    localScaleFactor = np.min(localView[:, 0])

    if localScaleFactor < 0:
        localView[:, 0] /= -localScaleFactor
    
    # secondary view to find the deepest out of transit dip, either secondary elcipse or binary system
    # search only outside +-2 transit durations so cant refind the primary
    outOfTransitMask = np.abs(phases) > 2 * duration / period

    secondaryPhases = phases[outOfTransitMask]
    secondaryFlux = flatFlux[outOfTransitMask]

    if secondaryPhases.size == 0:
        # fallback as oppisite of transit window
        secondaryPhase = 0.5

    else:
        windowWidth = duration / period

        # slide a window across out-of-transit points, compute mean flux at each position
        # the position with the lowest mean is the secondary eclipse candidate
        windowAverages = []

        for phase in secondaryPhases:
            windowAverageFlux = np.mean(secondaryFlux[np.abs(secondaryPhases - phase) < windowWidth / 2])
            windowAverages.append(windowAverageFlux)

        minimumIndex = np.argmin(windowAverages)
        secondaryPhase = secondaryPhases[minimumIndex]

    # bin a +- halfWidth window centred on the secondary phase, same resolution as local view
    secondaryBinEdges = np.linspace(secondaryPhase - halfWidth, secondaryPhase + halfWidth, secondaryBins + 1)

    secondaryMask = (phases >= secondaryPhase - halfWidth) & (phases <= secondaryPhase + halfWidth)
    secondaryWindowPhases = phases[secondaryMask]
    secondaryWindowFlux = flatFlux[secondaryMask]

    secondaryBinIndices = np.digitize(secondaryWindowPhases, secondaryBinEdges) - 1

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

    secondaryBaseline = np.median(secondaryView[~secondaryTransitFlags, 0])
    secondaryView[:, 0] -= secondaryBaseline

    secondaryScaleFactor = np.min(secondaryView[:, 0])

    if secondaryScaleFactor < 0:
        secondaryView[:, 0] /= -secondaryScaleFactor

    return (globalView, globalScaleFactor, localView, localScaleFactor, secondaryView, secondaryScaleFactor, secondaryPhase)

def main():
    args = parseArgs()

    eventsData = pd.read_csv(tceTable, comment = "#")

    # drop first row and reset index
    eventsData = eventsData.drop(0).reset_index(drop = True)

    # convert tic id row to int
    eventsData["TIC ID"] = eventsData["TIC ID"].astype(int)

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

    if args.limit != None:
        tceList = tceList[:args.limit]
        logger.info(f"Limiting to first {args.limit} entries")
    
    # creates output dir if doesn't exist
    args.output.parent.mkdir(parents = True, exist_ok = True)

    with h5py.File(args.output, "a") as database:
        for ticID, ticIndex, row in tceList:
            startTime = time.time()

            lightCurve = loadLightCurve(ticID, row)

            if lightCurve == None:
                logger.warning(f"Skipping TIC {ticID} TCE {ticIndex} due to load failure")
            
            else:
                times, fluxes = lightCurve
                logger.info(f"TIC {ticID}/{tceIndex}: {len(times)} instances loaded")

                detrendResult = detrend(times, fluxes, row)

                if detrendResult == None:
                    logger.warning(f"Skipping TIC {ticID} TCE {ticIndex} because of detrend failure...")
                
                else:
                    flatFlux, trend = detrendResult

                    logger.info(f"TIC {ticID}/{tceIndex} properly detrended")

                    phases, flatFlux = phaseFold(times, flatFlux, row)
                    logger.info(f"TIC {ticID}/{tceIndex} folded into {len(phases)} phases")

                    globalView, globalScaleFactor, localView, localScaleFactor, secondaryView, secondaryScaleFactor, secondaryPhase = buildViews(phases, flatFlux, row)
                    logger.info(f"TIC {ticID}/{tceIndex} views built: global {globalView.shape}, local {localView.shape}, secondary {secondaryView.shape}")

                    # Transit data
                    period = float(row["Period"])
                    duration = float(row["Duration"])
                    depth = float(row["Depth"])

                    # Stellar data
                    tBandMagnitude = float(row["Tmag"]) if pd.notna(row["Tmag"]) else 0.0
                    stellarMass = float(row["SMass"]) if pd.notna(row["SMass"]) else 0.0
                    stellarRadius = float(row["SRad"]) if pd.notna(row["SRad"]) else 0.0

                    # total instances after quality mask
                    nPoints = len(times)

                    timeBaseline = times[-1] - times[0]

                    nFolds = min(np.floor(timeBaseline / period), 100)
                    nFolds = np.log1p(nFolds)


                    scalars = np.array([
                        period, duration, depth, tBandMagnitude, stellarMass, stellarRadius,
                        nFolds, nPoints, globalScaleFactor, localScaleFactor,
                        secondaryScaleFactor, secondaryPhase

                    ], dtype=np.float32)

                    # Exoplanet transit, single transit, binary system, junk, not sure
                    consensusLabel = row["Consensus Label"]
                    labelMap = {"E": 0, "S": 1, "B": 2, "J": 3, "N": 4}

                    # mapped labels
                    label = np.int8(labelMap[consensusLabel] if pd.notna(consensusLabel) and consensusLabel in labelMap else -1)
                    exoplanetLabel = np.int8(1 if label == 0 else 0 ) 

            endTime = time.time()
            elapsedTime = endTime - startTime # seconds
            logger.info(f"TIC {ticID}/{tceIndex} processed in {elapsedTime:.3f}s")

if __name__ == "__main__":
    main()
