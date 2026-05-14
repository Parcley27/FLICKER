import argparse
import logging
import h5py
import pandas as pd
import numpy as np
import sys
import lightkurve as lk

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

def parseArgs():
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

def loadLightCurve(ticID, row):
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
        sortIndices = np.argsort(times)
        times, fluxes = times[sortIndices], fluxes[sortIndices]

        nFolds = (times[-1] - times[0]) / float(row["Period"])

        if nFolds < 3:
            logger.warning(f"Not enough folds for TIC {ticID}, skipping...")

            return None
        
        return (times, fluxes)


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
            lightCurve = loadLightCurve(ticID, row)

            if lightCurve is None:
                logger.warning(f"Skipping TIC {ticID} TCE {ticIndex} due to load failure")
            
            else:
                times, fluxes = lightCurve
                logger.info(f"TIC {ticID}/{tceIndex}: {len(times)} instances loaded")

if __name__ == "__main__":
    main()
