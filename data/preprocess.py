import argparse
import logging
import h5py
import pandas as pd
import sys

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
    logger.info(f"{len(fetchLogData)} entries with status 'ok'")

    availableStars = set(fetchLogData["ticID"])

    eventsData = eventsData[eventsData["TIC ID"].isin(availableStars)]
    logger.info(f"{len(eventsData)} star transit events with available lightcurves")

    tceList = []

    for ticId, group in eventsData.groupby("TIC ID"):
        for tceIndex, (_, row) in enumerate(group.iterrows()):
            tceList.append((ticId, tceIndex, row))

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
        pass

    print("Part 1 done...")

if __name__ == "__main__":
    main()
