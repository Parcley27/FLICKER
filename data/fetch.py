# Written with Claude Code

import argparse
import csv
import hashlib
import json
import logging
import ssl
import sys
import time
import urllib.request

try:
    import certifi
    
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile = certifi.where())

except ImportError:
    pass  # certifi not installed; rely on system certificates

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import lightkurve as lk

except ImportError:
    lk = None

repoRoot = Path(__file__).resolve().parent.parent
rawDir = repoRoot / "data" / "raw"
lightcurveDir = rawDir / "lightcurves"
csvDestination = rawDir / "tce_table.csv"
logPath = rawDir / "fetch_log.csv"

zenodoRecordID = "7411579"
zenodoApiUrl = f"https://zenodo.org/api/records/{zenodoRecordID}"

lightcurveAuthor = "QLP"
lightcurveMission = "TESS"

# Log messenger
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    stream = sys.stdout,

)

logger = logging.getLogger(__name__)

# Lightkurve is very chatty about quality masks so silence it below WARNING
logging.getLogger("lightkurve").setLevel(logging.WARNING)

# Complete record metedata
# list of files, download urls, and checksums
def fetchZenodoMetadata() -> dict:
    logger.info(f"Getting record {zenodoRecordID} from Zenodo API...")

    try:
        with urllib.request.urlopen(zenodoApiUrl, timeout = 30) as response:
            return json.loads(response.read())
    
    except Exception as error:
        logger.error(f"Could not reach api endpoint: {error}")

        sys.exit(1)

# Hash 1mb per chunk to compute checksum rather than complete file
## 1 << 20 is a bitshift that evaluates numerically to 1MB
def computeMD5Checksum(path: Path, chunkSize: int = 1 << 20) -> str:
    hasher = hashlib.md5()

    with open(path, "rb") as file:
        while dataChunk := file.read(chunkSize):
            hasher.update(dataChunk)
    
    return hasher.hexdigest()

# Callable class, runs after every new data packet is received
class DownloadProgress:
    def __init__(self, label: str):
        self.label = label
        self.startTime = time.time()

    def __call__(self, blocksReceived: int, blockSize: int, totalBytes: int):
        if totalBytes <= 0:
            return

        bytesDownloaded = min(blocksReceived * blockSize, totalBytes)
        percentComplete = 100.0 * bytesDownloaded / totalBytes

        timeElapsed = max(time.time() - self.startTime, 1e-6) # avoid division by 0
        downloadSpeed = bytesDownloaded / timeElapsed / (1024 ** 2)

        filledBlocks = int(percentComplete / 4)
        progressBar = "#" * filledBlocks + '-' * (25 - filledBlocks)

        sys.stdout.write(
            f"\r[{progressBar}] {percentComplete:5.1f}%  {downloadSpeed:.1f} MB/s  {self.label}"
        )
        sys.stdout.flush()

        if bytesDownloaded >= totalBytes:
            sys.stdout.write("\n")
            sys.stdout.flush()

def downloadZenodoCsv(recordFiles: dict) -> bool:
    csvFiles = {name: info for name, info in recordFiles.items() if name.endswith(".csv")}

    if not csvFiles:
        logger.error("No CSV file found in Zenodo record")
        
        return False

    fileName = next(
        (name for name in csvFiles if "tce" in name.lower()),
        next(iter(csvFiles))

    )

    fileInfo = csvFiles[fileName]
    downloadUrl = fileInfo["links"]["self"]

    expectedChecksum = fileInfo.get("checksum", "").replace("md5:", "") or None

    if csvDestination.exists():
        if expectedChecksum and computeMD5Checksum(csvDestination) != expectedChecksum:
            logger.warning("Existing label table checksum mismatch - deleting and re-downloading")
            csvDestination.unlink()

        else:
            logger.info("Label table already present and verified - skipping download")

            return True
    
    logger.info(f"Downloading TCE label table ({fileName})")
    rawDir.mkdir(parents = True, exist_ok = True)

    tempFile = csvDestination.with_suffix(".csv.part")

    try:
        urllib.request.urlretrieve(downloadUrl, tempFile, reporthook = DownloadProgress(fileName))
        tempFile.rename(csvDestination)

    except Exception as error:
        logger.error(f"Download failed: {error}")

        if tempFile.exists():
            tempFile.unlink()

        return False
    
    if expectedChecksum and computeMD5Checksum(csvDestination) != expectedChecksum:
        logger.error("Checksum failed after download - file may be corrupt")
        csvDestination.unlink()

        return False
    
    logger.info(f"Label table saved to {csvDestination.relative_to(repoRoot)}")

    return True

def loadLabelTable() -> list[dict]:
    with open(csvDestination, newline = "") as file:
        # The Zenodo CSV has leading comment lines starting with "#".
        # Filter them out so DictReader sees the real header row first.
        dataLines = (line for line in file if not line.startswith("#"))
        rows = list(csv.DictReader(dataLines))

    logger.info(f"Loaded {len(rows):,} TCEs from label table")

    return rows

def extractUniqueTicIDs(rows: list[dict]) -> list[str]:
    for columnName in ("TIC ID", "tic_id", "TIC_ID", "ticid", "TICID", "TIC", "tic"):
        if columnName in rows[0]:
            ticIDs = list(dict.fromkeys(
                str(row[columnName]).strip() for row in rows
                if str(row[columnName]).strip()
            ))

            logger.info(f"Found {len(ticIDs):,} unique TIC IDs (column: '{columnName}').")

            return ticIDs
    
    logger.error(
        f"Cannot find TIC ID column. Available columns: {list(rows[0].keys())}\n"
        "Update extractUniqueTicIDs() to match the actual column name"

    )

    sys.exit(1)

# These become dict keys throughout the rest of the script
logFields = ("ticID", "status", "fileCount", "sectors", "timestamp", "note")


def loadFetchLog() -> dict[str, dict]:
    # Returns a dict keyed by TIC ID so we can do O(1) lookups when deciding which stars to skip on a resumed run
    # Returns an empty dict if the log file doesn't exist yet (first run)
    if not logPath.exists():
        return {}

    with open(logPath, newline = "") as file:
        return {row["ticID"]: row for row in csv.DictReader(file)}


def appendToFetchLog(rows: list[dict]):
    shouldWriteHeader = not logPath.exists()

    with open(logPath, "a", newline = "") as file:
        csvWriter = csv.DictWriter(file, fieldnames = logFields)

        if shouldWriteHeader:
            csvWriter.writeheader()

        csvWriter.writerows(rows)


def downloadStar(ticID: str) -> dict:
    # Downloads all available QLP light curves for one star from MAST
    # Returns a dict with keys matching logFields describing the outcome
    starDir = lightcurveDir / ticID
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        searchResults = lk.search_lightcurve( # type: ignore
            f"TIC {ticID}", mission = lightcurveMission, author = lightcurveAuthor,

        )

        # Fall back to any TESS source if QLP has no coverage for this star
        usedFallback = False

        if len(searchResults) == 0:
            searchResults = lk.search_lightcurve(f"TIC {ticID}", mission = lightcurveMission) # type: ignore
            usedFallback = len(searchResults) > 0

        if len(searchResults) == 0:
            return {
                "ticID": ticID, "status": "no_data", "fileCount": 0,
                "sectors": "", "timestamp": timestamp,
                "note": "No TESS light curves found on MAST",

            }

        starDir.mkdir(parents = True, exist_ok = True)
        savedFiles = []
        savedSectors = []

        for result in searchResults:
            sector = getattr(result, "sector", "unk")
            exposureTime = getattr(result, "exptime", "unk")
            author = str(getattr(result, "author", "unk")).replace("/", "-")
            fileName = f"tic{ticID}_s{sector}_{author}_{exposureTime}s.fits"
            filePath = starDir / fileName

            # Skip sectors already on disk
            if filePath.exists():
                savedFiles.append(fileName)
                savedSectors.append(str(sector))

                continue

            try:
                # Snapshot existing FITS files before download so we can identify the new file by set difference afterward
                existingFits = set(starDir.rglob("*.fits"))

                result.download(download_dir = str(starDir), cache = False)

                newFits = set(starDir.rglob("*.fits")) - existingFits

                if newFits:
                    newFile = newFits.pop()

                    try:
                        newFile.rename(filePath)

                    except Exception as renameError:
                        logger.debug(f"TIC {ticID} sector {sector}: rename failed: {renameError}")

                if filePath.exists():
                    savedFiles.append(fileName)
                    savedSectors.append(str(sector))

            except Exception as sectorError:
                logger.debug(f"TIC {ticID} sector {sector}: download failed: {sectorError}")

                continue

        if not savedFiles:
            return {
                "ticID": ticID, "status": "failed", "fileCount": 0,
                "sectors": "", "timestamp": timestamp,
                "note": f"Search returned {len(searchResults)} result(s) but all downloads failed",

            }

        fallbackNote = "QLP unavailable - used fallback TESS source" if usedFallback else ""

        return {
            "ticID": ticID, "status": "ok", "fileCount": len(savedFiles),
            "sectors": ";".join(savedSectors), "timestamp": timestamp, "note": fallbackNote,

        }

    except Exception as error:
        return {
            "ticID": ticID, "status": "error", "fileCount": 0,
            "sectors": "", "timestamp": timestamp, "note": str(error),

        }


def main():
    parser = argparse.ArgumentParser(
        description = "Fetch Tey et al. labels from Zenodo, then download light curves from MAST."
    )
    parser.add_argument("--csv-only", action = "store_true",
        help = "Download the label table only; skip MAST downloads.")
    parser.add_argument("--workers", type = int, default = 8,
        help = "Parallel download threads (default 8). Keep at or below 8 to avoid MAST rate limits.")
    parser.add_argument("--limit", type = int, default = None,
        help = "Process only the first N TIC IDs. Useful for testing.")
    parser.add_argument("--resume", action = "store_true",
        help = "Skip TIC IDs already marked 'ok' in the fetch log.")
    parser.add_argument("--retry-failed", action = "store_true",
        help = "Re-attempt TIC IDs previously logged as 'failed' or 'error'.")
    
    args = parser.parse_args()

    rawDir.mkdir(parents = True, exist_ok = True)
    lightcurveDir.mkdir(parents = True, exist_ok = True)

    # Phase 1: label table from Zenodo
    logger.info("=== Phase 1: label table (Zenodo) ===")
    zenodoMetadata = fetchZenodoMetadata()
    recordFiles = {f["key"]: f for f in zenodoMetadata.get("files", [])}

    if not recordFiles:
        logger.error("Zenodo record returned no files. It may be restricted or the API changed.")

        sys.exit(1)

    logger.info(f"Files in record: {list(recordFiles.keys())}")

    if not downloadZenodoCsv(recordFiles):
        logger.error("Failed to obtain label table. Cannot continue.")

        sys.exit(1)

    if args.csv_only:
        logger.info("--csv-only set. Stopping after label table download.")

        return

    # Phase 2: light curves from MAST
    logger.info("=== Phase 2: light curves (MAST via Lightkurve) ===")

    if lk is None:
        logger.error("lightkurve is not installed. Run: pip install lightkurve")

        sys.exit(1)

    rows = loadLabelTable()
    ticIDs = extractUniqueTicIDs(rows)

    if args.resume or args.retry_failed:
        fetchLog = loadFetchLog()
        skipStatuses = {"ok"} if args.retry_failed else {"ok", "failed", "error", "no_data"}
        totalBefore = len(ticIDs)

        ticIDs = [t for t in ticIDs if fetchLog.get(t, {}).get("status") not in skipStatuses]
        logger.info(f"Resume: {totalBefore - len(ticIDs):,} stars skipped, {len(ticIDs):,} remaining.")

    if args.limit:
        ticIDs = ticIDs[:args.limit]
        logger.info(f"--limit {args.limit}: processing {len(ticIDs)} stars.")

    totalStars = len(ticIDs)
    completed = 0
    okCount = 0
    failedCount = 0
    noDataCount = 0

    logger.info(f"Fetching {totalStars:,} stars with {args.workers} worker(s) ...")
    logger.info(f"Output dir : {lightcurveDir.relative_to(repoRoot)}")
    logger.info(f"Fetch log  : {logPath.relative_to(repoRoot)}")

    print()

    logBatch: list[dict] = []

    with ThreadPoolExecutor(max_workers = args.workers) as pool:
        futures = {pool.submit(downloadStar, ticID): ticID for ticID in ticIDs}

        for future in as_completed(futures):
            result = future.result()
            logBatch.append(result)
            completed += 1

            status = result["status"]
            if status == "ok":
                okCount += 1

            elif status == "no_data":
                noDataCount += 1

            else:
                failedCount += 1

            try:
                print(
                    f"\r  {completed:>6,}/{totalStars:,}  ({100 * completed / totalStars:5.1f}%)  "
                    f"ok={okCount:,}  no_data={noDataCount}  failed={failedCount}",
                    end = "", flush = True,
                )
            except ValueError:
                pass

            # Flush to disk every 50 results so progress survives an interrupted run
            if len(logBatch) >= 50:
                appendToFetchLog(logBatch)
                
                logBatch.clear()

    if logBatch:
        appendToFetchLog(logBatch)

    try:
        print("\n")
    except ValueError:
        pass
    logger.info("=== Summary ===")
    logger.info(f"  OK        {okCount:>6,}")
    logger.info(f"  No data   {noDataCount:>6,}")
    logger.info(f"  Failed    {failedCount:>6,}")
    logger.info(f"  Log       {logPath.relative_to(repoRoot)}")

    if failedCount > 0:
        logger.info("  Retry with: python data/fetch.py --resume --retry-failed")

    logger.info("Next: python data/preprocess.py")


if __name__ == "__main__":
    main()