# written with Claude Code

import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

try:
    import psutil as _psutil
    _proc = _psutil.Process()
    _haveRam = True
except ImportError:
    _haveRam = False
    print("WARNING: psutil not installed — RAM tracking disabled. pip install psutil")

scriptDir = Path(__file__).resolve().parent
repoRoot = scriptDir.parent
sys.path.insert(0, str(repoRoot / "src" / "model"))

from network import TransitClassifier
from dataset import TransitDataset, makeSplits
from config import numClasses, defaultBatchSize

dataPath    = repoRoot / "src" / "data" / "processed" / "dataset.h5"
scalarsPath = repoRoot / "src" / "data" / "processed" / "scalar_stats.json"
soloCheckpoint = scriptDir / "Solo" / "best.pt"
choirDir       = scriptDir / "Choir" / "models"

trainBatchSize = defaultBatchSize   # 64 — matches actual training config
infBatchSize   = defaultBatchSize
trainWorkers   = 4
infWorkers     = 4
warmupSteps    = 100
timedSteps     = 300
totalSteps     = 20000              # canonical full-run length for extrapolation


def sep(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def syncIfCuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def resetVRAM():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peakVRAM_GB():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


class RamTracker:
    """Polls process RSS every `interval` seconds in a background thread.
    Use as a context manager: `with RamTracker() as ram: ...  peak = ram.peak_GB()`
    """

    def __init__(self, interval = 0.05):
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target = self._run, daemon = True)

    def __enter__(self):
        if _haveRam:
            self._peak = _proc.memory_info().rss
            self._stop.clear()
            self._thread = threading.Thread(target = self._run, daemon = True)
            self._thread.start()
        return self

    def __exit__(self, *_):
        if _haveRam:
            self._stop.set()
            self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            rss = _proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            self._stop.wait(self._interval)

    def peak_GB(self) -> float:
        return self._peak / 1e9 if _haveRam else float("nan")

    def current_GB(self) -> float:
        return _proc.memory_info().rss / 1e9 if _haveRam else float("nan")


def focalLoss(logits, targets, classWeights, gamma = 2.0, eBoost = 4.0):
    perSample = F.cross_entropy(logits, targets, weight = classWeights, reduction = "none")
    probs = torch.softmax(logits, dim = 1)
    trueProb = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focalW = (1 - trueProb) ** gamma
    recallW = torch.where(targets == 0, torch.full_like(focalW, eBoost), torch.ones_like(focalW))
    return (focalW * perSample * recallW).mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB total")

    if _haveRam:
        vm = _psutil.virtual_memory()
        print(f"RAM    : {vm.total / 1e9:.1f} GB total  ({vm.available / 1e9:.1f} GB available)")

    baselineRam = RamTracker().current_GB()
    print(f"RAM    : {baselineRam:.2f} GB process baseline (at import time)")

    print(f"\nLoading dataset...")
    splits = makeSplits(dataPath)
    trainIdx, _, testIdx = splits

    trainDataset = TransitDataset(dataPath, scalarsPath, trainIdx, augment = True)
    testDataset  = TransitDataset(dataPath, scalarsPath, testIdx)

    print(f"Train  : {len(trainDataset):,} TCEs")
    print(f"Test   : {len(testDataset):,} TCEs")
    print(f"RAM after dataset load : {RamTracker().current_GB():.2f} GB  (delta {RamTracker().current_GB() - baselineRam:+.2f} GB)")

    numParams = sum(p.numel() for p in TransitClassifier().parameters())
    print(f"Params : {numParams:,}  ({numParams * 4 / 1e6:.2f} MB fp32)")

    # ----------------------------------------------------------------
    # 1 · Training throughput + VRAM
    # ----------------------------------------------------------------

    sep("Training  (batch = {}, {} warmup + {} timed steps)".format(
        trainBatchSize, warmupSteps, timedSteps))

    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size = trainBatchSize, shuffle = True,
        num_workers = trainWorkers, pin_memory = (device.type == "cuda"),
        persistent_workers = (trainWorkers > 0),
    )

    model = TransitClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-3)

    counts = trainDataset.labelCounts
    total  = sum(counts)
    classWeights = torch.tensor(
        [(total / max(counts[i], 1)) ** 0.75 for i in range(numClasses)],
        dtype = torch.float32
    ).to(device)

    model.train()

    loaderIter = iter(trainLoader)

    def nextBatch():
        nonlocal loaderIter
        try:
            return next(loaderIter)
        except StopIteration:
            loaderIter = iter(trainLoader)
            return next(loaderIter)

    # Warmup
    print(f"Warming up...")
    resetVRAM()

    for _ in range(warmupSteps):
        batch = {k: v.to(device) for k, v in nextBatch().items()}
        optimizer.zero_grad()
        loss = focalLoss(model(batch), batch["label"], classWeights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    syncIfCuda(device)
    vramTrainPeak = peakVRAM_GB()
    resetVRAM()

    # Timed window
    print(f"Timing {timedSteps} steps...")
    syncIfCuda(device)

    with RamTracker() as ramTrain:
        t0 = time.perf_counter()

        for _ in range(timedSteps):
            batch = {k: v.to(device) for k, v in nextBatch().items()}
            optimizer.zero_grad()
            loss = focalLoss(model(batch), batch["label"], classWeights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        syncIfCuda(device)
        trainElapsed = time.perf_counter() - t0

    secPerStep      = trainElapsed / timedSteps
    stepsPerSec     = timedSteps / trainElapsed
    soloEst_s       = totalSteps * secPerStep
    choirEst_s      = soloEst_s * 10

    print(f"\nThroughput          : {stepsPerSec:.1f} steps/sec  ({secPerStep * 1000:.1f} ms/step)")
    print(f"VRAM peak (train)   : {vramTrainPeak:.3f} GB  (batch={trainBatchSize})")
    print(f"RAM  peak (train)   : {ramTrain.peak_GB():.2f} GB")
    print(f"\nExtrapolated to {totalSteps:,} steps:")
    print(f"  Solo  (1 model)   : {soloEst_s:.0f} s  = {soloEst_s / 60:.1f} min")
    print(f"  Choir (10 models) : {choirEst_s:.0f} s  = {choirEst_s / 3600:.2f} hr")

    del model, optimizer, trainLoader, loaderIter
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # 2 · Solo inference
    # ----------------------------------------------------------------

    sep("Solo inference")

    testLoader = torch.utils.data.DataLoader(
        testDataset, batch_size = infBatchSize, shuffle = False,
        num_workers = infWorkers, pin_memory = (device.type == "cuda"),
        persistent_workers = (infWorkers > 0),
    )

    soloModel = TransitClassifier().to(device)
    soloModel.load_state_dict(torch.load(soloCheckpoint, map_location = device, weights_only = True))
    soloModel.eval()

    # GPU warm-up pass
    with torch.no_grad():
        for batch in testLoader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = soloModel(batch)
            break

    resetVRAM()
    syncIfCuda(device)
    nTest = 0

    with RamTracker() as ramSoloGpu:
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in testLoader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = soloModel(batch)
                nTest += batch["label"].shape[0]

        syncIfCuda(device)
        soloGpuElapsed = time.perf_counter() - t0

    vramInfPeak = peakVRAM_GB()

    print(f"GPU (batch={infBatchSize}):")
    print(f"  {nTest} TCEs in {soloGpuElapsed * 1000:.1f} ms")
    print(f"  Throughput   : {nTest / soloGpuElapsed:.0f} TCEs/sec")
    print(f"  Per TCE      : {soloGpuElapsed / nTest * 1000:.3f} ms")
    print(f"  VRAM peak    : {vramInfPeak:.3f} GB")
    print(f"  RAM  peak    : {ramSoloGpu.peak_GB():.2f} GB")

    # CPU inference
    soloModelCpu = TransitClassifier()
    soloModelCpu.load_state_dict(torch.load(soloCheckpoint, map_location = "cpu", weights_only = True))
    soloModelCpu.eval()

    cpuLoader = torch.utils.data.DataLoader(
        testDataset, batch_size = infBatchSize, shuffle = False, num_workers = 0,
    )

    nTestCpu = 0

    with RamTracker() as ramSoloCpu:
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in cpuLoader:
                _ = soloModelCpu(batch)
                nTestCpu += batch["label"].shape[0]

        soloCpuElapsed = time.perf_counter() - t0

    print(f"\nCPU (batch={infBatchSize}, no workers):")
    print(f"  {nTestCpu} TCEs in {soloCpuElapsed * 1000:.1f} ms")
    print(f"  Throughput   : {nTestCpu / soloCpuElapsed:.0f} TCEs/sec")
    print(f"  Per TCE      : {soloCpuElapsed / nTestCpu * 1000:.3f} ms")
    print(f"  RAM  peak    : {ramSoloCpu.peak_GB():.2f} GB")

    del soloModel, soloModelCpu, cpuLoader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # 3 · Choir inference (10 models, sequential — mirrors choir.py)
    # ----------------------------------------------------------------

    sep("Choir inference (10 models, sequential)")

    choirCheckpoints = sorted(choirDir.glob("model_*.pt"))
    print(f"Models : {len(choirCheckpoints)}")

    resetVRAM()
    syncIfCuda(device)

    with RamTracker() as ramChoir:
        t0 = time.perf_counter()

        for ckpt in choirCheckpoints:
            m = TransitClassifier().to(device)
            m.load_state_dict(torch.load(ckpt, map_location = device, weights_only = True))
            m.eval()

            with torch.no_grad():
                for batch in testLoader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    _ = m(batch)

            del m

        syncIfCuda(device)
        choirElapsed = time.perf_counter() - t0

    vramChoirPeak = peakVRAM_GB()
    perModelSec = choirElapsed / len(choirCheckpoints)

    print(f"\nGPU (batch={infBatchSize}, {len(choirCheckpoints)} models x {nTest} TCEs):")
    print(f"  Total time        : {choirElapsed * 1000:.1f} ms")
    print(f"  Per model pass    : {perModelSec * 1000:.1f} ms  ({nTest / perModelSec:.0f} TCEs/sec)")
    print(f"  Per TCE (1 model) : {perModelSec / nTest * 1000:.3f} ms")
    print(f"  Per TCE (choir)   : {choirElapsed / nTest * 1000:.3f} ms  (all 10 models)")
    print(f"  VRAM peak         : {vramChoirPeak:.3f} GB")
    print(f"  RAM  peak         : {ramChoir.peak_GB():.2f} GB")

    # ----------------------------------------------------------------
    # 4 · Checkpoint sizes
    # ----------------------------------------------------------------

    sep("Checkpoint sizes")

    if soloCheckpoint.exists():
        print(f"Solo best.pt       : {soloCheckpoint.stat().st_size / 1e6:.2f} MB")

    if choirCheckpoints:
        perModelMB = choirCheckpoints[0].stat().st_size / 1e6
        totalChoirMB = sum(c.stat().st_size for c in choirCheckpoints) / 1e6
        print(f"Choir per model    : {perModelMB:.2f} MB")
        print(f"Choir total (10)   : {totalChoirMB:.2f} MB")

    print()


if __name__ == "__main__":
    main()
