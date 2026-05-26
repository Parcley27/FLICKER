#written with claude code

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

scriptDir = Path(__file__).resolve().parent
repoRoot = scriptDir.parent

soloCheckpoint = scriptDir / "Solo" / "best.pt"
choirDir = scriptDir / "Choir" / "models"
benchmarksRoot = scriptDir / "Benchmarks"
evaluatePy = repoRoot / "model" / "evaluation" / "evaluate.py"

expectedChoirSize = 10


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "FLICKER benchmark: run Solo + Choir Baritone + Choir Soprano on the test set and save all outputs to FLICKER/Benchmarks/")

    parser.add_argument("--data", type = Path, default = None,
        help = "Path to dataset.h5 (default: data/processed/dataset.h5)")
    parser.add_argument("--scalars", type = Path, default = None,
        help = "Path to scalar_stats.json (default: data/processed/scalar_stats.json)")
    parser.add_argument("--batch-size", type = int, default = 64,
        help = "Inference batch size (default: 64)")
    parser.add_argument("--workers", type = int, default = 4,
        help = "DataLoader worker count (default: 4)")
    parser.add_argument("--output-dir", type = Path, default = None,
        help = "Override output directory. Defaults to FLICKER/Benchmarks/<timestamp>/")

    return parser.parse_args()


def preflight() -> list[str]:
    errors: list[str] = []

    if not soloCheckpoint.exists():
        errors.append(
            f"Solo checkpoint missing: {soloCheckpoint}\n"
            "  Fix: cp model/checkpoints/best_<timestamp>.pt FLICKER/Solo/best.pt"
        )

    if not choirDir.exists():
        errors.append(
            f"Choir models directory not found: {choirDir}\n"
            "  Fix: mkdir FLICKER/Choir/models  then copy model_*.pt into it"
        )

    else:
        found = len(list(choirDir.glob("model_*.pt")))

        if found < expectedChoirSize:
            errors.append(
                f"Choir models incomplete: found {found}/{expectedChoirSize} in {choirDir}\n"
                "  Fix: python model/chorus.py --model-count 10  then copy checkpoints"
            )

    if not evaluatePy.exists():
        errors.append(f"Evaluator not found: {evaluatePy}")

    return errors


def main() -> int:
    args = parseArgs()

    print("FLICKER Benchmark")
    print(f"  Solo checkpoint : {soloCheckpoint}")
    print(f"  Choir models    : {choirDir}  ({len(list(choirDir.glob('model_*.pt')))} models)")
    print()

    errors = preflight()

    if errors:
        print("Pre-flight checks failed:")

        for error in errors:
            print(f"\n  ERROR: {error}")

        return 1

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = args.output_dir or (benchmarksRoot / timestamp)
    outputDir.mkdir(parents = True, exist_ok = True)

    print(f"Output directory: {outputDir}")
    print()

    cmd = [
        sys.executable,
        str(evaluatePy),
        "--solo-checkpoint", str(soloCheckpoint),
        "--choir-dir", str(choirDir),
        "--output-dir", str(outputDir),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
    ]

    if args.data is not None:
        cmd += ["--data", str(args.data)]

    if args.scalars is not None:
        cmd += ["--scalars", str(args.scalars)]

    process = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, encoding = "utf-8")

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end = "")

    process.wait()

    if process.returncode != 0:
        print(f"\nERROR: evaluator exited with code {process.returncode}")

        return process.returncode

    print()
    print(f"Benchmark complete. All outputs saved to:")
    print(f"  {outputDir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
