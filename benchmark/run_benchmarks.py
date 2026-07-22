#!/usr/bin/env python3
"""Run opt-in Ipopt benchmarks and compare them with a JSON baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = 1
BENCHMARKS = {
    "hessian": (
        "tnlp_adapter_eval_h_benchmark",
        "tnlp_adapter_eval_h_fixed_variable",
    ),
    "gradient": (
        "tnlp_adapter_eval_grad_f_benchmark",
        "tnlp_adapter_eval_grad_f_fixed_variable",
    ),
}


def run_text(arguments: list[str], *, cwd: pathlib.Path | None = None) -> str:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def git_metadata(source_root: pathlib.Path) -> dict[str, Any]:
    try:
        commit = run_text(["git", "rev-parse", "HEAD"], cwd=source_root)
        dirty = bool(run_text(["git", "status", "--porcelain"], cwd=source_root))
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"root": str(source_root), "commit": commit, "dirty": dirty}


def cpu_model() -> str | None:
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor() or None
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor() or None


def benchmark_command(executable: pathlib.Path, args: argparse.Namespace) -> list[str]:
    command = [
        str(executable),
        "--iterations",
        str(args.iterations),
        "--repetitions",
        str(args.repetitions),
        "--warmup",
        str(args.warmup),
    ]
    if args.cpu is not None:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("--cpu requires taskset on this platform")
        command = [taskset, "-c", str(args.cpu), *command]
    return command


def compare_with_baseline(
    current: dict[str, Any], baseline_path: pathlib.Path, max_regression_percent: float
) -> dict[str, Any]:
    baseline_document = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_name = {
        benchmark["name"]: benchmark for benchmark in baseline_document["benchmarks"]
    }
    baseline = baseline_by_name.get(current["name"])
    if baseline is None:
        raise RuntimeError(f"baseline has no benchmark named {current['name']!r}")
    if baseline["metric"] != current["metric"]:
        raise RuntimeError("baseline metric does not match current metric")

    baseline_value = float(baseline["median"])
    current_value = float(current["median"])
    ratio = current_value / baseline_value
    regression_percent = (ratio - 1.0) * 100.0
    passed = regression_percent <= max_regression_percent
    return {
        "baseline": str(baseline_path),
        "baseline_median": baseline_value,
        "current_median": current_value,
        "ratio": ratio,
        "regression_percent": regression_percent,
        "max_regression_percent": max_regression_percent,
        "passed": passed,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--max-regression-percent", type=float, default=5.0)
    parser.add_argument(
        "--benchmark",
        choices=(*BENCHMARKS, "all"),
        default="hessian",
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--iterations", type=int, default=20_000_000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup <= 0:
        parser.error("iterations and warmup must be positive")
    if args.repetitions < 3 or args.repetitions % 2 == 0:
        parser.error("repetitions must be odd and at least 3")
    if args.max_regression_percent < 0:
        parser.error("max regression percent cannot be negative")
    return args


def main() -> int:
    args = parse_arguments()
    build_dir = args.build_dir.resolve()
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else pathlib.Path(__file__).resolve().parent.parent
    )
    benchmark_dir = build_dir / "benchmark"
    selected = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    executables = [benchmark_dir / BENCHMARKS[name][0] for name in selected]

    if not args.no_build:
        subprocess.run(
            ["make", "-C", str(benchmark_dir), *(path.name for path in executables)],
            check=True,
        )
    for executable in executables:
        if not executable.exists():
            raise RuntimeError(
                f"{executable} does not exist; configure Ipopt with --enable-benchmarks"
            )

    environment = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[variable] = "1"

    expected_callbacks = args.repetitions * (args.iterations + args.warmup)
    commands: list[list[str]] = []
    benchmarks: list[dict[str, Any]] = []
    for name, executable in zip(selected, executables, strict=True):
        command = benchmark_command(executable, args)
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        benchmark = json.loads(completed.stdout)
        if benchmark.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("unsupported benchmark result schema")
        if benchmark.get("name") != BENCHMARKS[name][1]:
            raise RuntimeError("benchmark returned an unexpected name")
        if benchmark.get("callback_evaluations") != expected_callbacks:
            raise RuntimeError("benchmark callback-count correctness check failed")
        commands.append(command)
        benchmarks.append(benchmark)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": git_metadata(source_root),
        "host": {
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "cpu": args.cpu,
        },
        "environment": {
            variable: environment[variable]
            for variable in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "benchmarks": benchmarks,
    }
    result["command" if len(commands) == 1 else "commands"] = (
        commands[0] if len(commands) == 1 else commands
    )

    passed = True
    if args.baseline is not None:
        comparisons = [
            compare_with_baseline(
                benchmark, args.baseline.resolve(), args.max_regression_percent
            )
            for benchmark in benchmarks
        ]
        result["comparisons"] = comparisons
        passed = all(bool(comparison["passed"]) for comparison in comparisons)

    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0 if passed else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"run_benchmarks.py: {error}", file=sys.stderr)
        raise SystemExit(2)
