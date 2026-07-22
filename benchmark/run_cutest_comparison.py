#!/usr/bin/env python3
"""Compare two Ipopt libraries or executables, and optionally FATROP."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = 3
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def run_text(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    cwd: pathlib.Path | None = None,
) -> str:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def git_metadata(source_root: pathlib.Path) -> dict[str, Any]:
    try:
        commit = run_text(["git", "rev-parse", "HEAD"], cwd=source_root)
        dirty = bool(
            run_text(["git", "status", "--porcelain"], cwd=source_root)
        )
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


def variant_environment(
    base: dict[str, str], library_directory: pathlib.Path | None
) -> dict[str, str]:
    environment = base.copy()
    for variable in THREAD_VARIABLES:
        environment[variable] = "1"
    if library_directory is not None:
        previous = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = str(library_directory)
        if previous:
            environment["LD_LIBRARY_PATH"] += os.pathsep + previous
    return environment


def resolved_ipopt(
    executable: pathlib.Path, environment: dict[str, str]
) -> str:
    output = run_text(["ldd", str(executable)], environment=environment)
    for line in output.splitlines():
        if "libipopt.so" in line:
            _, separator, resolution = line.partition("=>")
            fields = (resolution if separator else line).split()
            if not fields or fields[0] == "not":
                raise RuntimeError("linked libipopt could not be resolved")
            return str(pathlib.Path(fields[0]).resolve())
    raise RuntimeError("ldd did not report a linked libipopt")


def with_affinity(command: list[str], cpu: int | None) -> list[str]:
    if cpu is not None:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("--cpu requires taskset on this platform")
        command = [taskset, "-c", str(cpu), *command]
    return command


def benchmark_command(
    executable: pathlib.Path,
    outsdif: pathlib.Path,
    problem_library: pathlib.Path,
    cpu: int | None,
) -> list[str]:
    return with_affinity(
        [str(executable), str(outsdif), str(problem_library)], cpu
    )


def fatrop_command(
    executable: pathlib.Path, case_name: str, cpu: int | None
) -> list[str]:
    dtoc1l = re.fullmatch(
        r"DTOC1L_N(?P<stages>\d+)_X(?P<controls>\d+)_Y(?P<states>\d+)",
        case_name,
    )
    dtoc3 = re.fullmatch(r"DTOC3_N(?P<stages>\d+)", case_name)
    dtoc6 = re.fullmatch(r"DTOC6_N(?P<stages>\d+)", case_name)
    if dtoc1l is not None:
        command = [
            str(executable),
            "--problem",
            "dtoc1l",
            "--stages",
            dtoc1l["stages"],
            "--controls",
            dtoc1l["controls"],
            "--states",
            dtoc1l["states"],
        ]
    elif dtoc3 is not None:
        command = [
            str(executable),
            "--problem",
            "dtoc3",
            "--stages",
            dtoc3["stages"],
        ]
    elif dtoc6 is not None:
        command = [
            str(executable),
            "--problem",
            "dtoc6",
            "--stages",
            dtoc6["stages"],
        ]
    else:
        raise RuntimeError(f"no FATROP mapping for CUTEst case: {case_name}")
    return with_affinity(command, cpu)


def parse_benchmark_output(output: str) -> dict[str, Any]:
    lines = output.splitlines()
    try:
        header = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("problem,classification,n,m,")
        )
    except StopIteration as error:
        raise RuntimeError("CUTEst benchmark emitted no CSV header") from error
    reader = csv.DictReader(io.StringIO("\n".join(lines[header:])))
    rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError("CUTEst benchmark emitted an unexpected CSV row count")
    row = rows[0]
    integer_fields = (
        "n",
        "m",
        "jacobian_nonzeros",
        "hessian_nonzeros",
        "status",
        "iterations",
    )
    floating_fields = (
        "ipopt_ms",
        "objective",
        "max_constraint_violation",
    )
    result: dict[str, Any] = dict(row)
    for field in integer_fields:
        result[field] = int(row[field])
    for field in floating_fields:
        result[field] = float(row[field])
    for field, value in tuple(result.items()):
        if field in integer_fields or field in floating_fields:
            continue
        if field in {
            "problem",
            "classification",
        }:
            continue
        if field.endswith("_ms"):
            result[field] = float(value)
        elif value and re.fullmatch(r"[+-]?\d+", value):
            result[field] = int(value)
    return result


def parse_fatrop_output(output: str) -> dict[str, Any]:
    lines = output.splitlines()
    try:
        header = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("problem,stages,controls,states,")
        )
    except StopIteration as error:
        raise RuntimeError("FATROP benchmark emitted no CSV header") from error
    reader = csv.DictReader(io.StringIO("\n".join(lines[header:])))
    rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError("FATROP benchmark emitted an unexpected CSV row count")
    row = rows[0]
    result: dict[str, Any] = dict(row)
    for field in (
        "stages",
        "controls",
        "states",
        "primal_variables",
        "constraints",
        "status",
        "iterations",
    ):
        result[field] = int(row[field])
    for field in (
        "fatrop_ms",
        "objective",
        "max_constraint_violation",
    ):
        result[field] = float(row[field])
    return result


def run_once(
    command: list[str], environment: dict[str, str]
) -> dict[str, Any]:
    return parse_benchmark_output(run_text(command, environment=environment))


def run_fatrop_once(
    command: list[str], environment: dict[str, str]
) -> dict[str, Any]:
    return parse_fatrop_output(run_text(command, environment=environment))


def summarize(
    samples: list[dict[str, Any]], elapsed_field: str
) -> dict[str, Any]:
    elapsed = [float(sample[elapsed_field]) for sample in samples]
    objectives = [float(sample["objective"]) for sample in samples]
    violations = [
        float(sample["max_constraint_violation"]) for sample in samples
    ]
    result = {
        "samples_ms": elapsed,
        "median_ms": statistics.median(elapsed),
        "minimum_ms": min(elapsed),
        "maximum_ms": max(elapsed),
        "statuses": [int(sample["status"]) for sample in samples],
        "iterations": [int(sample["iterations"]) for sample in samples],
        "objectives": objectives,
        "maximum_constraint_violations": violations,
    }
    excluded = {
        "problem",
        "classification",
        "n",
        "m",
        "jacobian_nonzeros",
        "hessian_nonzeros",
        "status",
        "iterations",
        elapsed_field,
        "objective",
        "max_constraint_violation",
    }
    telemetry: dict[str, Any] = {}
    for field, value in samples[0].items():
        if field in excluded or not isinstance(value, (int, float)):
            continue
        values = [sample[field] for sample in samples]
        telemetry[field] = {"samples": values}
        if field.endswith("_ms"):
            telemetry[field]["median"] = statistics.median(values)
    if telemetry:
        result["telemetry"] = telemetry
    return result


def relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(1.0, abs(first), abs(second))


def validate_case(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    objective_tolerance: float,
    constraint_tolerance: float,
) -> None:
    all_samples = baseline + candidate
    metadata_fields = (
        "problem",
        "classification",
        "n",
        "m",
        "jacobian_nonzeros",
        "hessian_nonzeros",
    )
    reference = all_samples[0]
    for sample in all_samples:
        if any(sample[field] != reference[field] for field in metadata_fields):
            raise RuntimeError("baseline and candidate problem metadata differ")
        if sample["status"] not in (0, 1):
            raise RuntimeError(
                f"{sample['problem']} returned unsuccessful status "
                f"{sample['status']}"
            )
        if sample["max_constraint_violation"] > constraint_tolerance:
            raise RuntimeError(
                f"{sample['problem']} constraint violation exceeds tolerance"
            )

    baseline_iterations = {sample["iterations"] for sample in baseline}
    candidate_iterations = {sample["iterations"] for sample in candidate}
    if len(baseline_iterations) != 1 or len(candidate_iterations) != 1:
        raise RuntimeError("iteration count is not deterministic within a variant")
    if baseline_iterations != candidate_iterations:
        raise RuntimeError("baseline and candidate iteration counts differ")

    baseline_objective = statistics.median(
        float(sample["objective"]) for sample in baseline
    )
    for sample in baseline + candidate:
        if relative_difference(
            baseline_objective, float(sample["objective"])
        ) > objective_tolerance:
            raise RuntimeError("objective is not stable across variants")

    if "candidate_requests" in candidate[0]:
        for sample in candidate:
            if (
                sample["candidate_requests"] <= 0
                or sample["candidate_accepted"]
                != sample["candidate_requests"]
                or sample["candidate_fallbacks"] != 0
                or sample["backend_factory_failures"] != 0
            ):
                raise RuntimeError(
                    "candidate-first executable did not use its backend "
                    "successfully on every request"
                )


def validate_fatrop_case(
    baseline: list[dict[str, Any]],
    fatrop: list[dict[str, Any]],
    objective_tolerance: float,
    constraint_tolerance: float,
) -> None:
    reference_n = int(baseline[0]["n"])
    baseline_objective = statistics.median(
        float(sample["objective"]) for sample in baseline
    )
    for sample in fatrop:
        if sample["status"] != 0:
            raise RuntimeError(
                f"FATROP {sample['problem']} returned unsuccessful status "
                f"{sample['status']}"
            )
        if sample["primal_variables"] != reference_n:
            raise RuntimeError("FATROP and CUTEst primal dimensions differ")
        if sample["max_constraint_violation"] > constraint_tolerance:
            raise RuntimeError("FATROP constraint violation exceeds tolerance")
        if relative_difference(
            baseline_objective, float(sample["objective"])
        ) > objective_tolerance:
            raise RuntimeError("FATROP and CUTEst objectives differ")
    if len({sample["iterations"] for sample in fatrop}) != 1:
        raise RuntimeError("FATROP iteration count is not deterministic")


def discover_cases(cases_directory: pathlib.Path) -> list[str]:
    return sorted(
        child.name
        for child in cases_directory.iterdir()
        if child.is_dir()
        and (child / "OUTSDIF.d").is_file()
        and (child / f"lib{child.name}.so").is_file()
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executable",
        type=pathlib.Path,
        help="one executable used with two ABI-compatible Ipopt libraries",
    )
    parser.add_argument("--baseline-executable", type=pathlib.Path)
    parser.add_argument("--candidate-executable", type=pathlib.Path)
    parser.add_argument("--cases-dir", type=pathlib.Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--baseline-library-dir", type=pathlib.Path)
    parser.add_argument("--candidate-library-dir", type=pathlib.Path)
    parser.add_argument("--fatrop-executable", type=pathlib.Path)
    parser.add_argument("--fatrop-source-root", type=pathlib.Path)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--source-root", type=pathlib.Path)
    parser.add_argument(
        "--max-objective-relative-difference", type=float, default=1e-8
    )
    parser.add_argument(
        "--max-constraint-violation", type=float, default=1e-6
    )
    arguments = parser.parse_args()
    if arguments.repetitions < 3 or arguments.repetitions % 2 == 0:
        parser.error("--repetitions must be odd and at least 3")
    if arguments.warmup < 0:
        parser.error("--warmup cannot be negative")
    if arguments.max_objective_relative_difference < 0:
        parser.error("objective tolerance cannot be negative")
    if arguments.max_constraint_violation < 0:
        parser.error("constraint tolerance cannot be negative")
    distinct_executables = (
        arguments.baseline_executable is not None
        or arguments.candidate_executable is not None
    )
    if distinct_executables:
        if (
            arguments.executable is not None
            or arguments.baseline_executable is None
            or arguments.candidate_executable is None
        ):
            parser.error(
                "use either --executable or both --baseline-executable "
                "and --candidate-executable"
            )
    elif arguments.executable is None:
        parser.error(
            "--executable or both variant executable options are required"
        )
    elif arguments.candidate_library_dir is None:
        parser.error(
            "--candidate-library-dir is required with a shared executable"
        )
    if (
        arguments.fatrop_source_root is not None
        and arguments.fatrop_executable is None
    ):
        parser.error("--fatrop-source-root requires --fatrop-executable")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    shared_executable = (
        arguments.executable.resolve()
        if arguments.executable is not None
        else None
    )
    baseline_executable = (
        shared_executable
        if shared_executable is not None
        else arguments.baseline_executable.resolve()
    )
    candidate_executable = (
        shared_executable
        if shared_executable is not None
        else arguments.candidate_executable.resolve()
    )
    cases_directory = arguments.cases_dir.resolve()
    candidate_library_directory = (
        arguments.candidate_library_dir.resolve()
        if arguments.candidate_library_dir is not None
        else None
    )
    baseline_library_directory = (
        arguments.baseline_library_dir.resolve()
        if arguments.baseline_library_dir is not None
        else None
    )
    fatrop_executable = (
        arguments.fatrop_executable.resolve()
        if arguments.fatrop_executable is not None
        else None
    )
    fatrop_source_root = (
        arguments.fatrop_source_root.resolve()
        if arguments.fatrop_source_root is not None
        else None
    )
    source_root = (
        arguments.source_root.resolve()
        if arguments.source_root is not None
        else pathlib.Path(__file__).resolve().parent.parent
    )
    if not baseline_executable.is_file():
        raise RuntimeError(
            f"baseline benchmark executable does not exist: "
            f"{baseline_executable}"
        )
    if not candidate_executable.is_file():
        raise RuntimeError(
            f"candidate benchmark executable does not exist: "
            f"{candidate_executable}"
        )
    if not cases_directory.is_dir():
        raise RuntimeError(f"CUTEst cases directory does not exist: {cases_directory}")
    if (
        candidate_library_directory is not None
        and not candidate_library_directory.is_dir()
    ):
        raise RuntimeError("candidate library directory does not exist")
    if fatrop_executable is not None and not fatrop_executable.is_file():
        raise RuntimeError(f"FATROP executable does not exist: {fatrop_executable}")
    if fatrop_source_root is not None and not fatrop_source_root.is_dir():
        raise RuntimeError("FATROP source root does not exist")
    if (
        baseline_library_directory is not None
        and not baseline_library_directory.is_dir()
    ):
        raise RuntimeError("baseline library directory does not exist")

    cases = arguments.cases or discover_cases(cases_directory)
    if not cases:
        raise RuntimeError("no decoded CUTEst cases were selected")

    base_environment = os.environ.copy()
    baseline_environment = variant_environment(
        base_environment, baseline_library_directory
    )
    candidate_environment = variant_environment(
        base_environment, candidate_library_directory
    )
    fatrop_environment = variant_environment(base_environment, None)
    baseline_ipopt = resolved_ipopt(
        baseline_executable, baseline_environment
    )
    candidate_ipopt = resolved_ipopt(
        candidate_executable, candidate_environment
    )
    if (
        baseline_executable == candidate_executable
        and baseline_ipopt == candidate_ipopt
    ):
        raise RuntimeError("baseline and candidate resolve to the same libipopt")

    results: list[dict[str, Any]] = []
    for case_name in cases:
        case_directory = cases_directory / case_name
        outsdif = case_directory / "OUTSDIF.d"
        problem_library = case_directory / f"lib{case_name}.so"
        if not outsdif.is_file() or not problem_library.is_file():
            raise RuntimeError(f"decoded CUTEst case is incomplete: {case_name}")
        baseline_command = benchmark_command(
            baseline_executable, outsdif, problem_library, arguments.cpu
        )
        candidate_command = benchmark_command(
            candidate_executable, outsdif, problem_library, arguments.cpu
        )
        fatrop_case_command = (
            fatrop_command(fatrop_executable, case_name, arguments.cpu)
            if fatrop_executable is not None
            else None
        )

        def execute(variant: str) -> dict[str, Any]:
            if variant == "baseline":
                return run_once(baseline_command, baseline_environment)
            if variant == "candidate":
                return run_once(candidate_command, candidate_environment)
            if variant == "fatrop" and fatrop_case_command is not None:
                return run_fatrop_once(
                    fatrop_case_command, fatrop_environment
                )
            raise RuntimeError(f"unknown benchmark variant: {variant}")

        for _ in range(arguments.warmup):
            execute("baseline")
            execute("candidate")
            if fatrop_case_command is not None:
                execute("fatrop")

        samples: dict[str, list[dict[str, Any]]] = {
            "baseline": [],
            "candidate": [],
        }
        variants = ["baseline", "candidate"]
        if fatrop_case_command is not None:
            samples["fatrop"] = []
            variants.append("fatrop")
        for repetition in range(arguments.repetitions):
            offset = repetition % len(variants)
            order = variants[offset:] + variants[:offset]
            for variant in order:
                samples[variant].append(execute(variant))

        validate_case(
            samples["baseline"],
            samples["candidate"],
            arguments.max_objective_relative_difference,
            arguments.max_constraint_violation,
        )
        baseline_summary = summarize(samples["baseline"], "ipopt_ms")
        candidate_summary = summarize(samples["candidate"], "ipopt_ms")
        baseline_median = float(baseline_summary["median_ms"])
        candidate_median = float(candidate_summary["median_ms"])
        metadata = samples["baseline"][0]
        result = {
            "case": case_name,
            "problem": metadata["problem"],
            "classification": metadata["classification"],
            "n": metadata["n"],
            "m": metadata["m"],
            "jacobian_nonzeros": metadata["jacobian_nonzeros"],
            "hessian_nonzeros": metadata["hessian_nonzeros"],
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "baseline_over_candidate": baseline_median / candidate_median,
            "candidate_change_percent": (
                candidate_median / baseline_median - 1.0
            ) * 100.0,
        }
        if fatrop_case_command is not None:
            validate_fatrop_case(
                samples["baseline"],
                samples["fatrop"],
                arguments.max_objective_relative_difference,
                arguments.max_constraint_violation,
            )
            fatrop_summary = summarize(samples["fatrop"], "fatrop_ms")
            fatrop_metadata = samples["fatrop"][0]
            fatrop_summary.update(
                {
                    field: fatrop_metadata[field]
                    for field in (
                        "stages",
                        "controls",
                        "states",
                        "primal_variables",
                        "constraints",
                    )
                }
            )
            fatrop_median = float(fatrop_summary["median_ms"])
            result.update(
                {
                    "fatrop": fatrop_summary,
                    "baseline_over_fatrop": baseline_median / fatrop_median,
                    "candidate_over_fatrop": candidate_median / fatrop_median,
                }
            )
        results.append(result)

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": git_metadata(source_root),
        "host": {
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "cpu": arguments.cpu,
        },
        "protocol": {
            "repetitions": arguments.repetitions,
            "warmup": arguments.warmup,
            "interleaved": True,
            "interleaved_variants": [
                "baseline",
                "candidate",
                *(["fatrop"] if fatrop_executable is not None else []),
            ],
            "max_objective_relative_difference": (
                arguments.max_objective_relative_difference
            ),
            "max_constraint_violation": arguments.max_constraint_violation,
            "thread_environment": {
                variable: "1" for variable in THREAD_VARIABLES
            },
        },
        "baseline": {
            "executable": str(baseline_executable),
            "library_directory": (
                str(baseline_library_directory)
                if baseline_library_directory is not None
                else None
            ),
            "resolved_ipopt": baseline_ipopt,
        },
        "candidate": {
            "executable": str(candidate_executable),
            "library_directory": (
                str(candidate_library_directory)
                if candidate_library_directory is not None
                else None
            ),
            "resolved_ipopt": candidate_ipopt,
        },
        "cases": results,
    }
    if fatrop_executable is not None:
        document["fatrop"] = {
            "executable": str(fatrop_executable),
            "source": (
                git_metadata(fatrop_source_root)
                if fatrop_source_root is not None
                else None
            ),
        }
    serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        csv.Error,
        ValueError,
    ) as error:
        print(f"run_cutest_comparison.py: {error}", file=sys.stderr)
        raise SystemExit(2)
