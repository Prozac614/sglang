"""
Test runner for multimodal_gen with auto load-balanced partitioning at test case level.

Features:
- Automatically loads test case estimated times from perf_baselines.json
- Uses LPT (Longest Processing Time) algorithm for balanced partitioning
- Supports computing optimal partition count to meet time constraints
- Filters test cases using pytest -k for fine-grained control

Usage:
    # Run with specific partition
    python3 run_suite.py --suite 2-gpu --partition-id 0 --total-partitions 5

    # Compute optimal partition count (for CI dynamic matrix)
    python3 run_suite.py --suite 2-gpu --compute-partitions --max-time 1200

    # Run all cases without partitioning
    python3 run_suite.py --suite 2-gpu
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.test.ci.ci_utils import (
    TestCase,
    auto_partition_by_time,
    compute_optimal_partitions,
)

logger = init_logger(__name__)

# Startup overhead per test case (seconds): server startup, model loading, warmup
STARTUP_OVERHEAD_SEC = 300

# Suite configurations: maps suite name to (test_file, case_import_path)
SUITE_CONFIG = {
    "1-gpu": {
        "test_file": "test_server.py",
        "cases_module": "sglang.multimodal_gen.test.server.testcase_configs",
        "cases_attr": ["ONE_GPU_CASES"],
    },
    "2-gpu": {
        "test_file": "test_server_2_gpu.py",
        "cases_module": "sglang.multimodal_gen.test.server.testcase_configs",
        "cases_attr": ["TWO_GPU_CASES"],
    },
}


def load_baseline_times() -> dict:
    """Load expected_e2e_ms from perf_baselines.json."""
    baseline_path = Path(__file__).parent / "server" / "perf_baselines.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        case_id: scenario.get("expected_e2e_ms", 300000)
        for case_id, scenario in data.get("scenarios", {}).items()
    }


def load_test_cases(suite: str) -> List[TestCase]:
    """Load test cases for a suite and attach estimated times from baseline."""
    import importlib

    config = SUITE_CONFIG[suite]
    baseline_times = load_baseline_times()

    # Dynamically import the cases
    module = importlib.import_module(config["cases_module"])
    all_cases = []
    for attr in config["cases_attr"]:
        cases = getattr(module, attr, [])
        all_cases.extend(cases)

    # Convert to TestCase objects with estimated times
    test_cases = []
    for case in all_cases:
        case_id = case.id
        if case_id in baseline_times:
            # Convert ms to seconds + startup overhead
            est_time = baseline_times[case_id] / 1000 + STARTUP_OVERHEAD_SEC
        else:
            # Default: 5 minutes inference + startup overhead
            est_time = 300 + STARTUP_OVERHEAD_SEC
            logger.warning(f"Case {case_id} not found in baseline, using default time")

        test_cases.append(
            TestCase(
                id=case_id,
                estimated_time=est_time,
                test_file=config["test_file"],
            )
        )

    return test_cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multimodal_gen test suite with auto load-balanced partitioning"
    )
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=list(SUITE_CONFIG.keys()),
        help="The test suite to run (e.g., 1-gpu, 2-gpu)",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="Index of the current partition (for parallel execution)",
    )
    parser.add_argument(
        "--total-partitions",
        type=int,
        default=1,
        help="Total number of partitions",
    )
    parser.add_argument(
        "--compute-partitions",
        action="store_true",
        help="Compute and print optimal partition count (for CI dynamic matrix)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=1200,
        help="Maximum time per partition in seconds (default: 1200 = 20min)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="server",
        help="Base directory for tests relative to this script's parent",
    )
    return parser.parse_args()


def run_pytest_with_filter(test_file: Path, case_ids: List[str]) -> int:
    """Run pytest with -k filter for specific test cases."""
    if not case_ids:
        print("No test cases to run.")
        return 0

    # Build -k filter expression: "case1 or case2 or case3"
    filter_expr = " or ".join(case_ids)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        "-v",
        "--log-cli-level=INFO",
        str(test_file),
        "-k",
        filter_expr,
    ]

    max_retries = 4
    returncode = 1

    for i in range(max_retries + 1):
        run_cmd = list(cmd)
        if i > 0:
            run_cmd.insert(-2, "--last-failed")  # Insert before test file

        if i > 0:
            logger.info(
                f"Performance assertion failed. Retrying ({i}/{max_retries}) with --last-failed..."
            )

        logger.info(f"Running command: {' '.join(run_cmd)}")

        process = subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                output_lines.append(line)

        returncode = process.poll()

        if returncode == 0:
            return 0

        # Check if the failure is due to a performance assertion
        full_output = "".join(output_lines)
        is_perf_assertion = (
            "multimodal_gen/test/server/test_server_utils.py" in full_output
            and "AssertionError" in full_output
        )
        is_flaky_ci_assertion = "SafetensorError" in full_output

        if not (is_perf_assertion or is_flaky_ci_assertion):
            return returncode

    logger.info("Max retry exceeded")
    return returncode


def main():
    args = parse_args()

    # Load all test cases for the suite
    all_cases = load_test_cases(args.suite)

    # If computing partitions, just output the count and exit
    if args.compute_partitions:
        optimal_count = compute_optimal_partitions(all_cases, args.max_time)
        # Output as JSON array for GitHub Actions matrix
        partition_list = list(range(optimal_count))
        print(json.dumps(partition_list))
        return 0

    # Apply LPT partitioning at test case level
    if args.total_partitions > 1:
        my_cases = auto_partition_by_time(
            all_cases, args.partition_id, args.total_partitions
        )
    else:
        my_cases = all_cases

    # Print partition info
    total_est_time = sum(c.estimated_time for c in my_cases)
    print(
        f"Suite: {args.suite} | Partition: {args.partition_id}/{args.total_partitions}"
    )
    print(f"Estimated time: {total_est_time:.0f}s ({total_est_time/60:.1f}min)")
    print(f"Test cases ({len(my_cases)}):")
    for c in my_cases:
        print(f"  - {c.id} ({c.estimated_time:.0f}s / {c.estimated_time/60:.1f}min)")

    if not my_cases:
        print("No test cases assigned to this partition. Exiting success.")
        return 0

    # Resolve test file path
    test_root = Path(__file__).resolve().parent
    target_dir = test_root / args.base_dir

    # Use the test file from the first case (all cases in a suite share the same file)
    test_file = target_dir / my_cases[0].test_file

    if not test_file.exists():
        print(f"Error: Test file {test_file} does not exist.")
        return 1

    # Run pytest with case filter
    case_ids = [c.id for c in my_cases]
    exit_code = run_pytest_with_filter(test_file, case_ids)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
