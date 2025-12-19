import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, TypeVar, Union

from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import CIRegistry


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


@dataclass
class TestCase:
    """A single test case with pytest node ID and estimated time."""

    id: str  # Case ID, e.g., "wan2_1_t2v_14b_2gpu"
    estimated_time: float = 300  # Estimated time in seconds, default 5 minutes
    test_file: str = ""  # Associated test file (optional)


# Generic type for partitionable items
T = TypeVar("T", TestFile, TestCase)


def _lpt_partition(items: List[T], size: int) -> List[List[T]]:
    """
    Partition items into `size` sublists with approximately equal sums of estimated times
    using a greedy algorithm (LPT heuristic).

    Args:
        items: List of objects with estimated_time attribute
        size: Number of partitions

    Returns:
        List of partitions, each containing items assigned to that partition
    """
    if not items or size <= 0:
        return [[] for _ in range(max(1, size))]

    # Sort items by estimated_time in descending order (LPT heuristic)
    sorted_items = sorted(items, key=lambda x: x.estimated_time, reverse=True)

    partitions: List[List[T]] = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedily assign each item to the partition with the smallest current total time
    for item in sorted_items:
        min_sum_idx = min(range(size), key=partition_sums.__getitem__)
        partitions[min_sum_idx].append(item)
        partition_sums[min_sum_idx] += item.estimated_time

    return partitions


def auto_partition_by_time(items: List[T], rank: int, size: int) -> List[T]:
    """
    Partition items into `size` sublists with approximately equal sums of estimated times
    using a greedy algorithm (LPT heuristic), and return the partition for the specified rank.

    Args:
        items: List of TestFile or TestCase objects with estimated_time attribute
        rank: Index of the partition to return (0 to size-1)
        size: Number of partitions

    Returns:
        List of items assigned to the specified rank's partition
    """
    partitions = _lpt_partition(items, size)
    if rank < size:
        return partitions[rank]
    return []


def compute_optimal_partitions(
    items: List[T], max_time_sec: int = 1200, min_partitions: int = 1
) -> int:
    """
    Compute the minimum number of partitions needed so that the maximum partition
    time does not exceed max_time_sec.

    Args:
        items: List of TestFile or TestCase objects with estimated_time attribute
        max_time_sec: Maximum allowed time per partition in seconds (default 20 minutes)
        min_partitions: Minimum number of partitions to return (default 1)

    Returns:
        Optimal number of partitions
    """
    if not items:
        return min_partitions

    total_time = sum(item.estimated_time for item in items)
    max_item_time = max(item.estimated_time for item in items)

    # Lower bound: at least ceil(total / max_time) partitions
    lower_bound = max(min_partitions, math.ceil(total_time / max_time_sec))

    # If the largest item exceeds max_time, we can't satisfy the constraint
    # but we still try to minimize max partition time
    if max_item_time > max_time_sec:
        # Start from lower_bound and find where adding more partitions doesn't help
        pass

    # Try increasing partition count until constraint is satisfied or we hit item count
    for n in range(lower_bound, len(items) + 1):
        partitions = _lpt_partition(items, n)
        max_partition_time = max(
            sum(item.estimated_time for item in p) for p in partitions if p
        )
        if max_partition_time <= max_time_sec:
            return n

    # Worst case: one partition per item
    return len(items)


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


def run_unittest_files(
    files: Union[List[TestFile], List[CIRegistry]],
    timeout_per_file: float,
    continue_on_error: bool = False,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        timeout_per_file: Timeout in seconds for each test file
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []

    for i, file in enumerate(files):
        if isinstance(file, CIRegistry):
            filename, estimated_time = file.filename, file.est_time
        else:
            # FIXME: remove this branch after migrating all tests to use CIRegistry
            filename, estimated_time = file.name, file.estimated_time

        process = None

        def run_one_file(filename):
            nonlocal process

            filename = os.path.join(os.getcwd(), filename)
            print(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {filename}\n.\n.\n",
                flush=True,
            )
            tic = time.perf_counter()

            process = subprocess.Popen(
                ["python3", filename], stdout=None, stderr=None, env=os.environ
            )
            process.wait()
            elapsed = time.perf_counter() - tic

            print(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
                flush=True,
            )
            return process.returncode

        try:
            ret_code = run_with_timeout(
                run_one_file, args=(filename,), timeout=timeout_per_file
            )
            if ret_code != 0:
                print(
                    f"\n✗ FAILED: {filename} returned exit code {ret_code}\n",
                    flush=True,
                )
                success = False
                failed_tests.append((filename, f"exit code {ret_code}"))
                if not continue_on_error:
                    # Stop at first failure for PR tests
                    break
                # Otherwise continue to next test for nightly tests
            else:
                passed_tests.append(filename)
        except TimeoutError:
            kill_process_tree(process.pid)
            time.sleep(5)
            print(
                f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n",
                flush=True,
            )
            success = False
            failed_tests.append((filename, f"timeout after {timeout_per_file}s"))
            if not continue_on_error:
                # Stop at first timeout for PR tests
                break
            # Otherwise continue to next test for nightly tests

    if success:
        print(f"Success. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Test Summary: {len(passed_tests)}/{len(files)} passed", flush=True)
    print(f"{'='*60}", flush=True)
    if passed_tests:
        print("✓ PASSED:", flush=True)
        for test in passed_tests:
            print(f"  {test}", flush=True)
    if failed_tests:
        print("\n✗ FAILED:", flush=True)
        for test, reason in failed_tests:
            print(f"  {test} ({reason})", flush=True)
    print(f"{'='*60}\n", flush=True)

    return 0 if success else -1
