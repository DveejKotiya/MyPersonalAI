"""Automated test runner for Jarvis.

Runs all test suites across all project phases, collects results, and displays
a clean pass/fail summary table with execution timings.

Usage:
    python run_tests.py
"""

import sys
import time
import unittest
from io import StringIO
from pathlib import Path


def run_all_tests() -> int:
    """Discover and execute all project test suites and print a summary table.

    Returns:
        0 if all tests pass, 1 if any failure or error occurs.
    """
    base_dir = Path(__file__).resolve().parent
    test_files = sorted(
        [
            f.name
            for f in base_dir.glob("*test*.py")
            if f.is_file() and not f.name.startswith((".", "_")) and f.name != "run_tests.py"
        ]
    )

    print("=" * 72)
    print("  Jarvis Automated Test Suite Runner")
    print(f"  Discovered {len(test_files)} test suite(s)")
    print("=" * 72)

    total_tests = 0
    total_failures = 0
    total_errors = 0
    suite_results: list[dict[str, str | int | float]] = []

    t_suite_start = time.perf_counter()

    for test_file in test_files:
        module_name = test_file[:-3]  # Strip '.py'
        t0 = time.perf_counter()

        loader = unittest.TestLoader()
        try:
            suite = loader.loadTestsFromName(module_name)
        except Exception as load_err:
            suite_results.append({
                "file": test_file,
                "count": 0,
                "status": "LOAD ERROR",
                "duration": 0.0,
                "error": str(load_err),
            })
            total_errors += 1
            continue

        # Suppress test output noise during execution to keep output clean
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=1)
        result = runner.run(suite)
        elapsed = time.perf_counter() - t0

        test_count = result.testsRun
        failures_count = len(result.failures)
        errors_count = len(result.errors)

        total_tests += test_count
        total_failures += failures_count
        total_errors += errors_count

        if failures_count == 0 and errors_count == 0:
            status = "PASS"
        else:
            status = f"FAIL ({failures_count}F, {errors_count}E)"

        suite_results.append({
            "file": test_file,
            "count": test_count,
            "status": status,
            "duration": elapsed,
            "output": stream.getvalue(),
        })

    total_elapsed = time.perf_counter() - t_suite_start

    # Print summary table
    print(f"\n{'Test Suite File':<28} {'Tests':<8} {'Time (s)':<12} {'Status'}")
    print("-" * 72)
    for res in suite_results:
        print(f"{res['file']:<28} {res['count']:<8} {res['duration']:<12.2f} {res['status']}")
    print("-" * 72)
    print(
        f"Total: {total_tests} test(s) across {len(test_files)} suite(s) "
        f"in {total_elapsed:.2f}s."
    )

    if total_failures == 0 and total_errors == 0:
        print("\n[SUCCESS] ALL TEST SUITES PASSED!\n")
        return 0
    else:
        print(f"\n[FAILURE] {total_failures} failure(s) and {total_errors} error(s) detected.\n")
        print("=" * 72)
        print("  Failure & Error Details:")
        print("=" * 72)
        for res in suite_results:
            if "FAIL" in str(res.get("status", "")) or "ERROR" in str(res.get("status", "")):
                print(f"\n--- {res['file']} ---")
                print(res.get("output", ""))
                if "error" in res:
                    print(res["error"])
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
