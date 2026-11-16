"""Log Viewer CLI for Jarvis (Phase 5).

Provides a clean, aligned tabular view of audit activity in logs/actions.log:
  python logviewer.py --last 20
  python logviewer.py --tool open_app
  python logviewer.py --errors
  python logviewer.py --today

Highlights errors and formats timestamps, tool names, arguments, and execution results.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Enable ANSI colors on Windows console
if sys.platform == "win32":
    os.system("")

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_RED_BOLD = "\033[91m\033[1m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_DIM = "\033[90m"


def get_log_file_path() -> Path:
    """Resolve path to logs/actions.log based on project root."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "logs" / "actions.log"


def load_log_entries(log_file: Path) -> list[dict[str, Any]]:
    """Parse JSON lines from logs/actions.log into dictionaries."""
    if not log_file.is_file():
        return []

    entries = []
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    entry = json.loads(clean_line)
                    if isinstance(entry, dict):
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as err:
        print(f"[LogViewer Error] Could not read {log_file}: {err}", file=sys.stderr)

    return entries


def format_arguments(args: Any) -> str:
    """Format argument dictionary into a compact string."""
    if not args:
        return "(none)"
    if isinstance(args, dict):
        pairs = []
        for k, v in args.items():
            val_str = str(v)
            if len(val_str) > 30:
                val_str = val_str[:27] + "..."
            pairs.append(f"{k}={val_str!r}")
        return ", ".join(pairs)
    val = str(args)
    return (val[:35] + "...") if len(val) > 35 else val


def format_time(ts_str: str) -> str:
    """Convert ISO timestamp to readable local time."""
    try:
        # Handles ISO format like '2026-10-30T11:45:00.123456+00:00'
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # Fallback to direct slice
        return ts_str[:19].replace("T", " ")


def is_error_entry(entry: dict[str, Any]) -> bool:
    """Determine whether a log entry represents a failed or cancelled action."""
    status = entry.get("status", "").upper()
    return status != "SUCCESS"


def filter_entries(
    entries: list[dict[str, Any]],
    tool_filter: str | None = None,
    errors_only: bool = False,
    today_only: bool = False,
    last_n: int | None = None,
) -> list[dict[str, Any]]:
    """Apply filters and slicing to log entries."""
    filtered = entries

    # Filter by --today
    if today_only:
        today_utc = datetime.now(UTC).strftime("%Y-%m-%d")
        today_local = datetime.now().strftime("%Y-%m-%d")
        filtered = [
            e for e in filtered
            if e.get("timestamp", "").startswith(today_utc) or e.get("timestamp", "").startswith(today_local)
        ]

    # Filter by --tool
    if tool_filter:
        target_tool = tool_filter.strip().lower()
        filtered = [
            e for e in filtered
            if e.get("tool", "").lower() == target_tool
        ]

    # Filter by --errors
    if errors_only:
        filtered = [e for e in filtered if is_error_entry(e)]

    # Slice by --last
    if last_n is not None and last_n > 0:
        filtered = filtered[-last_n:]

    return filtered


def print_table(entries: list[dict[str, Any]], total_count: int, log_path: Path) -> None:
    """Print an aligned ASCII table with highlighted error statuses."""
    if not entries:
        print("\nNo matching log entries found.")
        print(f"Log source: {log_path} (total entries: {total_count})\n")
        return

    # Calculate column widths
    w_time = 19
    w_tool = 16
    w_args = 36
    w_result = 45

    header_time = "TIME"
    header_tool = "TOOL"
    header_args = "ARGUMENTS"
    header_res = "RESULT"

    sep = (
        f"+{'-' * (w_time + 2)}"
        f"+{'-' * (w_tool + 2)}"
        f"+{'-' * (w_args + 2)}"
        f"+{'-' * (w_result + 2)}+"
    )

    header_line = (
        f"| {header_time:<{w_time}} "
        f"| {header_tool:<{w_tool}} "
        f"| {header_args:<{w_args}} "
        f"| {header_res:<{w_result}} |"
    )

    print()
    print(sep)
    print(header_line)
    print(sep)

    error_count = 0

    for entry in entries:
        ts = format_time(entry.get("timestamp", ""))
        tool_name = entry.get("tool", "")[:w_tool]
        args_str = format_arguments(entry.get("arguments", ""))
        if len(args_str) > w_args:
            args_str = args_str[:w_args - 3] + "..."

        status = entry.get("status", "UNKNOWN")
        output = entry.get("output_preview", "").replace("\n", " ")
        if len(output) > 28:
            output = output[:25] + "..."

        is_err = is_error_entry(entry)
        if is_err:
            error_count += 1
            status_tag = f"[{status}]"
            res_content = f"{status_tag} {output}".strip()
            if len(res_content) > w_result:
                res_content = res_content[:w_result - 3] + "..."
            colored_res = f"{COLOR_RED_BOLD}{res_content:<{w_result}}{COLOR_RESET}"
        else:
            status_tag = "[OK]"
            res_content = f"{status_tag} {output}".strip()
            if len(res_content) > w_result:
                res_content = res_content[:w_result - 3] + "..."
            colored_res = f"{COLOR_GREEN}{res_content:<{w_result}}{COLOR_RESET}"

        row_line = (
            f"| {ts:<{w_time}} "
            f"| {tool_name:<{w_tool}} "
            f"| {args_str:<{w_args}} "
            f"| {colored_res} |"
        )
        print(row_line)

    print(sep)

    # Footer summary
    error_summary = (
        f"{COLOR_RED_BOLD}{error_count} error{'s' if error_count != 1 else ''}{COLOR_RESET}"
        if error_count > 0 else "0 errors"
    )
    print(f"Showing {len(entries)} of {total_count} logged actions ({error_summary}).\n")


def main() -> None:
    """Parse CLI flags and display filtered log actions."""
    parser = argparse.ArgumentParser(
        description="Jarvis Action Log Viewer - inspect and filter audit activity in logs/actions.log"
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Number of most recent entries to display (default: 20 when no other filters given)",
    )
    parser.add_argument(
        "--tool",
        type=str,
        metavar="NAME",
        help="Filter entries by tool name (e.g., 'open_app', 'get_weather')",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Show only actions that encountered errors or were cancelled",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Show only actions executed today",
    )

    args = parser.parse_args()

    log_path = get_log_file_path()
    if not log_path.is_file():
        print(f"\nNo action logs found at {log_path}.")
        print("Logs are automatically created as tools are executed.\n")
        sys.exit(0)

    entries = load_log_entries(log_path)
    total_entries = len(entries)

    # Determine default limit
    last_n = args.last
    if last_n is None and not (args.tool or args.errors or args.today):
        last_n = 20

    filtered = filter_entries(
        entries=entries,
        tool_filter=args.tool,
        errors_only=args.errors,
        today_only=args.today,
        last_n=last_n,
    )

    print_table(filtered, total_count=total_entries, log_path=log_path)


if __name__ == "__main__":
    main()

