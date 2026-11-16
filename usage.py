"""Usage tracking and token cost management module for Jarvis (Phase 5).

This module tracks model token consumption, logs every LLM call to data/usage.csv,
enforces the daily token limit (DAILY_TOKEN_LIMIT), and calculates estimated costs
based on configurable prices in config.py.
"""

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import config
import state

_USAGE_LOCK = threading.Lock()


def get_usage_csv_path() -> Path:
    """Return the absolute path to data/usage.csv, creating data/ directory if needed."""
    base_dir = getattr(config, "BASE_DIR", Path(__file__).resolve().parent)
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "usage.csv"


def _ensure_csv_header(path: Path) -> None:
    """Ensure data/usage.csv exists and has the appropriate header line."""
    if not path.is_file() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "model", "input_tokens", "output_tokens", "image_tokens", "total_tokens"])


def log_token_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    image_tokens: int = 0,
    timestamp: str | None = None,
) -> None:
    """Log the token counts of a single model call to data/usage.csv.

    Args:
        model: Model identifier (e.g. 'llama3.1').
        input_tokens: Number of prompt/context tokens evaluated.
        output_tokens: Number of completion tokens generated.
        image_tokens: Number of vision/image tokens processed (default: 0).
        timestamp: Optional ISO timestamp override (defaults to datetime.now().isoformat()).
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    input_tokens = max(0, int(input_tokens))
    output_tokens = max(0, int(output_tokens))
    image_tokens = max(0, int(image_tokens))
    total_tokens = input_tokens + output_tokens + image_tokens

    with _USAGE_LOCK:
        path = get_usage_csv_path()
        _ensure_csv_header(path)
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, model, input_tokens, output_tokens, image_tokens, total_tokens])
        except Exception as err:
            print(f"[Usage Warning] Failed to log token usage: {err}")


def calculate_cost(input_tokens: int, output_tokens: int, image_tokens: int = 0) -> float:
    """Calculate estimated dollar cost given input, output, and image token counts.

    Uses PRICE_PER_M_INPUT_TOKENS and PRICE_PER_M_OUTPUT_TOKENS from config.py.
    Image tokens are priced at the prompt input token rate.
    """
    price_in = getattr(config, "PRICE_PER_M_INPUT_TOKENS", 0.15)
    price_out = getattr(config, "PRICE_PER_M_OUTPUT_TOKENS", 0.60)
    cost_in = ((input_tokens + image_tokens) / 1_000_000.0) * price_in
    cost_out = (output_tokens / 1_000_000.0) * price_out
    return cost_in + cost_out


def format_cost(cost: float) -> str:
    """Format a dollar amount cleanly with appropriate precision."""
    if cost <= 0:
        return "$0.0000"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def get_usage_summary() -> dict[str, Any]:
    """Calculate token usage and cost metrics for today and the current month.

    Returns:
        Dictionary containing 'today', 'this_month', 'daily_limit',
        'is_limit_exceeded', and 'is_overridden'.
    """
    now = datetime.now()
    today_date = now.date()
    current_year = now.year
    current_month = now.month

    today_in, today_out, today_img = 0, 0, 0
    month_in, month_out, month_img = 0, 0, 0

    with _USAGE_LOCK:
        path = get_usage_csv_path()
        if path.is_file():
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip CSV header
                    for row in reader:
                        if not row or len(row) < 5:
                            continue
                        try:
                            ts_str = row[0]
                            dt_part = ts_str.split("T")[0]
                            row_date = datetime.strptime(dt_part, "%Y-%m-%d").date()
                            row_in = int(row[2])
                            row_out = int(row[3])
                            # Support both legacy 5-column and 6-column formats
                            row_img = int(row[4]) if len(row) >= 6 else 0
                        except Exception:
                            continue

                        # Check month match
                        if row_date.year == current_year and row_date.month == current_month:
                            month_in += row_in
                            month_out += row_out
                            month_img += row_img

                        # Check day match
                        if row_date == today_date:
                            today_in += row_in
                            today_out += row_out
                            today_img += row_img
            except Exception as err:
                print(f"[Usage Warning] Error reading usage.csv: {err}")

    today_total = today_in + today_out + today_img
    month_total = month_in + month_out + month_img
    today_cost = calculate_cost(today_in, today_out, today_img)
    month_cost = calculate_cost(month_in, month_out, month_img)

    daily_limit = getattr(config, "DAILY_TOKEN_LIMIT", 200000)
    is_overridden = is_limit_overridden()
    is_exceeded = (today_total >= daily_limit) and not is_overridden
    pct_used = (today_total / daily_limit * 100.0) if daily_limit > 0 else 0.0

    return {
        "today": {
            "input_tokens": today_in,
            "output_tokens": today_out,
            "image_tokens": today_img,
            "total_tokens": today_total,
            "cost": today_cost,
        },
        "this_month": {
            "input_tokens": month_in,
            "output_tokens": month_out,
            "image_tokens": month_img,
            "total_tokens": month_total,
            "cost": month_cost,
        },
        "daily_limit": daily_limit,
        "percentage_used": pct_used,
        "is_limit_exceeded": is_exceeded,
        "is_overridden": is_overridden,
    }


def is_limit_overridden() -> bool:
    """Check if the user has overridden the daily token limit for today."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    override_date = state.get_token_limit_override_date()
    return override_date == today_str


def override_daily_limit() -> None:
    """Allow further model calls today by marking the token limit as overridden."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    state.set_token_limit_override_date(today_str)


def is_daily_limit_exceeded() -> bool:
    """Check whether today's token usage has reached DAILY_TOKEN_LIMIT without override."""
    if is_limit_overridden():
        return False
    summary = get_usage_summary()
    return summary["is_limit_exceeded"]


def format_usage_report(summary: dict[str, Any] | None = None) -> str:
    """Return a clean formatted table summarizing today's and this month's usage."""
    if summary is None:
        summary = get_usage_summary()

    today = summary["today"]
    month = summary["this_month"]
    limit = summary["daily_limit"]
    pct = summary["percentage_used"]

    if summary["is_limit_exceeded"]:
        limit_status = f"{today['total_tokens']:,} / {limit:,} tokens ({pct:.1f}% used - LIMIT REACHED)"
    elif summary["is_overridden"]:
        limit_status = f"{today['total_tokens']:,} / {limit:,} tokens ({pct:.1f}% used - OVERRIDDEN)"
    else:
        limit_status = f"{today['total_tokens']:,} / {limit:,} tokens ({pct:.1f}% used)"

    border = "-" * 50
    lines = [
        "Token Usage & Cost Summary:",
        border,
        "Today:",
        f"  • Input Tokens:  {today['input_tokens']:,}",
        f"  • Output Tokens: {today['output_tokens']:,}",
        f"  • Image Tokens:  {today.get('image_tokens', 0):,}",
        f"  • Total Tokens:  {today['total_tokens']:,}",
        f"  • Est. Cost:     {format_cost(today['cost'])}",
        f"  • Daily Limit:   {limit_status}",
        "",
        "This Month:",
        f"  • Input Tokens:  {month['input_tokens']:,}",
        f"  • Output Tokens: {month['output_tokens']:,}",
        f"  • Image Tokens:  {month.get('image_tokens', 0):,}",
        f"  • Total Tokens:  {month['total_tokens']:,}",
        f"  • Est. Cost:     {format_cost(month['cost'])}",
        border,
    ]
    return "\n".join(lines)


def format_spoken_usage_report(summary: dict[str, Any] | None = None) -> str:
    """Return a concise, natural summary suitable for Text-to-Speech (< 20s)."""
    if summary is None:
        summary = get_usage_summary()

    today = summary["today"]
    month = summary["this_month"]
    pct = summary["percentage_used"]

    cost_str = format_cost(today["cost"])
    if summary["is_limit_exceeded"]:
        status = "Your daily token limit has been reached."
    elif summary["is_overridden"]:
        status = "Your daily limit is currently overridden."
    else:
        status = f"You have used {pct:.1f} percent of your daily limit."

    img_count = today.get("image_tokens", 0)
    img_clause = f" including {img_count:,} image tokens," if img_count > 0 else ""

    return (
        f"Today you have used {today['total_tokens']:,} tokens{img_clause} costing approximately {cost_str}. "
        f"This month's total is {month['total_tokens']:,} tokens. {status}"
    )
