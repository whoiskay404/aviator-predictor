"""CSV round storage for Aviator Analyzer.

Columns: ``timestamp,multiplier,source``. Timestamps are ISO-8601 local
time for Africa/Gaborone (fixed UTC+02:00) and may be empty for rows that
have not been timed yet (seed data).
"""

from __future__ import annotations

import csv
import io
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

COLUMNS = ["timestamp", "multiplier", "source"]
VALID_SOURCES = ("manual", "import", "seed")
DEFAULT_SOURCE = "import"
MIN_MULTIPLIER = 1.0
MAX_MULTIPLIER = 100_000.0

ROOT = Path(__file__).resolve().parent

GABORONE = timezone(timedelta(hours=2), "UTC+02:00")

SEED_MULTIPLIERS: list[float] = [
    1.2, 1.18, 2.53, 3.3, 2.5, 1.88, 2.4, 2.72, 1.28, 1.34,
    5.93, 2.89, 1.61, 1.14, 2.74, 1.79, 1.37, 13.92, 2.5, 2.49,
    1.25, 1.06, 2.24, 8.1, 1.06, 2.34, 1.86, 1.3, 1.28, 3.73,
    1.44, 1.23, 1.96, 1.1, 1.91, 1.05, 1.61, 3.0, 1.16, 37.02,
    1.21, 10.42, 2.74, 1.31, 1.0, 1.83, 2.37, 9.32, 2.34, 79.88,
    1.04, 1.16, 2.27, 3.87, 1.57, 2.45, 83.73, 2.41, 1.06, 12.85,
]


def default_path() -> Path:
    env = os.environ.get("AVIATOR_CSV")
    if env:
        return Path(env)
    return ROOT / "data" / "rounds.csv"


def now_iso() -> str:
    return datetime.now(GABORONE).replace(microsecond=0).isoformat()


def parse_timestamp(text: str) -> str:
    """Validate an ISO timestamp and return it normalised."""
    raw = str(text).strip()
    if not raw:
        raise ValueError("empty timestamp")
    candidate = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"not a valid ISO timestamp: {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=GABORONE)
    return dt.isoformat(timespec="seconds")


def parse_multiplier(text: object) -> float:
    """Parse a multiplier; accepts a trailing ``x`` (``2x`` -> 2.0)."""
    raw = str(text).strip()
    if raw.lower().endswith("x"):
        raw = raw[:-1].strip()
    if not raw:
        raise ValueError("empty multiplier")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"not a number: {text!r}") from exc
    if not (MIN_MULTIPLIER <= value <= MAX_MULTIPLIER):
        raise ValueError(
            f"multiplier must be between {MIN_MULTIPLIER:.2f} and "
            f"{MAX_MULTIPLIER:g} (got {value:g})"
        )
    return float(value)


def parse_number_list(text: str) -> tuple[list[float], list[str]]:
    """Split a free-form list on commas, spaces, semicolons or newlines."""
    tokens = [t for t in re.split(r"[,;\s]+", str(text).strip()) if t]
    values: list[float] = []
    bad: list[str] = []
    for token in tokens:
        try:
            values.append(parse_multiplier(token))
        except ValueError:
            bad.append(token)
    return values, bad


def minute_of(timestamp: object) -> Optional[int]:
    if timestamp is None or (isinstance(timestamp, float) and pd.isna(timestamp)):
        return None
    raw = str(timestamp).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.minute


def hour_of(timestamp: object) -> Optional[int]:
    if timestamp is None or (isinstance(timestamp, float) and pd.isna(timestamp)):
        return None
    raw = str(timestamp).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.hour


def is_untimed(timestamp: object) -> bool:
    if timestamp is None or (isinstance(timestamp, float) and pd.isna(timestamp)):
        return True
    return not str(timestamp).strip()


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(columns=COLUMNS)
    if df is None or len(df) == 0:
        return out
    work = df.copy()
    if "timestamp" not in work.columns:
        work["timestamp"] = ""
    if "source" not in work.columns:
        work["source"] = DEFAULT_SOURCE
    out = work[COLUMNS].copy()
    out["timestamp"] = out["timestamp"].fillna("").astype(str).str.strip()
    source = out["source"].fillna(DEFAULT_SOURCE).astype(str).str.strip().str.lower()
    out["source"] = source.mask(source == "", DEFAULT_SOURCE)
    out["multiplier"] = pd.to_numeric(out["multiplier"], errors="coerce")
    out = out[out["multiplier"].notna()].reset_index(drop=True)
    return out


def validate_df(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if list(df.columns) != COLUMNS:
        errors.append(f"columns must be {COLUMNS}, got {list(df.columns)}")
        return errors
    for i, row in df.iterrows():
        n = int(i) + 1
        try:
            parse_multiplier(row["multiplier"])
        except ValueError as exc:
            errors.append(f"Row {n}: {exc}")
        source = str(row["source"]).strip().lower()
        if source not in VALID_SOURCES:
            errors.append(
                f"Row {n}: unknown source {source!r} (use manual, import or seed)"
            )
        stamp = row["timestamp"]
        if not is_untimed(stamp):
            try:
                parse_timestamp(stamp)
            except ValueError as exc:
                errors.append(f"Row {n}: {exc}")
    return errors


def read_csv_text(text: str,
                  default_source: str = DEFAULT_SOURCE) -> tuple[pd.DataFrame, list[str]]:
    """Parse uploaded/pasted CSV into a frame plus per-row error messages."""
    if text is None:
        return empty_df(), []
    cleaned = text.lstrip("﻿")
    if not cleaned.strip():
        return empty_df(), []

    rows = list(csv.reader(io.StringIO(cleaned)))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return empty_df(), []

    header = [cell.strip().lower() for cell in rows[0]]
    if "multiplier" in header:
        data_rows = rows[1:]
        has_header = True
        if "timestamp" not in header:
            return empty_df(), [
                "Header must contain a 'timestamp' column "
                "(expected: timestamp,multiplier,source)"
            ]
    else:
        if len(rows[0]) == 1:
            data_rows = rows
            header = ["multiplier"]
            has_header = False
        else:
            return empty_df(), [
                "Row 1: missing header row; expected 'timestamp,multiplier,source'"
                " (a bare single column of multipliers is also accepted)"
            ]

    index = {name: i for i, name in enumerate(header)}
    records: list[dict] = []
    errors: list[str] = []
    for offset, row in enumerate(data_rows):
        label = f"Row {offset + 1}"
        cell = row[index["multiplier"]].strip() if len(row) > index["multiplier"] else ""
        try:
            multiplier = parse_multiplier(cell)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue

        stamp_raw = ""
        if has_header and "timestamp" in index and len(row) > index["timestamp"]:
            stamp_raw = row[index["timestamp"]].strip()
        stamp = ""
        if stamp_raw:
            try:
                stamp = parse_timestamp(stamp_raw)
            except ValueError as exc:
                errors.append(f"{label}: {exc}")
                continue

        source = default_source
        if has_header and "source" in index and len(row) > index["source"]:
            candidate = row[index["source"]].strip().lower()
            if candidate:
                if candidate not in VALID_SOURCES:
                    errors.append(
                        f"{label}: unknown source {candidate!r} "
                        "(use manual, import or seed)"
                    )
                    continue
                source = candidate

        records.append({"timestamp": stamp, "multiplier": multiplier, "source": source})

    return normalize_df(pd.DataFrame(records, columns=COLUMNS)), errors


def df_to_csv(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return ",".join(COLUMNS) + "\n"
    out = normalize_df(df).copy()
    out["multiplier"] = out["multiplier"].map(lambda v: f"{float(v):.2f}")
    buffer = io.StringIO()
    out.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def template_csv() -> str:
    return (
        "timestamp,multiplier,source\n"
        "2026-03-14T20:03:25+02:00,1.50,manual\n"
        "2026-03-14T20:04:00+02:00,2.31,manual\n"
        ",1.18,seed\n"
    )


def seed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [""] * len(SEED_MULTIPLIERS),
        "multiplier": [float(v) for v in SEED_MULTIPLIERS],
        "source": ["seed"] * len(SEED_MULTIPLIERS),
    })


def sort_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """Untimed rows first (stable), then timestamped rows ascending."""
    if df is None or len(df) == 0:
        return empty_df()
    work = normalize_df(df).reset_index(drop=True)
    work["_order"] = range(len(work))
    untimed = work[work["timestamp"].map(is_untimed)]
    timed = work[~work["timestamp"].map(is_untimed)].sort_values(
        "timestamp", kind="stable"
    )
    out = pd.concat([untimed, timed], ignore_index=True)[COLUMNS]
    return out.reset_index(drop=True)


def merge_rounds(base: pd.DataFrame,
                 incoming: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Merge two round sets, de-duplicating.

    Timestamped rows de-dupe on ``timestamp + multiplier``; untimed rows
    de-dupe by per-multiplier occurrence counts (the larger count wins).
    Returns ``(merged, added, duplicates)``.
    """
    base_df = normalize_df(base)
    inc_df = normalize_df(incoming)
    if len(inc_df) == 0:
        return sort_rounds(base_df), 0, 0

    base_timed = base_df[base_df["timestamp"].map(is_untimed).astype(bool).eq(False)]
    base_untimed = base_df[base_df["timestamp"].map(is_untimed).astype(bool)]
    inc_timed = inc_df[inc_df["timestamp"].map(is_untimed).astype(bool).eq(False)]
    inc_untimed = inc_df[inc_df["timestamp"].map(is_untimed).astype(bool)]

    seen = set(zip(base_timed["timestamp"], base_timed["multiplier"].astype(float)))
    added_timed = []
    duplicates = 0
    for stamp, mult, source in inc_timed.itertuples(index=False):
        key = (stamp, float(mult))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        added_timed.append({"timestamp": stamp, "multiplier": float(mult), "source": source})

    def counts(frame: pd.DataFrame) -> dict[float, int]:
        result: dict[float, int] = {}
        for value in frame["multiplier"]:
            key = float(value)
            result[key] = result.get(key, 0) + 1
        return result

    base_counts = counts(base_untimed)
    inc_counts = counts(inc_untimed)
    inc_sources: dict[float, str] = {}
    for value, source in zip(inc_untimed["multiplier"], inc_untimed["source"]):
        inc_sources.setdefault(float(value), str(source))
    added_untimed: list[dict] = []
    for value in sorted(inc_counts):
        extra = inc_counts[value] - base_counts.get(value, 0)
        if extra > 0:
            added_untimed.extend(
                {"timestamp": "", "multiplier": value,
                 "source": inc_sources.get(value, DEFAULT_SOURCE)}
                for _ in range(extra)
            )
        duplicates += min(inc_counts[value], base_counts.get(value, 0))

    added = len(added_timed) + len(added_untimed)
    merged = pd.concat([
        base_untimed,
        pd.DataFrame(added_untimed, columns=COLUMNS),
        base_timed,
        pd.DataFrame(added_timed, columns=COLUMNS),
    ], ignore_index=True)
    return sort_rounds(merged), added, duplicates


def load_rounds(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the CSV, creating it with seed data on first run."""
    target = Path(path) if path is not None else default_path()
    if not target.exists():
        seeded = seed_df()
        save_rounds(seeded, target)
        return seeded
    text = target.read_text(encoding="utf-8-sig")
    if not text.strip():
        return empty_df()
    parsed, _errors = read_csv_text(text, default_source=DEFAULT_SOURCE)
    return sort_rounds(parsed)


def save_rounds(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    target = Path(path) if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(df_to_csv(df), encoding="utf-8")
    return target


def batch_rounds(values: Sequence[float], start: datetime,
                 interval_seconds: float) -> pd.DataFrame:
    """Build a frame for a pasted list, spaced at a fixed round interval."""
    rows = []
    for i, value in enumerate(values):
        when = start + timedelta(seconds=interval_seconds * i)
        rows.append({
            "timestamp": when.astimezone(GABORONE).isoformat(timespec="seconds"),
            "multiplier": float(value),
            "source": "manual",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stamp_untimed(df: pd.DataFrame, start: datetime,
                  interval_seconds: float) -> tuple[pd.DataFrame, int]:
    """Give untimed rows sequential timestamps starting at `start`."""
    work = normalize_df(df).reset_index(drop=True)
    count = 0
    for i in range(len(work)):
        if is_untimed(work.at[i, "timestamp"]):
            when = start + timedelta(seconds=interval_seconds * count)
            work.at[i, "timestamp"] = when.astimezone(GABORONE).isoformat(
                timespec="seconds"
            )
            count += 1
    return sort_rounds(work), count


# - enrichment (odds CSV)

ENRICH_DIR = ROOT / "csv_data_odds"
ENRICH_NAME = "aviator_rounds.csv"
ENRICH_COLUMNS = ["timestamp", "multiplier"]

_TIME_ALIASES = ("estimated_time", "timestamp", "time", "datetime", "date")
_MULT_ALIASES = ("multiplier", "odd", "odds", "value", "crash")


def enrichment_path() -> Path:
    return ENRICH_DIR / ENRICH_NAME


def read_enrichment_csv(text: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse an enrichment CSV of odds with estimated timestamps.

    Flexible columns: multiplier from ``multiplier``/``odd``/``odds``/``value``,
    time from ``estimated_time``/``timestamp``/``time``/``date``. Timeless rows
    and unreadable timestamps are kept as untimed; bad multipliers are dropped.
    Returns ``(frame with ENRICH_COLUMNS, error messages)``.
    """
    errors: list[str] = []
    try:
        raw = pd.read_csv(io.StringIO(str(text)))
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to the UI
        return pd.DataFrame(columns=ENRICH_COLUMNS), [f"could not read CSV: {exc}"]
    if raw.empty:
        return pd.DataFrame(columns=ENRICH_COLUMNS), ["CSV has no rows"]

    lookup = {str(c).strip().lower(): c for c in raw.columns}
    mult_col = next((lookup[a] for a in _MULT_ALIASES if a in lookup), None)
    time_col = next((lookup[a] for a in _TIME_ALIASES if a in lookup), None)
    if mult_col is None:
        return pd.DataFrame(columns=ENRICH_COLUMNS), [
            "no multiplier column found (expected one of: "
            + ", ".join(_MULT_ALIASES) + ")"
        ]

    stamps: list[str] = []
    mults: list[float] = []
    bad_time = 0
    for i, row in raw.iterrows():
        try:
            multiplier = parse_multiplier(row[mult_col])
        except ValueError as exc:
            errors.append(f"row {i + 2}: {exc}")
            continue
        stamp = ""
        if time_col is not None and not pd.isna(row[time_col]):
            try:
                stamp = parse_timestamp(str(row[time_col]))
            except ValueError:
                stamp = ""
                bad_time += 1
        stamps.append(stamp)
        mults.append(multiplier)

    if bad_time:
        errors.append(f"{bad_time} row(s) had unreadable timestamps; kept as "
                      "untimed")
    if not mults:
        errors.append("no valid multipliers found")
    return pd.DataFrame({"timestamp": stamps, "multiplier": mults}), errors


def load_enrichment(path: Optional[Path] = None
                    ) -> tuple[pd.DataFrame, list[str]]:
    """Load the enrichment CSV (defaults to the bundled odds file)."""
    target = Path(path) if path is not None else enrichment_path()
    if not target.exists():
        return pd.DataFrame(columns=ENRICH_COLUMNS), [
            f"enrichment file not found: {target}"
        ]
    text = target.read_text(encoding="utf-8-sig", errors="replace")
    return read_enrichment_csv(text)
