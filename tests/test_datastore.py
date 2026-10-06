from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd
import pytest

import datastore as ds


def round_trip(text: str):
    frame, errors = ds.read_csv_text(text)
    return frame, errors


# - parsing

def test_parse_multiplier_plain():
    assert ds.parse_multiplier("1.5") == pytest.approx(1.5)
    assert ds.parse_multiplier("2") == pytest.approx(2.0)


def test_parse_multiplier_trailing_x():
    assert ds.parse_multiplier("1.75x") == pytest.approx(1.75)
    assert ds.parse_multiplier("  3X  ") == pytest.approx(3.0)


def test_parse_multiplier_rejects_comma_decimal():
    with pytest.raises(ValueError):
        ds.parse_multiplier("1,5")


def test_parse_multiplier_invalid():
    for bad in ["", "abc", "0", "0.5", "-2", "x", "1e100"]:
        with pytest.raises(ValueError):
            ds.parse_multiplier(bad)


def test_parse_multiplier_bounds():
    assert ds.parse_multiplier("100000") == pytest.approx(100000.0)
    with pytest.raises(ValueError):
        ds.parse_multiplier("100001")


def test_parse_number_list():
    values, bad = ds.parse_number_list("1.5 2, 3.0x\nfoo 0.2")
    assert values == [1.5, 2.0, 3.0]
    assert "foo" in bad and "0.2" in bad


def test_parse_number_list_empty():
    values, bad = ds.parse_number_list("")
    assert values == [] and bad == []


# - timestamps

def test_now_iso_format():
    stamp = ds.now_iso()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset() == timedelta(hours=2)
    assert parsed.tzinfo is not None


def test_parse_timestamp_variants():
    assert ds.parse_timestamp("2025-03-01T14:05:00+02:00") is not None
    assert ds.parse_timestamp("2025-03-01 14:05:00") == "2025-03-01T14:05:00+02:00"
    assert ds.parse_timestamp("2025-03-01T14:05") is not None
    with pytest.raises(ValueError):
        ds.parse_timestamp("not a time")
    with pytest.raises(ValueError):
        ds.parse_timestamp("")


def test_is_untimed():
    assert ds.is_untimed("")
    assert ds.is_untimed(None)
    assert ds.is_untimed(float("nan"))
    assert not ds.is_untimed("2025-03-01T14:05:00+02:00")


def test_minute_and_hour_of():
    stamp = "2025-03-01T22:07:00+02:00"
    assert ds.minute_of(stamp) == 7
    assert ds.hour_of(stamp) == 22
    assert ds.minute_of("") is None
    assert ds.hour_of("") is None


# - CSV

def test_read_csv_text_round_trip():
    source = "timestamp,multiplier,source\n2025-03-01T14:05:00+02:00,1.50,manual\n"
    frame, errors = round_trip(source)
    assert errors == []
    assert len(frame) == 1
    assert frame.iloc[0]["multiplier"] == pytest.approx(1.5)
    assert frame.iloc[0]["source"] == "manual"


def test_read_csv_text_missing_source_defaults_to_import():
    source = "timestamp,multiplier\n2025-03-01T14:05:00+02:00,2.00\n"
    frame, errors = round_trip(source)
    assert errors == []
    assert frame.iloc[0]["source"] == "import"


def test_read_csv_text_headerless_single_column():
    frame, errors = round_trip("1.25\n1.50\n2.00\n")
    assert errors == []
    assert list(frame["multiplier"]) == [1.25, 1.5, 2.0]
    assert (frame["source"] == "import").all()
    assert (frame["timestamp"] == "").all()


def test_read_csv_text_headerless_two_columns_error():
    frame, errors = round_trip("1.25,manual\n1.50,manual\n")
    assert len(frame) == 0
    assert any("timestamp" in error.lower() or "header" in error.lower()
               for error in errors)


def test_read_csv_text_bad_header():
    frame, errors = round_trip("foo,bar\n1,2\n")
    assert len(frame) == 0
    assert errors


def test_read_csv_text_empty():
    frame, errors = round_trip("")
    assert len(frame) == 0
    assert errors == []


def test_read_csv_text_reports_bad_rows():
    source = (
        "timestamp,multiplier,source\n"
        "2025-03-01T14:05:00+02:00,1.50,manual\n"
        "2025-03-01T14:05:30+02:00,0.50,manual\n"
        "2025-03-01T14:06:00+02:00,abc,manual\n"
    )
    frame, errors = round_trip(source)
    assert len(frame) == 1
    assert len(errors) == 2
    assert any(error.startswith("Row 3") for error in errors)


def test_df_to_csv_and_back():
    frame = pd.DataFrame({
        "timestamp": ["2025-03-01T14:05:00+02:00", ""],
        "multiplier": [1.501, 83.7],
        "source": ["manual", "seed"],
    })
    text = ds.df_to_csv(frame)
    back, errors = round_trip(text)
    assert errors == []
    assert len(back) == 2
    assert back.iloc[0]["multiplier"] == pytest.approx(1.50)
    assert back.iloc[1]["multiplier"] == pytest.approx(83.70)


def test_template_csv_parses():
    frame, errors = round_trip(ds.template_csv())
    assert errors == []
    assert len(frame) >= 2


# - frames

def test_empty_df_columns():
    frame = ds.empty_df()
    assert list(frame.columns) == ds.COLUMNS
    assert len(frame) == 0


def test_normalize_df_fills_and_casts():
    frame = pd.DataFrame({
        "timestamp": [None, "2025-03-01T14:05:00+02:00"],
        "multiplier": ["1.50", "2"],
        "source": [None, "MANUAL"],
    })
    out = ds.normalize_df(frame)
    assert out.iloc[0]["timestamp"] == ""
    assert out.iloc[0]["source"] == "import"
    assert out.iloc[1]["source"] == "manual"
    assert out["multiplier"].dtype.kind == "f"


def test_validate_df_errors():
    bad = pd.DataFrame({
        "timestamp": ["2025-03-01T14:05:00+02:00"],
        "multiplier": [0.5],
        "source": ["manual"],
    })
    errors = ds.validate_df(bad)
    assert any("1.00" in error for error in errors)

    bad2 = pd.DataFrame({
        "timestamp": [""],
        "multiplier": [2.0],
        "source": ["mystery"],
    })
    errors2 = ds.validate_df(bad2)
    assert any("source" in error for error in errors2)

    good = ds.seed_df()
    assert ds.validate_df(good) == []


# - seed

def test_seed_df():
    frame = ds.seed_df()
    assert len(frame) == 60
    assert list(frame.columns) == ds.COLUMNS
    assert (frame["timestamp"] == "").all()
    assert (frame["source"] == "seed").all()
    assert frame["multiplier"].min() == pytest.approx(1.0)
    assert frame["multiplier"].max() == pytest.approx(83.73)
    assert frame["multiplier"].mean() == pytest.approx(5.954, abs=5e-3)
    assert frame["multiplier"].median() == pytest.approx(1.935, abs=5e-3)
    assert frame["multiplier"].std(ddof=1) == pytest.approx(15.126, abs=5e-3)


def test_seed_values_match_legacy_first_rows():
    assert list(ds.seed_df()["multiplier"][:5]) == pytest.approx(
        ds.SEED_MULTIPLIERS[:5]
    )
    assert ds.SEED_MULTIPLIERS[0] == pytest.approx(1.2)
    assert ds.SEED_MULTIPLIERS[-1] == pytest.approx(12.85)


# - merge

def _timed(rows):
    return pd.DataFrame(rows, columns=ds.COLUMNS)


def test_merge_drops_identical_timed_duplicates():
    base = _timed([("2025-03-01T14:05:00+02:00", 1.5, "manual")])
    incoming = _timed([("2025-03-01T14:05:00+02:00", 1.5, "import")])
    merged, added, dupes = ds.merge_rounds(base, incoming)
    assert added == 0
    assert dupes == 1
    assert len(merged) == 1


def test_merge_keeps_different_multiplier_same_time():
    base = _timed([("2025-03-01T14:05:00+02:00", 1.5, "manual")])
    incoming = _timed([("2025-03-01T14:05:00+02:00", 2.5, "import")])
    merged, added, dupes = ds.merge_rounds(base, incoming)
    assert added == 1
    assert dupes == 0
    assert len(merged) == 2


def test_merge_untimed_counts_max_per_multiplier():
    base = _timed([("", 1.5, "seed"), ("", 1.5, "seed"), ("", 2.0, "seed")])
    incoming = _timed([("", 1.5, "seed"), ("", 3.0, "seed")])
    merged, added, dupes = ds.merge_rounds(base, incoming)
    counts = merged[merged["timestamp"] == ""]["multiplier"].value_counts()
    assert counts[1.5] == 2
    assert counts[2.0] == 1
    assert counts[3.0] == 1
    assert added == 1


def test_merge_untimed_shrinks_never():
    base = _timed([("", 1.5, "seed"), ("", 1.5, "seed")])
    incoming = _timed([("", 1.5, "seed")])
    _, added, dupes = ds.merge_rounds(base, incoming)
    assert added == 0 and dupes == 1
    merged, _, _ = ds.merge_rounds(base, incoming)
    assert len(merged[merged["timestamp"] == ""]) == 2


def test_merge_combines_sources():
    base = ds.empty_df()
    incoming = _timed([("", 1.5, "seed")])
    merged, added, _ = ds.merge_rounds(base, incoming)
    assert added == 1
    assert merged.iloc[0]["source"] == "seed"


def test_merge_sorts():
    base = _timed([("2025-03-01T15:00:00+02:00", 1.5, "manual")])
    incoming = _timed([
        ("2025-03-01T14:00:00+02:00", 2.0, "import"),
        ("", 3.0, "seed"),
    ])
    merged, _, _ = ds.merge_rounds(base, incoming)
    stamps = list(merged["timestamp"])
    assert stamps[0] == ""
    assert stamps[1] == "2025-03-01T14:00:00+02:00"
    assert stamps[2] == "2025-03-01T15:00:00+02:00"


# - batch helpers

def test_batch_rounds_sequential():
    start = datetime(2025, 3, 1, 14, 0, tzinfo=ds.GABORONE)
    frame = ds.batch_rounds([1.1, 1.2, 1.3], start, 30.0)
    assert len(frame) == 3
    assert frame.iloc[0]["timestamp"] == "2025-03-01T14:00:00+02:00"
    assert frame.iloc[1]["timestamp"] == "2025-03-01T14:00:30+02:00"
    assert frame.iloc[2]["timestamp"] == "2025-03-01T14:01:00+02:00"
    assert (frame["source"] == "manual").all()


def test_batch_rounds_empty():
    start = datetime(2025, 3, 1, 14, 0, tzinfo=ds.GABORONE)
    assert len(ds.batch_rounds([], start, 30.0)) == 0


def test_stamp_untimed():
    frame = ds.seed_df()
    start = datetime(2025, 3, 1, 14, 0, tzinfo=ds.GABORONE)
    stamped, count = ds.stamp_untimed(frame, start, 30.0)
    assert count == 60
    assert stamped["timestamp"].map(ds.is_untimed).sum() == 0
    first = stamped.iloc[0]["timestamp"]
    assert first == "2025-03-01T14:00:00+02:00"


def test_stamp_untimed_nothing_to_do():
    frame = _timed([("2025-03-01T14:05:00+02:00", 1.5, "manual")])
    start = datetime(2025, 3, 1, 14, 0, tzinfo=ds.GABORONE)
    stamped, count = ds.stamp_untimed(frame, start, 30.0)
    assert count == 0


def test_stamp_untimed_keeps_timed_rows():
    frame = pd.concat([
        _timed([("2025-03-01T10:00:00+02:00", 2.0, "manual")]),
        _timed([("", 1.5, "seed")]),
    ], ignore_index=True)
    start = datetime(2025, 3, 1, 14, 0, tzinfo=ds.GABORONE)
    stamped, count = ds.stamp_untimed(frame, start, 60.0)
    assert count == 1
    assert stamped.iloc[0]["timestamp"] == "2025-03-01T10:00:00+02:00"


def test_sort_rounds_untimed_first():
    frame = _timed([
        ("2025-03-01T15:00:00+02:00", 1.5, "manual"),
        ("", 9.0, "seed"),
        ("2025-03-01T14:00:00+02:00", 2.5, "manual"),
    ])
    out = ds.sort_rounds(frame)
    assert out.iloc[0]["timestamp"] == ""
    assert out.iloc[1]["timestamp"] == "2025-03-01T14:00:00+02:00"


# - IO

def test_read_write_round_trip(tmp_path, monkeypatch):
    target = tmp_path / "rounds.csv"
    monkeypatch.setenv("AVIATOR_CSV", str(target))
    assert not target.exists()
    frame = ds.load_rounds()
    assert len(frame) == 60
    assert target.exists()

    bigger = pd.concat([frame, pd.DataFrame([{
        "timestamp": "2025-03-01T14:00:00+02:00",
        "multiplier": 1.75,
        "source": "manual",
    }])], ignore_index=True)
    ds.save_rounds(bigger)

    reloaded = ds.load_rounds()
    assert len(reloaded) == 61
    assert reloaded["multiplier"].max() == pytest.approx(83.73)


def test_load_rounds_existing_file_without_seeding(tmp_path, monkeypatch):
    target = tmp_path / "rounds.csv"
    monkeypatch.setenv("AVIATOR_CSV", str(target))
    frame = ds.seed_df()
    ds.save_rounds(frame)
    saved = target.read_text(encoding="utf-8")
    reloaded = ds.load_rounds()
    assert len(reloaded) == 60
    assert target.read_text(encoding="utf-8") == saved


def test_default_path_env_override(tmp_path, monkeypatch):
    from pathlib import Path

    target = tmp_path / "custom.csv"
    monkeypatch.setenv("AVIATOR_CSV", str(target))
    assert ds.default_path() == Path(str(target))
    assert ds.default_path().name == "custom.csv"


def test_valid_sources():
    assert set(ds.VALID_SOURCES) == {"manual", "import", "seed"}


# - enrichment CSV

def test_read_enrichment_csv_bundled_format():
    text = ("seq,segment,estimated_time,multiplier\n"
            "1,1,2026-10-06 13:14:30,5.21\n"
            "2,1,2026-10-06 13:14:52,1.92\n"
            "3,1,,1.30\n")
    frame, errors = ds.read_enrichment_csv(text)
    assert errors == []
    assert len(frame) == 3
    assert frame.iloc[0]["multiplier"] == pytest.approx(5.21)
    assert frame.iloc[0]["timestamp"].endswith("+02:00")
    assert ds.minute_of(frame.iloc[0]["timestamp"]) == 14
    assert frame.iloc[2]["timestamp"] == ""


def test_read_enrichment_csv_odd_column_aliases():
    text = ("odd,timestamp\n1.50,2026-10-06 13:20:00\n")
    frame, errors = ds.read_enrichment_csv(text)
    assert errors == []
    assert len(frame) == 1
    assert frame.iloc[0]["multiplier"] == pytest.approx(1.50)


def test_read_enrichment_csv_bad_rows():
    text = ("multiplier,estimated_time\n"
            "notanumber,2026-10-06 13:14:30\n"
            "1.5,not-a-date\n"
            "0.5,2026-10-06 13:15:00\n")
    frame, errors = ds.read_enrichment_csv(text)
    assert len(frame) == 1
    assert frame.iloc[0]["multiplier"] == pytest.approx(1.5)
    assert frame.iloc[0]["timestamp"] == ""
    assert any("not a number" in e for e in errors)
    assert any("unreadable timestamp" in e for e in errors)


def test_read_enrichment_csv_missing_multiplier_column():
    frame, errors = ds.read_enrichment_csv("a,b\n1,2\n")
    assert len(frame) == 0
    assert any("multiplier" in e for e in errors)


def test_load_enrichment_bundled():
    frame, errors = ds.load_enrichment()
    if len(frame) == 0:
        pytest.skip(f"bundled enrichment file missing: {errors}")
    assert len(frame) >= 300
    timed = sum(1 for t in frame["timestamp"] if not ds.is_untimed(t))
    assert timed >= len(frame) * 0.95
    assert frame["multiplier"].min() >= 1.0


def test_load_enrichment_missing_file(tmp_path):
    frame, errors = ds.load_enrichment(tmp_path / "nope.csv")
    assert len(frame) == 0
    assert errors
