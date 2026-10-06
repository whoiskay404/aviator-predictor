from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest
from streamlit.testing.v1 import AppTest

import datastore as ds

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIATOR_CSV", str(tmp_path / "rounds.csv"))
    monkeypatch.setenv("AVIATOR_NO_AUTOREFRESH", "1")
    return AppTest.from_file(APP, default_timeout=120)


def button_by_label(at, label_prefix: str, exact: bool = False):
    for button in at.button:
        if exact:
            if str(button.label) == label_prefix:
                return button
        elif str(button.label).startswith(label_prefix):
            return button
    raise AssertionError(f"button matching {label_prefix!r} not found")


def test_app_loads_with_seed_data(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "Aviator Predictor"
    assert [t.label for t in app.tabs] == ["Predict", "Data"]


def test_predict_button_produces_prediction(app):
    app.run()
    button_by_label(app, "PREDICT").click().run()
    assert not app.exception
    assert "prediction" in app.session_state
    blocks = [str(b.value) for b in app.success] + \
             [str(b.value) for b in app.warning]
    assert any("ODD" in block for block in blocks)
    pred = app.session_state["prediction"]
    assert 0 <= pred["minute"] <= 59
    assert pred["decision"] in {"go", "go_data", "skip"}
    assert 0.0 <= pred["headline_band"] <= 1.0
    assert 1.0 <= pred["odd"] <= 2.0
    assert pred["odd_p"] == pytest.approx(0.97 / pred["odd"])
    assert pred["golden"] == (pred["decision"] == "go")


def test_reset_button_clears_prediction(app):
    app.run()
    button_by_label(app, "PREDICT").click().run()
    assert "prediction" in app.session_state
    button_by_label(app, "Reset", exact=True).click().run()
    assert not app.exception
    assert "prediction" not in app.session_state


def test_predict_outside_window_still_gives_odd(app):
    app.run()
    app.selectbox(key="predict_minute").set_value("16").run()
    button_by_label(app, "PREDICT").click().run()
    assert not app.exception
    pred = app.session_state["prediction"]
    assert pred["decision"] == "skip"
    assert not pred["golden"]
    assert 1.0 <= pred["odd"] <= 1.45
    assert not pred["odd_promoted"]
    assert "enriched CSV" in pred["odd_source"]
    warnings = "\n".join(str(b.value) for b in app.warning)
    assert "OUTSIDE GOLDEN WINDOW" in warnings
    assert f"{pred['odd']:.2f}x" in warnings
    joined = (warnings + "\n" +
              "\n".join(str(b.value) for b in app.success)).lower()
    assert "sit this round out" not in joined
    assert "no stake" not in joined


def test_predict_never_repeats_recent_odd(app):
    app.run()
    odds = []
    for _ in range(3):
        button_by_label(app, "PREDICT").click().run()
        assert not app.exception
        pred = app.session_state["prediction"]
        assert pred["odd_p"] == pytest.approx(0.97 / pred["odd"])
        odds.append(pred["odd"])
    assert odds[1] != odds[0]
    assert odds[2] not in odds[:2]


def test_predict_with_timestamped_data(app):
    app.run()
    # give the app 400 timestamped rounds so the window data is real
    import random

    random.seed(1)
    rows = ["timestamp,multiplier,source"]
    base = datetime(2026, 3, 1, 12, 0, tzinfo=ds.GABORONE)
    for i in range(600):
        minute = i % 60
        r = random.random()
        value = 1.05 if r < 0.2 else (1.6 if r < 0.6 else 3.5)
        rows.append(
            f"{(base + timedelta(seconds=30 * i)).isoformat(timespec='seconds')},"
            f"{value:.2f},manual"
        )
    ds.save_rounds(ds.read_csv_text("\n".join(rows))[0])
    fresh = AppTest.from_file(APP, default_timeout=120)
    fresh.run()
    button_by_label(fresh, "PREDICT").click().run()
    assert not fresh.exception
    pred = fresh.session_state["prediction"]
    assert pred["window_n"] > 0
    assert pred["confidence"] in {"data-backed", "provisional", "theoretical"}


def test_predict_tab_has_quality_strip_and_plan(app):
    app.run()
    assert not app.exception
    metrics = [str(block.label) for block in app.metric]
    assert "Timestamped rounds" in metrics
    assert "Inside your windows" in metrics
    assert "Prediction mode" in metrics
    headers = [str(block.value) for block in app.subheader]
    assert any("minute plan" in h for h in headers)
    assert any("Are your windows actually safer" in h for h in headers)


def test_quick_entry_adds_round(app):
    app.run()
    app.text_input(key="quick_value").set_value("1.75").run()
    assert not app.exception
    frame = ds.load_rounds()
    assert len(frame) == 61
    assert frame.iloc[-1]["multiplier"] == pytest.approx(1.75)
    assert frame.iloc[-1]["source"] == "manual"
    assert frame.iloc[-1]["timestamp"] != ""


def test_quick_entry_rejects_bad_value(app):
    app.run()
    app.text_input(key="quick_value").set_value("0.5").run()
    assert len(ds.load_rounds()) == 60


def test_import_merge_button(app):
    app.run()
    csv_text = (
        "timestamp,multiplier,source\n"
        "2025-03-01T14:00:00+02:00,2.50,manual\n"
        "2025-03-01T14:00:30+02:00,3.50,manual\n"
    )
    incoming, errors = ds.read_csv_text(csv_text)
    assert errors == []
    app.session_state["pending_import"] = {"df": incoming, "errors": errors,
                                           "name": "backup.csv"}
    app.session_state["pending_file"] = ("backup.csv", len(csv_text))
    app.run()
    button_by_label(app, "Merge imported rows").click().run()
    assert not app.exception
    frame = ds.load_rounds()
    assert len(frame) == 62
    merged = frame[frame["timestamp"] != ""]
    assert len(merged) == 2


def test_seed_restore_button(app):
    app.run()
    app.text_input(key="quick_value").set_value("9.99").run()
    assert len(ds.load_rounds()) == 61
    button_by_label(app, "Restore seed data").click().run()
    assert not app.exception
    assert len(ds.load_rounds()) == 61  # not restored yet: confirm step pending
    button_by_label(app, "Yes, restore seed data").click().run()
    assert not app.exception
    frame = ds.load_rounds()
    assert len(frame) == 60
    assert not (frame["multiplier"] == 9.99).any()


def test_clear_all_requires_confirmation(app):
    app.run()
    button_by_label(app, "Clear all data").click().run()
    assert not app.exception
    assert len(ds.load_rounds()) == 60  # not cleared yet
    button_by_label(app, "Yes, clear everything").click().run()
    assert not app.exception
    assert len(ds.load_rounds()) == 0


def test_data_tab_reports_seed_rounds_as_unusable(app):
    app.run()
    warnings = "\n".join(str(block.value) for block in app.warning)
    assert "timestamp" in warnings.lower()
    headers = [str(block.value) for block in app.subheader]
    assert any("Feeding the predictor" in h for h in headers)
    assert any("Enrichment CSV" in h for h in headers)


def test_responsible_tab_shows_contacts(app):
    app.run()
    joined = "\n".join(str(b.value) for b in app.markdown)
    assert "1144" in joined
    assert "Gambling Authority" in joined


def test_dev_credit_in_sidebar(app):
    app.run()
    assert not app.exception
    joined = "\n".join(str(b.value) for b in app.sidebar.markdown)
    assert "Developed by" in joined
    assert "Karabo Kosi" in joined
    assert "+267 78966834" in joined
    assert "karaboemma25@gmail.com" in joined
    assert "https://whoiskay.vercel.app/" in joined
    assert "linkedin.com/in/karabo-kosi-534501380" in joined
    assert "youtube.com/@whoiskay404" in joined
    assert "instagram.com/kaysantanaax" in joined
    assert "tiktok.com/@karabo_kosi" in joined
    assert "<svg" in joined


def test_rules_section_moved_to_sidebar(app):
    app.run()
    assert not app.exception
    sidebar = "\n".join(str(b.value) for b in app.sidebar.markdown)
    main_md = "\n".join(str(b.value) for b in app.main.markdown)
    assert "The rhythm" in sidebar
    assert "doubling ladder" in sidebar
    assert "Risk" in sidebar
    assert "The rhythm" not in main_md
    assert "doubling ladder" not in main_md
    sidebar_labels = [str(e.label) for e in app.sidebar.expander]
    main_labels = [str(e.label) for e in app.main.expander]
    assert "Rules & stake strategy" in sidebar_labels
    assert "Rules & stake strategy" not in main_labels


def test_apply_table_edits_accepts_valid_state(app):
    app.run()
    assert not app.exception
    button_by_label(app, "Apply table edits").click().run()
    assert not app.exception
    # flash banner confirms the edit path ran without validation errors
    success = "\n".join(str(b.value) for b in app.success)
    assert "Table edits applied" in success
    assert len(ds.load_rounds()) == 60


def test_batch_add_and_stamp(app):
    app.run()
    app.session_state["batch_list"] = "1.40 1.60 2.10"
    button_by_label(app, "Add rounds").click().run()
    assert not app.exception
    frame = ds.load_rounds()
    assert len(frame) == 63
    new_rows = frame.tail(3)
    assert new_rows["timestamp"].map(ds.is_untimed).sum() == 0
    assert (new_rows["source"] == "manual").all()
