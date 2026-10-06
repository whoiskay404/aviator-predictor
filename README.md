# Aviator Predictor

An offline Python/Streamlit predictor and data logger for Aviator-style
crash-game rounds. It exists to make one call - stake or skip - from your
logged rounds plus the game's verified 97% RTP maths, and it tells you
honestly when it doesn't have enough data yet.

**No betting automation. No casino connection. No guaranteed predictions.**
Every round of a provably fair crash game is independent; this tool gives
probability estimates, never certainties.

-----> https://aviator-odds.streamlit.app/

## Features

### Predict tab (the point of the app)

The main page is intentionally sparse - a slim header (title + a status chip
like `60 rounds · THEORY mode`), one honest disclaimer line, then the hero:

- **PREDICT** - one big circular button. Press it and the result circle
  shows the suggested odd, whether the minute is a **Golden window** or
  **Outside window**, plus `P(reach) … · pressed HH:MM:SS`; a quiet **Reset**
  pill clears it for the next round. Below the hero: the round selector
  (Auto from the clock, or any minute 00-59) and a 4-pill strip showing the
  next minutes with their odds.
- **Always an odd** - every minute gets a suggestion in **1.00-2.00x**:
  minutes in your golden windows and data-validated minutes get a data-based
  odd (usually the cautious 1.00-1.45x band), and minutes outside the golden
  windows get an advanced odd taken from the enrichment CSV, clearly flagged
  **OUTSIDE GOLDEN WINDOW** instead of being skipped. The same odd never
  shows twice in a row - if a repeat would appear (presses, strip pills or
  plan rows) it is nudged one 0.05 grid step inside its band and the Details
  card says so.
- **Everything else sits behind collapsed expanders** so the page stays
  uncluttered:
  - **Details** - the reasoning card (alert, confidence, basis, odd source,
    metric cards, comparison tables), the data-quality metrics
    (timestamped / in-window / needed-for-data-backed / mode) and the
    progress bar.
  - **Minute plan & windows** - the 60-minute plan, window health tables
    with Fisher exact + Mann-Whitney U verdicts (p hidden below n = 30),
    editable golden windows (defaults 03-04, 20-22, 29-31, 40-42, 45-47,
    50-52, 57-59), data-favoured minutes gated by Wilson intervals, and the
    cash-out target table (every target -3% EV, 97% RTP).
  - **Sources & safety** - Spribe's RTP 97%, the published provably fair
    formula P(reach x) = 0.97/x with the FairPlay Audit verification, and
    the responsible-gambling contacts (Botswana 1144 etc.).
- **Sidebar** - a collapsed **Status** expander (rounds stored, data
  file path with copy button, last refresh time), the developer credit:
  *Developed by Karabo Kosi* with icon links (phone, Gmail, website,
  LinkedIn, YouTube, Instagram, TikTok) and the phone + email in text, and a
  collapsed **Rules & stake strategy** expander right under the contacts -
  the play rhythm (press PREDICT, sit out the next round, then enter every
  second round: 1, 3, 5 …), the risk ↔ target rule (risk low → target high,
  risk high → target low) and the doubling stake ladder (P10 → P20 → double
  after each loss until a win, then back to P10), with the honest note that
  the ladder never changes the -3% edge.
- **Mobile friendly** - below 760px every column row stacks vertically
  (hero, metrics, editors), the PREDICT and odd circles shrink to 200px
  (150px in short landscape viewports), the minute strip wraps and the
  sidebar collapses into the hamburger menu.

### Data tab (feeding the predictor)

Quality summary at the top, then three tidy sections:

- **Add rounds** (open by default) - fast live entry (type the multiplier,
  Enter; timestamp fills in automatically in Africa/Gaborone, UTC+02:00) and
  the bulk paste form (start time + seconds-per-round interval, or stamp
  existing untimed rows).
- **Import / Export** - CSV merge upload with duplicate skipping, export and
  blank template downloads, plus the **Enrichment CSV** block (upload or
  reload the bundled `csv_data_odds/aviator_rounds.csv` of odds with
  estimated timestamps; it drives the outside-golden-window odds. Bundled
  file: 334 rounds read by eye from screenshots, times estimated at ~22 s/round
  (±1 min, gaps between segments)).
- **Danger zone** (red-outlined buttons, each behind a confirm step) -
  **Restore seed data** and **Clear all data**.
- **In-app editor** - `st.data_editor` over the logged rounds (Apply
  validates every row: multiplier ≥ 1.00, valid source).

## Requirements

- Python 3.11+ (developed and tested on Python 3.12, Windows)

## Windows setup

```bat
cd path\to\aviator_odds
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

If you prefer not to activate the venv:

```bat
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run app.py
```

## Data

Rounds are stored in `data/rounds.csv` with columns
`timestamp,multiplier,source` (sources: `manual`, `import`, `seed`).
Rows without a timestamp (the bundled 60-round seed sample) are excluded from
time-based analysis until stamped.

- Set the `AVIATOR_CSV` environment variable to use a different file path.
- Set `AVIATOR_NO_AUTOREFRESH=1` to disable the live sidebar/timer auto-refresh.
- On first run the seed file is created automatically; **Restore seed data** and
  **Clear all data** are on the Data tab.

### CSV format

```csv
timestamp,multiplier,source
2025-03-01T14:05:00+02:00,1.50,manual
,83.73,seed
```

Duplicates (same timestamp + multiplier) are skipped on merge; untimed rows are
merged per-multiplier with a max-count rule.

## Tests

```bat
venv\Scripts\python.exe -m pytest
```

Covers the statistics functions against reference values (Wilson intervals,
percentiles, Fisher exact, Mann-Whitney, chi-square, BH/Bonferroni correction),
the prediction engine, CSV parsing/merging/seed data, and the app itself via
`streamlit.testing.v1.AppTest`.

## Project layout

```
app.py          Streamlit UI (2 tabs: Predict, Data)
stats.py        Statistical helpers + prediction engine (pure functions)
datastore.py    CSV parsing, merging, seed data, time helpers
data/rounds.csv Round storage (created on first run)
tests/          pytest suite
```

## Honest limits

- Confidence intervals widen and verdicts become unreliable on small samples -
  the app says **Not enough data** rather than guessing.
- Scanning 60 minutes at once manufactures false positives; corrected p-values
  (BH, Bonferroni) account for that.
- No staking system beats a negative expected value. Martingale loses faster
  when it loses; flat staking bleeds at the same theoretical rate.
