# scripts/ — Week-1 data feasibility

Two steps to learn your **real usable N** (the number that decides scope).

## 1. Build the titles list  ✅ already run
```bash
python build_titles.py
```
Flattens all 1,276 ScriptBase archives into `../data/titles.csv`
(columns: `raw_name, title, year, imdb_score, meta_score, genres`).
Already generated — 1,276 films, 100% with a year.

## 2. Run the join-yield probe  ← your turn (needs your TMDB key)
```bash
pip install requests python-dotenv
echo "TMDB_API_KEY=your_key_here" > .env      # in this scripts/ folder
python join_yield_probe.py ../data/titles.csv
```
For each film it looks up TMDB, pulls budget / revenue / rating / vote count, flags whether
it has **usable financials AND reception**, computes ROI, and prints a yield summary +
`join_yield_<timestamp>.csv`.

Run with **no argument** first (`python join_yield_probe.py`) to smoke-test your key against the
built-in case-study panel (Ratatouille, Coco, Wish, the Toy Story sequels).

### Reading the result
- **≥ ~500 usable** → on track; proceed.
- **< ~500 usable** → the proposal's Week-1 "behind schedule" trigger — widen the corpus
  (Script Slug / 8FLiX for newer + animated titles) or rescope before building further.

> Heads up: ScriptBase already gives you `imdb_score` (a reception signal), but **not budget or
> revenue** — that's exactly what the TMDB join adds, and why the probe is the real feasibility test.
