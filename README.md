# gapcheck

Independent demo, not affiliated with SecureBio.

Given dated benchmark scores for two groups of models (for example closed-weight and open-weight), `gapcheck` estimates how many months the second group trails the first at a chosen score level. It gives a bootstrap interval and warns when the straight-line assumption looks shaky.

I built it after reading a job post about estimating the gap between closed and open-weight models. It is a small sketch of one way to do that, with the assumption checks printed next to the number.

## Run it

Python 3.10 or newer. No dependencies.

```
pip install -e .
gapcheck examples/example_scores.csv --level 0.4 0.5 0.6
```

Input is a CSV with the columns `model,group,date,score`. Lines starting with `#` are skipped. Use `--early` and `--late` to name the groups (defaults: `closed`, `open`). `--logit` fits on the logit scale when scores are between 0 and 1.

The file in `examples/` is **invented example data**, not real benchmark results.

Output on that file:

```
Level 0.4: open trails closed by 12.0 months (95% bootstrap interval 10.8 to 12.9)
Level 0.5: open trails closed by 10.9 months (95% bootstrap interval 9.8 to 11.9)
Level 0.6: open trails closed by 9.7 months (95% bootstrap interval 8.3 to 11.3)
```

The lag shrinks as the level goes up, because the two fitted slopes differ (0.185 and 0.225 per year). One "gap" number would hide that. This is why the tool takes several levels.

## How it works

1. Fit `score = a + b * year` for each group by least squares.
2. The lag at level `s` is the time distance between the two lines at `s`, in months.
3. Resample each group's models with replacement 2000 times, refit, and take the 2.5% and 97.5% points of the lags.
4. Print warnings: fewer than 6 models in a group, R² under 0.8, a level outside the observed score range, or slopes whose ratio is outside 0.7 to 1.3.

## What was tested

- 14 offline tests (`pytest`), run in CI on every push.
- Exact line fit on a toy case.
- A known lag (8 months) is recovered exactly with no noise, and falls inside the interval with noise (one seed).
- The seeded bootstrap gives the same interval twice.
- Each warning fires on a case built to trigger it.
- Flat trends and groups with fewer than 2 points are rejected.
- The CSV reader skips comment lines; the command line runs on the example file and returns an error code on bad input.

## What was not tested

- Real benchmark data. Everything above uses invented or synthetic numbers.
- Whether a straight line is a good model for real score trends. Scores saturate near 1, which is what `--logit` is for, but I have not checked it on real data.
- The coverage of the interval. I tested one seed, not how often the 95% interval contains the truth.
- Models with several benchmarks, correlated errors, or changes in what a benchmark measures over time.
- Anything about biology, safety evaluation, or how any organisation builds its own index.
