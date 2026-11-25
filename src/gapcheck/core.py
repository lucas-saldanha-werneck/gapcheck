"""Lag between two groups (e.g. closed vs open-weight) on one benchmark.

Model: score = a + b * t for each group (t in years). The lag at score level s
is the horizontal distance between the two fitted lines, in months.
Stdlib only. Bootstrap is seeded, so results repeat.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Fit:
    a: float
    b: float
    r2: float
    n: int
    t_min: float
    t_max: float

    def predict(self, t: float) -> float:
        return self.a + self.b * t

    def time_at(self, s: float) -> float:
        return (s - self.a) / self.b


@dataclass
class Result:
    level: float
    lag_months: float
    ci_low: float
    ci_high: float
    fits: dict
    warnings: list = field(default_factory=list)


def to_years(d: date) -> float:
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def read_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        lines = (ln for ln in f if not ln.lstrip().startswith("#"))
        for r in csv.DictReader(lines):
            rows.append((r["group"].strip(), to_years(date.fromisoformat(r["date"])), float(r["score"])))
    return rows


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def fit_line(points) -> Fit:
    n = len(points)
    if n < 2:
        raise ValueError("need at least 2 points per group")
    mx = sum(t for t, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((t - mx) ** 2 for t, _ in points)
    if sxx == 0:
        raise ValueError("all dates in a group are equal")
    b = sum((t - mx) * (y - my) for t, y in points) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for _, y in points)
    ss_res = sum((y - (a + b * t)) ** 2 for t, y in points)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    ts = [t for t, _ in points]
    return Fit(a, b, r2, n, min(ts), max(ts))


def _lag(fa: Fit, fb: Fit, s: float) -> float:
    """Months by which group B reaches level s after group A."""
    return (fb.time_at(s) - fa.time_at(s)) * 12


def estimate(rows, early: str, late: str, level: float, use_logit=False,
             boots: int = 2000, seed: int = 0) -> Result:
    """rows: (group, t_years, score). `early` is the group expected to lead."""
    f = (lambda x: logit(x)) if use_logit else (lambda x: x)
    pts = {g: [(t, f(s)) for gg, t, s in rows if gg == g] for g in (early, late)}
    lvl = f(level)
    fits = {g: fit_line(p) for g, p in pts.items()}
    if min(x.b for x in fits.values()) <= 0:
        raise ValueError("a group has a flat or falling trend; lag is not defined")
    lag = _lag(fits[early], fits[late], lvl)

    rng = random.Random(seed)
    sims = []
    for _ in range(boots):
        try:
            bf = {g: fit_line([rng.choice(p) for _ in p]) for g, p in pts.items()}
            if min(x.b for x in bf.values()) <= 0:
                continue
            sims.append(_lag(bf[early], bf[late], lvl))
        except ValueError:
            continue
    sims.sort()
    if len(sims) < boots // 2:
        lo = hi = float("nan")
    else:
        lo = sims[int(0.025 * len(sims))]
        hi = sims[min(len(sims) - 1, int(0.975 * len(sims)))]

    w = []
    for g, x in fits.items():
        if x.n < 6:
            w.append(f"{g}: only {x.n} models; interval is unreliable")
        if x.r2 < 0.8:
            w.append(f"{g}: straight line fits poorly (R2={x.r2:.2f})")
        if not (x.predict(x.t_min) <= lvl <= x.predict(x.t_max)) and not (x.predict(x.t_max) <= lvl <= x.predict(x.t_min)):
            w.append(f"{g}: level {level} is outside its observed score range (extrapolation)")
    ratio = fits[early].b / fits[late].b
    if not 0.7 <= ratio <= 1.3:
        w.append(f"slopes differ (ratio {ratio:.2f}); the lag depends on the level you pick")
    if math.isnan(lo):
        w.append("bootstrap failed on most resamples; no interval")
    return Result(level, lag, lo, hi, fits, w)


def report(res: Result, early: str, late: str) -> str:
    out = [f"Level {res.level}: {late} trails {early} by {res.lag_months:.1f} months "
           f"(95% bootstrap interval {res.ci_low:.1f} to {res.ci_high:.1f})"]
    for g, x in res.fits.items():
        out.append(f"  {g}: n={x.n}, slope={x.b:.3f}/year, R2={x.r2:.2f}")
    for w in res.warnings:
        out.append(f"  WARNING: {w}")
    return "\n".join(out)
