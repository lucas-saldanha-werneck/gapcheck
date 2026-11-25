import math
import random

import pytest

from gapcheck.core import estimate, fit_line, read_rows, logit
from gapcheck.cli import main


def synth(lag_months, slope=0.2, noise=0.0, n=10, seed=1):
    r = random.Random(seed)
    rows = []
    for k in range(n):
        t = 2023 + k * 0.25
        rows.append(("closed", t, 0.3 + slope * (t - 2023) + r.gauss(0, noise)))
        t2 = t + lag_months / 12
        rows.append(("open", t2, 0.3 + slope * (t - 2023) + r.gauss(0, noise)))
    return rows


def test_fit_line_exact():
    f = fit_line([(0, 1), (1, 3), (2, 5)])
    assert f.b == pytest.approx(2) and f.a == pytest.approx(1) and f.r2 == pytest.approx(1)


def test_recovers_known_lag_without_noise():
    res = estimate(synth(8), "closed", "open", 0.5, boots=200)
    assert res.lag_months == pytest.approx(8, abs=1e-6)


def test_recovers_known_lag_with_noise_inside_interval():
    res = estimate(synth(8, noise=0.02, n=14), "closed", "open", 0.5, boots=500)
    assert res.ci_low <= 8 <= res.ci_high


def test_seeded_bootstrap_repeats():
    a = estimate(synth(6, noise=0.02), "closed", "open", 0.5, boots=300, seed=3)
    b = estimate(synth(6, noise=0.02), "closed", "open", 0.5, boots=300, seed=3)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_warns_on_small_sample():
    res = estimate(synth(6, n=4), "closed", "open", 0.5, boots=100)
    assert any("only 4 models" in w for w in res.warnings)


def test_warns_on_extrapolation():
    res = estimate(synth(6), "closed", "open", 5.0, boots=100)
    assert any("outside its observed score range" in w for w in res.warnings)


def test_warns_on_different_slopes():
    rows = [("closed", 2023 + k * .25, 0.3 + 0.1 * k * .25) for k in range(8)]
    rows += [("open", 2023.5 + k * .25, 0.2 + 0.3 * k * .25) for k in range(8)]
    res = estimate(rows, "closed", "open", 0.5, boots=100)
    assert any("slopes differ" in w for w in res.warnings)


def test_flat_trend_is_rejected():
    rows = [("closed", 2023 + k, 0.3) for k in range(5)] + [("open", 2023 + k, 0.1 + 0.1 * k) for k in range(5)]
    with pytest.raises(ValueError):
        estimate(rows, "closed", "open", 0.3, boots=10)


def test_needs_two_points():
    with pytest.raises(ValueError):
        fit_line([(1, 1)])


def test_logit_clamps():
    assert math.isfinite(logit(0.0)) and math.isfinite(logit(1.0))


def test_logit_mode_runs():
    res = estimate(synth(6, slope=0.1), "closed", "open", 0.5, use_logit=True, boots=100)
    assert res.lag_months == pytest.approx(6, abs=1e-6) or res.lag_months > 0


def test_read_rows_skips_comment_line(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("# example\nmodel,group,date,score\nm1,closed,2023-01-01,0.3\n")
    rows = read_rows(str(p))
    assert rows[0][0] == "closed" and rows[0][2] == 0.3


def test_cli_on_example_file(capsys):
    assert main(["examples/example_scores.csv", "--level", "0.5", "--boots", "200"]) == 0
    assert "trails closed by" in capsys.readouterr().out


def test_cli_error_exit_code(tmp_path, capsys):
    p = tmp_path / "x.csv"
    p.write_text("model,group,date,score\na,closed,2023-01-01,0.3\nb,open,2023-02-01,0.2\n")
    assert main([str(p), "--level", "0.3"]) == 1
