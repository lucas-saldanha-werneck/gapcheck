import argparse
import sys

from .core import estimate, read_rows, report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gapcheck", description="Lag between two model groups on a benchmark.")
    p.add_argument("csv", help="columns: model,group,date,score")
    p.add_argument("--early", default="closed")
    p.add_argument("--late", default="open")
    p.add_argument("--level", type=float, nargs="+", required=True, help="score level(s), same scale as the CSV")
    p.add_argument("--logit", action="store_true", help="fit on logit scale (scores must be 0-1)")
    p.add_argument("--boots", type=int, default=2000)
    a = p.parse_args(argv)
    rows = read_rows(a.csv)
    for lv in a.level:
        try:
            print(report(estimate(rows, a.early, a.late, lv, a.logit, a.boots), a.early, a.late))
        except ValueError as e:
            print(f"Level {lv}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
