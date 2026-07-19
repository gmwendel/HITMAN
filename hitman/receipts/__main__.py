"""CLI: ``python -m hitman.receipts <model_dir> <testpoint_spec> ...``

A testpoint spec is ``path.root:tag:x,y,z,zen,az,t,E``. Example::

    python -m hitman.receipts training_runs/run7_production \\
        testpoints/e3MeV_zen000.root:e3MeV_zen000:0,0,0,0,1.5708,0,3 \\
        --store datagen/water_1M/store --baseline training_runs/run6_exact/receipts.json

Exit code: 0 if the worst threshold finding is pass/warn, 1 if any fail.
"""

import argparse
import sys

from hitman.receipts import schema, thresholds
from hitman.receipts.runner import run_model_dir


def _parse_testpoint(spec: str):
    path, tag, truth = spec.split(":")
    truth = tuple(float(v) for v in truth.split(","))
    if len(truth) != 7:
        raise ValueError(f"testpoint {tag}: truth needs 7 comma-separated values")
    return tag, path, truth


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hitman.receipts")
    ap.add_argument("model_dir")
    ap.add_argument("testpoints", nargs="+", help="path.root:tag:x,y,z,zen,az,t,E")
    ap.add_argument("--store", required=True, help="marginal-pool HitStore dir")
    ap.add_argument("--n-pool", type=int, default=4_000_000)
    ap.add_argument("--baseline", default=None, help="baseline receipts.json for regression gates")
    ap.add_argument("--out", default="receipts.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tps = [_parse_testpoint(s) for s in args.testpoints]
    receipts, out_path = run_model_dir(
        args.model_dir, tps, args.store, n_pool=args.n_pool, out_name=args.out, seed=args.seed
    )

    baseline = schema.load_receipts(args.baseline) if args.baseline else None
    findings = thresholds.evaluate(receipts, baseline=baseline)
    summ = thresholds.summary(findings)
    print("\nthreshold findings:")
    for f in findings:
        if f.status != thresholds.PASS:
            print(f"  [{f.status.upper():4s}] {f.testpoint} :: {f.metric} — {f.detail}")
    print(f"summary: {summ['counts']}  worst={summ['worst']}")
    return 1 if summ["worst"] == thresholds.FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
