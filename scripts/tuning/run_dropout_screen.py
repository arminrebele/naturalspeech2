#!/usr/bin/env python3
"""Screen one module's dropout sites, then emit paired keep/reject verdicts.

For the given module it runs the rolling baseline plus one trial per site (that
site at --rate, every other site at its rolling-baseline value), aggregates each
treatment against the baseline (scripts/tuning/aggregate.py), and — with
--commit — folds the survivors into the rolling-baseline JSON for the next
module. Run the modules in aggregate.ORDER, reviewing verdicts between each.

Each trial is a fresh `scripts/train.py +experiment=dropout_trial` subprocess
(fresh CUDA context + compile cache, no cross-run state). ALL dropout sites are
passed explicitly every run, so the trial is fully determined regardless of the
(paper) dropout defaults in config/model/base.yaml — the tuning baseline is zero
dropout, and the search tests adding a low rate at one site at a time.

This is the per-site screen + gate only. The combo / rate-refine stage on the
survivors is a follow-up; for now the gate just reports which sites cleared.

  python scripts/tuning/run_dropout_screen.py --module duration_predictor --commit
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import ALL_SITES, KEEP_THRESHOLD, MODULES, evaluate_site

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "scripts" / "train.py"


def load_baseline(path: Path) -> dict:
    """Rolling baseline {site: rate}; all sites zero if the file doesn't exist yet."""
    if path.exists():
        baseline = json.loads(path.read_text())
        # Surface drift between the saved baseline and the current site manifest loudly.
        assert set(baseline) == set(ALL_SITES), (
            f"baseline sites {sorted(baseline)} != manifest sites {sorted(ALL_SITES)}"
        )
        return baseline
    return {site: 0.0 for site in ALL_SITES}


def trial_overrides(dropout: dict, out_path: Path, run_name: str, group: str, extra: list) -> list:
    return [
        "+experiment=dropout_trial",
        f"setup.dropout_trial_out={out_path}",
        f"wandb.run_name={run_name}",
        f"wandb.group={group}",
        *[f"model.{site}={val}" for site, val in dropout.items()],
        *extra,
    ]


def run_trial(dropout: dict, out_path: Path, run_name: str, group: str, extra: list, dry_run: bool):
    overrides = trial_overrides(dropout, out_path, run_name, group, extra)
    cmd = [sys.executable, str(TRAIN), *overrides]
    print(f"[trial] {run_name}")
    print(f"        {' '.join(overrides)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO)


def screen_module(module: str, baseline: dict, rate: float, group: str,
                  workdir: Path, extra: list, dry_run: bool) -> dict:
    spec = MODULES[module]
    if not dry_run:
        workdir.mkdir(parents=True, exist_ok=True)

    base_out = workdir / "baseline.jsonl"
    run_trial(baseline, base_out, f"{module}__baseline", group, extra, dry_run)

    results = {}
    for site in spec["sites"]:
        treat = dict(baseline)
        treat[site] = rate
        out = workdir / f"site__{site.replace('.', '_')}.jsonl"
        run_trial(treat, out, f"{module}__{site.split('.')[-1]}", group, extra, dry_run)
        if not dry_run:
            results[site] = evaluate_site(base_out, out, spec["primary"], spec["guard"])
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True, choices=list(MODULES))
    ap.add_argument("--rate", type=float, default=0.1)
    ap.add_argument("--baseline-file", type=Path, default=REPO / "research" / "dropout_rolling_baseline.json")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--commit", action="store_true", help="fold survivors into the rolling baseline")
    ap.add_argument("--dry-run", action="store_true", help="print trial commands without running or aggregating")
    ap.add_argument("--override", nargs="*", default=[], help="extra Hydra overrides for every trial")
    args = ap.parse_args()

    baseline = load_baseline(args.baseline_file)
    group = f"dropout-screen-{args.module}"
    workdir = args.workdir or (REPO / "research" / "dropout_screen" / args.module)

    results = screen_module(args.module, baseline, args.rate, group, workdir, args.override, args.dry_run)
    if args.dry_run:
        print("\n[dry-run] no trials launched, no aggregation.")
        return

    print("\n" + "=" * 72)
    print(f"VERDICTS — {args.module}  (rate {args.rate}, KEEP iff z <= {KEEP_THRESHOLD})")
    print("=" * 72)
    survivors = []
    for site, r in results.items():
        print(f"  {site:<46} {r['decision']:<14} z={r['z']:+.2f}  "
              f"(Δ={r['mean_delta']:+.4f} ± {r['std_delta']:.4f}, n={r['n']})")
        for term, gz in r["regressions"]:
            print(f"      ! consumer regression: {term} z={gz:+.2f}  (manual call)")
        if r["decision"].startswith("KEEP"):
            survivors.append(site)

    if survivors:
        print(f"\n  → {len(survivors)} site(s) cleared → run combos / rate-refine on: {survivors}")
    else:
        print("\n  → no site cleared → discard module (all its sites stay 0).")

    if args.commit:
        for site in survivors:
            baseline[site] = args.rate
        args.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_file.write_text(json.dumps(baseline, indent=2))
        print(f"\n  committed {len(survivors)} survivor(s) at rate {args.rate} → {args.baseline_file}")


if __name__ == "__main__":
    main()
