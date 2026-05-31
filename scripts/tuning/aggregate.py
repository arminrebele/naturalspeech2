#!/usr/bin/env python3
"""Aggregate dropout-trial JSONL dumps into keep/reject verdicts.

Each trial writes per-eval held-out losses (see scripts/train.py
`_append_dropout_trial_eval`): one JSON line per eval step holding the per-term
dev and test losses. A site's effect is judged by a PAIRED effect size against
the rolling baseline run:

    at each shared eval step:  delta = treatment_metric - baseline_metric
    z = mean(delta) / std(delta)     over the converged-tail window

Pairing cancels the common downward trend AND common-mode batch difficulty, so
std(delta) is the genuine differential noise rather than the trend (each eval is
a full deterministic held-out pass, so within-run spread is almost entirely the
trend — using it directly would make the test nearly blind). It is an effect size
(NOT divided by sqrt(n)): consecutive deltas are autocorrelated and there is one
run per config, so this is a clear-separation heuristic, not a calibrated p-value.

Convention: delta > 0 means the treatment is WORSE (higher held-out loss).

Decision (underfitting prior → burden of proof is on KEEPING dropout):
    KEEP a site only if z <= keep_threshold (-2): a clear held-out benefit.
    Otherwise REJECT.
    Backbone modules additionally flag any consumer term that regresses
    (guard z >= guard_threshold, +2) even when the primary improves — the
    teacher-forced diffusion metric is blind to predictor degradation that only
    bites at inference, so a regressed consumer is surfaced for a manual call.

The per-module metric (`primary`) and the backbone consumer `guard` terms live in
MODULES below; the orchestrator (run_dropout_screen.py) imports them too.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean, pstdev


# Per-module decision metric. `sites` are the model-config dropout dot-paths to
# screen; `primary` terms are summed into the headline metric; `guard` terms are
# each checked individually for a regression (backbones only).
#
#   heads (DP/PP/ALN)  -> own loss term(s); they are teacher-forced out of the
#                         diffusion path, so the diffusion loss can't see them.
#   DIFF               -> the dedicated diffusion group (data+score+ce_rvq).
#   backbones (SPE/PE) -> diffusion group as headline + a consumer guard on the
#                         predictor/aligner terms they feed (teacher forcing hides
#                         the inference-time cost of regressing those).
MODULES = {
    "duration_predictor": {
        "sites": [
            "duration_predictor.conv_dropout",
            "duration_predictor.attn_weights_dropout",
            "duration_predictor.attn_out_dropout",
        ],
        "primary": ["duration_predictor_loss"],
        "guard": [],
    },
    "pitch_predictor": {
        "sites": [
            "pitch_predictor.conv_dropout",
            "pitch_predictor.attn_weights_dropout",
            "pitch_predictor.attn_out_dropout",
        ],
        "primary": ["pitch_predictor_loss"],
        "guard": ["pitch_voicing_loss"],
    },
    "aligner": {
        "sites": ["aligner.dropout"],
        "primary": ["forward_sum_loss", "bin_loss"],
        "guard": [],
    },
    "diffusion_model": {
        "sites": [
            "diffusion_model.attn_weights_dropout",
            "diffusion_model.attn_out_dropout",
            "diffusion_model.wavenet_attn_weights_dropout",
            "diffusion_model.wavenet_attn_out_dropout",
            "diffusion_model.wavenet_gate_dropout",
        ],
        "primary": ["data_loss", "score_loss", "ce_rvq_loss"],
        "guard": [],
    },
    "speech_prompt_encoder": {
        "sites": [
            "speech_prompt_encoder.conv_dropout",
            "speech_prompt_encoder.attn_weights_dropout",
            "speech_prompt_encoder.attn_out_dropout",
        ],
        "primary": ["data_loss", "score_loss", "ce_rvq_loss"],
        "guard": ["duration_predictor_loss", "pitch_predictor_loss"],
    },
    "phoneme_encoder": {
        "sites": [
            "phoneme_encoder.conv_dropout",
            "phoneme_encoder.attn_weights_dropout",
            "phoneme_encoder.attn_out_dropout",
        ],
        "primary": ["data_loss", "score_loss", "ce_rvq_loss"],
        "guard": ["duration_predictor_loss", "pitch_predictor_loss", "forward_sum_loss", "bin_loss"],
    },
}

# Greedy between-module order: independent heads first, then the diffusion
# backbone, then the shared encoders (SPE feeds DP/PP/DIFF; PE feeds everything).
ORDER = [
    "duration_predictor", "pitch_predictor", "aligner",
    "diffusion_model", "speech_prompt_encoder", "phoneme_encoder",
]

ALL_SITES = [site for m in ORDER for site in MODULES[m]["sites"]]

KEEP_THRESHOLD = -2.0
GUARD_THRESHOLD = 2.0


def load_series(jsonl_path, terms) -> dict:
    """{step: combined_metric}, combined = sum over `terms` of (dev+test)/2.

    dev and test are averaged into one held-out number. The combination is linear
    and identical for every run, so it cancels in the paired difference — the
    average-vs-count-weighted choice is second order for the z."""
    series = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            value = sum((rec["dev"][t] + rec["test"][t]) / 2.0 for t in terms)
            series[rec["step"]] = value
    return series


def paired_z(base: dict, treat: dict):
    """Paired effect size on the step-aligned difference. Returns (z, mean, std, n)."""
    steps = sorted(set(base) & set(treat))
    if not steps:
        raise ValueError("no shared eval steps between baseline and treatment dumps")
    deltas = [treat[s] - base[s] for s in steps]
    mean_d = fmean(deltas)
    std_d = pstdev(deltas)
    if std_d == 0.0:
        # Perfectly consistent: zero effect -> 0; constant nonzero offset -> ±inf.
        z = 0.0 if mean_d == 0.0 else math.copysign(math.inf, mean_d)
    else:
        z = mean_d / std_d
    return z, mean_d, std_d, len(steps)


def evaluate_site(base_path, treat_path, primary_terms, guard_terms,
                  keep_threshold=KEEP_THRESHOLD, guard_threshold=GUARD_THRESHOLD) -> dict:
    """Paired verdict for one treatment dump vs the baseline dump."""
    z, mean_d, std_d, n = paired_z(
        load_series(base_path, primary_terms), load_series(treat_path, primary_terms),
    )
    guards = []
    for term in guard_terms:
        gz, *_ = paired_z(load_series(base_path, [term]), load_series(treat_path, [term]))
        guards.append((term, gz))
    regressions = [(t, gz) for t, gz in guards if gz >= guard_threshold]

    if z <= keep_threshold:
        decision = "KEEP_BUT_FLAG" if regressions else "KEEP"
    else:
        decision = "REJECT"
    return {
        "decision": decision, "z": z, "mean_delta": mean_d, "std_delta": std_d,
        "n": n, "guards": guards, "regressions": regressions,
    }


def main():
    ap = argparse.ArgumentParser(description="Aggregate a module's dropout-trial dumps into verdicts.")
    ap.add_argument("--module", required=True, choices=list(MODULES))
    ap.add_argument("--baseline", required=True, type=Path, help="baseline trial JSONL dump")
    ap.add_argument("--treatments", required=True, nargs="+", type=Path,
                    help="treatment JSONL dumps, one per screened site (filename stem labels the row)")
    args = ap.parse_args()

    spec = MODULES[args.module]
    print(f"== {args.module} (keep iff z <= {KEEP_THRESHOLD}; metric={'+'.join(spec['primary'])}) ==")
    for path in args.treatments:
        r = evaluate_site(args.baseline, path, spec["primary"], spec["guard"])
        print(f"  {path.stem:<40} {r['decision']:<14} z={r['z']:+.2f}  "
              f"(Δ={r['mean_delta']:+.4f} ± {r['std_delta']:.4f}, n={r['n']})")
        for term, gz in r["regressions"]:
            print(f"      ! consumer regression: {term} z={gz:+.2f}")


if __name__ == "__main__":
    main()
