"""Recalculate selected fixed-preset results from saved configuration summaries."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Binary": ([0.541010, 0.511233, 0.629993, 0.459233], [(55, 39, 194), (256, 3, 29)]),
    "Ternary": ([1.534508, 1.509456, 1.632267, 1.508175], [(16, 24, 248), (283, 0, 5)]),
}
DIMENSIONS = ("cell", "feature", "target", "predictor")
REFS = ("fixed12_Bayes", "fixed60_Bayes")
TOLERANCE = 1e-12


def load(name):
    with (ROOT / "results" / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return tuple(row[k] for k in DIMENSIONS)


def cost(row):
    return float(row["total_cost"])


def main():
    metrics = load("policy_metrics.csv")
    priors = load("training_reference_metrics.csv")
    result = {}
    for target, (expected_means, expected_counts) in EXPECTED.items():
        gated = [r for r in metrics if r["target"] == target and r["policy"] == "gated_Bayes"]
        prior = [r for r in priors if r["target"] == target and r["name"] == "training_prior_Bayes"]
        assert len(gated) == len(prior) == 288
        assert len({(*key(r), r["variant"]) for r in gated}) == 288
        assert {(*key(r), r["variant"]) for r in gated} == {(*key(r), r["variant"]) for r in prior}
        means = [round(mean(map(cost, gated)), 6)]
        paired = []
        for policy in REFS:
            fixed = [r for r in metrics if r["target"] == target and r["policy"] == policy]
            look = {key(r): cost(r) for r in fixed}
            assert len(look) == len(fixed) == 96 and look.keys() == {key(r) for r in gated}
            means.append(round(mean(look.values()), 6))
            delta = [cost(r) - look[key(r)] for r in gated]
            counts = tuple(sum(test(x) for x in delta) for test in (
                lambda x: x < -TOLERANCE, lambda x: abs(x) <= TOLERANCE, lambda x: x > TOLERANCE))
            paired.append(counts)
        means.append(round(mean(map(cost, prior)), 6))
        assert means == expected_means and paired == expected_counts
        result[target] = {
            "mean_total_cost": dict(zip(("gated_Bayes", *REFS, "training_prior_Bayes"), means)),
            "paired_lower_tied_higher": dict(zip(REFS, paired)),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
