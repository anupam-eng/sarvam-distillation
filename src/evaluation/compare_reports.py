import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two evaluation reports")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--lower_is_better", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.baseline, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(args.candidate, "r", encoding="utf-8") as handle:
        candidate = json.load(handle)

    baseline_value = baseline[args.metric]
    candidate_value = candidate[args.metric]
    delta = candidate_value - baseline_value

    if args.lower_is_better:
        improved = candidate_value < baseline_value
    else:
        improved = candidate_value > baseline_value

    print(
        json.dumps(
            {
                "metric": args.metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "improved": improved,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
