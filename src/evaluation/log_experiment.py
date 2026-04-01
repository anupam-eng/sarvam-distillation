import argparse
import json
import os
from datetime import datetime, timezone

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Append an experiment report to a JSONL log")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    record = {
        "task": args.task,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_path": os.path.abspath(args.report),
        "metrics": {key: value for key, value in report.items() if isinstance(value, (int, float, str, bool))},
        "config": config,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(record["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
