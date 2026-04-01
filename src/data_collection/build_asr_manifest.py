import argparse
import glob
import json
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Build ASR manifests from paired audio-text data")
    parser.add_argument("--config", type=str, default="../../config/asr_config.yaml")
    return parser.parse_args()


def collect_pairs(input_dir):
    records = []
    for json_path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        wav_path = json_path.replace(".json", ".wav")
        if not os.path.exists(wav_path):
            continue
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        text = (payload.get("text") or payload.get("transcript") or "").strip()
        if not text:
            continue
        records.append(
            {
                "audio_path": os.path.abspath(wav_path),
                "text": text,
                "language": payload.get("language"),
                "source": input_dir,
            }
        )
    return records


def extend_from_manifest(records, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            audio_path = sample["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(os.path.dirname(manifest_path), audio_path)
            records.append(
                {
                    "audio_path": os.path.abspath(audio_path),
                    "text": sample["text"],
                    "language": sample.get("language"),
                    "source": manifest_path,
                }
            )


def write_manifest(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_records = collect_pairs(config["data"]["teacher_train_dir"])
    eval_records = collect_pairs(config["data"]["teacher_eval_dir"])

    for manifest_path in config["data"].get("additional_train_manifests", []):
        extend_from_manifest(train_records, manifest_path)

    write_manifest(train_records, config["data"]["train_manifest_path"])
    write_manifest(eval_records, config["data"]["eval_manifest_path"])

    print(
        f"Built ASR manifests with {len(train_records)} train records and "
        f"{len(eval_records)} eval records."
    )


if __name__ == "__main__":
    main()
