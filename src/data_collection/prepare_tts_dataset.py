import argparse
import glob
import json
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a Coqui-compatible VITS dataset")
    parser.add_argument("--config", type=str, default="../../config/tts_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    teacher_dir = config["data"]["teacher_train_audio_dir"]
    dataset_dir = config["data"]["student_dataset_dir"]
    wavs_dir = os.path.join(dataset_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    json_files = sorted(glob.glob(os.path.join(teacher_dir, "*.json")))
    written = 0

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        for json_path in json_files:
            wav_path = json_path.replace(".json", ".wav")
            if not os.path.exists(wav_path):
                continue

            with open(json_path, "r", encoding="utf-8") as handle:
                sample = json.load(handle)

            basename = os.path.basename(wav_path)
            dataset_wav_path = os.path.join(wavs_dir, basename)
            if not os.path.exists(dataset_wav_path):
                try:
                    os.link(wav_path, dataset_wav_path)
                except OSError:
                    import shutil

                    shutil.copy2(wav_path, dataset_wav_path)

            text = (sample.get("text") or "").replace("|", " ").strip()
            if not text:
                continue
            stem = os.path.splitext(basename)[0]
            metadata_file.write(f"{stem}|{text}|{text}\n")
            written += 1

    print(f"Prepared {written} TTS training samples in {dataset_dir}")


if __name__ == "__main__":
    main()
