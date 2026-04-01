from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from src.vits_pipeline.common import PROJECT_ROOT, ensure_dir, load_config, normalize_text, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a raw dataset for Coqui VITS training")
    parser.add_argument("--config", type=str, default="config/tts_config.yaml")
    return parser.parse_args()


def iter_manifest_rows(dataset_cfg: dict, source_root: Path, manifest_path: Path):
    manifest_format = dataset_cfg["manifest_format"].lower()
    delimiter = dataset_cfg.get("delimiter", "|")

    if manifest_format == "jsonl":
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if manifest_format in {"csv", "tsv"}:
        csv_delimiter = "," if manifest_format == "csv" else "\t"
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=csv_delimiter)
            if not reader.fieldnames:
                raise ValueError(f"Manifest {manifest_path} is missing a header row")
            for row in reader:
                yield row
        return

    if manifest_format == "ljspeech":
        with manifest_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(delimiter, maxsplit=2)
                if len(parts) < 2:
                    continue
                yield {
                    dataset_cfg["audio_column"]: parts[0],
                    dataset_cfg["text_column"]: parts[1],
                }
        return

    raise ValueError(f"Unsupported manifest format: {manifest_format}")


def resolve_audio_path(audio_ref: str, source_root: Path) -> Path | None:
    raw = Path(audio_ref)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                source_root / raw,
                source_root / "wavs" / raw,
                source_root / f"{audio_ref}.wav",
                source_root / "wavs" / f"{audio_ref}.wav",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def choose_eval_count(total_samples: int, ratio: float, max_samples: int) -> int:
    if total_samples < 2:
        raise ValueError("Need at least 2 valid samples after filtering to create train and eval splits")

    target = int(total_samples * ratio)
    if ratio > 0 and target == 0:
        target = 1
    if max_samples > 0:
        target = min(target, max_samples)
    target = max(1, target)
    return min(target, total_samples - 1)


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['basename']}|{row['text']}|{row['text']}\n")


def main() -> None:
    import librosa
    import soundfile as sf

    args = parse_args()
    config, config_path = load_config(args.config)
    base_dir = PROJECT_ROOT

    dataset_cfg = config["dataset"]
    project_cfg = config.get("project", {})

    source_root = resolve_path(dataset_cfg["source_root"], base_dir)
    manifest_path = resolve_path(dataset_cfg["manifest_path"], base_dir)
    processed_dir = resolve_path(dataset_cfg["processed_dir"], base_dir)
    wavs_dir = processed_dir / "wavs"

    if not source_root.exists():
        raise FileNotFoundError(f"Dataset source_root does not exist: {source_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    ensure_dir(wavs_dir)

    audio_column = dataset_cfg["audio_column"]
    text_column = dataset_cfg["text_column"]
    speaker_name = dataset_cfg.get("default_speaker_name") or "speaker"
    target_sample_rate = int(dataset_cfg["target_sample_rate"])

    kept_rows = []
    filtered_missing_audio = 0
    filtered_text = 0
    filtered_duration = 0
    filtered_audio = 0

    for row_index, row in enumerate(iter_manifest_rows(dataset_cfg, source_root, manifest_path), start=1):
        audio_ref = str(row.get(audio_column, "")).strip()
        text = str(row.get(text_column, "")).strip()

        if dataset_cfg.get("normalize_whitespace", True):
            text = normalize_text(text)

        if not audio_ref:
            filtered_missing_audio += 1
            continue
        if not text or len(text) < int(dataset_cfg["min_text_chars"]) or len(text) > int(dataset_cfg["max_text_chars"]):
            filtered_text += 1
            continue

        source_audio_path = resolve_audio_path(audio_ref, source_root)
        if source_audio_path is None:
            filtered_missing_audio += 1
            continue

        try:
            info = sf.info(str(source_audio_path))
        except RuntimeError:
            filtered_audio += 1
            continue

        if info.duration < float(dataset_cfg["min_audio_seconds"]) or info.duration > float(dataset_cfg["max_audio_seconds"]):
            filtered_duration += 1
            continue

        try:
            audio, _ = librosa.load(str(source_audio_path), sr=target_sample_rate, mono=True)
        except Exception:
            filtered_audio += 1
            continue

        if audio.size == 0:
            filtered_audio += 1
            continue

        basename = f"sample_{len(kept_rows):06d}"
        output_audio_path = wavs_dir / f"{basename}.wav"
        sf.write(str(output_audio_path), audio, target_sample_rate, subtype="PCM_16")

        kept_rows.append(
            {
                "basename": basename,
                "text": text,
                "speaker_name": speaker_name,
                "duration_seconds": round(len(audio) / target_sample_rate, 3),
                "source_audio": str(source_audio_path),
                "row_index": row_index,
            }
        )

    eval_count = choose_eval_count(
        total_samples=len(kept_rows),
        ratio=float(dataset_cfg["eval_split_ratio"]),
        max_samples=int(dataset_cfg["eval_max_samples"]),
    )

    rng = random.Random(int(project_cfg.get("seed", 17)))
    shuffled_rows = list(kept_rows)
    rng.shuffle(shuffled_rows)

    eval_rows = shuffled_rows[:eval_count]
    train_rows = shuffled_rows[eval_count:]

    write_manifest(processed_dir / "metadata_train.csv", train_rows)
    write_manifest(processed_dir / "metadata_eval.csv", eval_rows)

    with (processed_dir / "eval_prompts.txt").open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(f"{row['text']}\n")

    summary = {
        "project": config.get("project", {}).get("name"),
        "source_root": str(source_root),
        "manifest_path": str(manifest_path),
        "processed_dir": str(processed_dir),
        "target_sample_rate": target_sample_rate,
        "speaker_name": speaker_name,
        "language": dataset_cfg.get("language"),
        "counts": {
            "kept": len(kept_rows),
            "train": len(train_rows),
            "eval": len(eval_rows),
            "filtered_missing_audio": filtered_missing_audio,
            "filtered_text": filtered_text,
            "filtered_duration": filtered_duration,
            "filtered_audio": filtered_audio,
        },
        "hours": {
            "total": round(sum(row["duration_seconds"] for row in kept_rows) / 3600, 3),
            "train": round(sum(row["duration_seconds"] for row in train_rows) / 3600, 3),
            "eval": round(sum(row["duration_seconds"] for row in eval_rows) / 3600, 3),
        },
    }
    write_json(processed_dir / "dataset_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
