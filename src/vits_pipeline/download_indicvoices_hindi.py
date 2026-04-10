from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from src.vits_pipeline.common import PROJECT_ROOT, ensure_dir, load_config, normalize_text, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and filter Hindi IndicVoices data for VITS training")
    parser.add_argument("--config", type=str, default="config/tts_config.yaml")
    return parser.parse_args()


def should_run(download_cfg: dict) -> bool:
    return bool(download_cfg.get("enabled")) and download_cfg.get("source") in {"indicvoices_hf", "indicvoices_public_archives"}


def get_token(download_cfg: dict) -> str:
    import os

    token_env = download_cfg.get("hf_token_env", "HF_TOKEN")
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise EnvironmentError(f"Missing Hugging Face token in environment variable: {token_env}")
    return token


def selected_text(example: dict, transcript_priority: list[str]) -> str:
    for field in transcript_priority:
        value = str(example.get(field, "") or "").strip()
        if value:
            return normalize_text(value)
    return ""


DISFLUENCY_MARKERS = re.compile(
    r"\[(?:stammers?|inhaling|breathing|cough(?:ing)?|laughter|noise|pause|overlap|unclear|inaudible)\]",
    re.IGNORECASE,
)


def matches_filters(example: dict, download_cfg: dict, transcript_priority: list[str]) -> tuple[bool, str]:
    text = selected_text(example, transcript_priority)
    duration = float(example.get("duration") or 0.0)
    task_name = str(example.get("task_name") or "").strip()
    scenario = str(example.get("scenario") or "").strip()

    if not text:
        return False, "missing_text"
    if DISFLUENCY_MARKERS.search(text):
        return False, "disfluency"
    if len(text) < int(download_cfg["min_text_chars"]) or len(text) > int(download_cfg["max_text_chars"]):
        return False, "text_length"
    if duration < float(download_cfg["min_duration_seconds"]) or duration > float(download_cfg["max_duration_seconds"]):
        return False, "duration"

    include_task_names = set(download_cfg.get("include_task_names") or [])
    exclude_task_names = set(download_cfg.get("exclude_task_names") or [])
    include_scenarios = set(download_cfg.get("include_scenarios") or [])
    exclude_scenarios = set(download_cfg.get("exclude_scenarios") or [])

    if include_task_names and task_name not in include_task_names:
        return False, "task_name"
    if task_name in exclude_task_names:
        return False, "task_name"
    if include_scenarios and scenario not in include_scenarios:
        return False, "scenario"
    if scenario in exclude_scenarios:
        return False, "scenario"

    return True, "kept"


def iter_examples(dataset_name: str, config_name: str, split: str, token: str, streaming: bool = True):
    from datasets import load_dataset

    return load_dataset(dataset_name, config_name, split=split, streaming=streaming, token=token)


def iter_audio_examples(dataset_name: str, config_name: str, split: str, token: str, audio_field: str, sample_rate: int):
    from datasets import Audio, load_dataset

    dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, token=token)
    dataset = dataset.cast_column(audio_field, Audio(sampling_rate=sample_rate))
    return dataset


def download_public_archive(url: str, destination: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_public_archive(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as handle:
        handle.extractall(target_dir)


def candidate_audio_paths(extracted_root: Path, json_path: Path, payload: dict, audio_field: str) -> list[Path]:
    candidates: list[Path] = []
    audio_ref = str(payload.get(audio_field) or payload.get("audio_path") or "").strip()
    if audio_ref:
        audio_ref_path = Path(audio_ref)
        if audio_ref_path.is_absolute():
            candidates.append(audio_ref_path)
        else:
            candidates.extend(
                [
                    json_path.parent / audio_ref_path,
                    extracted_root / audio_ref_path,
                    extracted_root / audio_ref_path.name,
                ]
            )

    for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        candidates.append(json_path.with_suffix(ext))

    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def build_public_example(extracted_root: Path, json_path: Path, payload: dict, download_cfg: dict, transcript_priority: list[str]) -> dict | None:
    import soundfile as sf

    audio_path = None
    for candidate in candidate_audio_paths(extracted_root, json_path, payload, download_cfg["audio_field"]):
        if candidate.exists():
            audio_path = candidate.resolve()
            break
    if audio_path is None:
        return None

    speaker_id = str(payload.get("speaker_id") or payload.get("speakerId") or payload.get("speaker") or "unknown_speaker").strip()
    duration = payload.get("duration")
    if duration in (None, ""):
        try:
            duration = sf.info(str(audio_path)).duration
        except RuntimeError:
            duration = 0.0

    return {
        "text": selected_text(payload, transcript_priority),
        "duration": float(duration or 0.0),
        "task_name": str(payload.get("task_name") or "").strip(),
        "scenario": str(payload.get("scenario") or "").strip(),
        "speaker_id": speaker_id or "unknown_speaker",
        "gender": str(payload.get("gender") or "").strip(),
        "state": str(payload.get("state") or "").strip(),
        "local_audio_path": str(audio_path),
        "lang": str(payload.get("lang") or "hi").strip(),
    }


def public_archive_audio_path(extracted_root: Path, json_path: Path, payload: dict, audio_field: str) -> Path | None:
    for candidate in candidate_audio_paths(extracted_root, json_path, payload, audio_field):
        if candidate.exists():
            return candidate.resolve()
    return None


def public_archive_speaker_id(json_path: Path, payload: dict, download_cfg: dict) -> str:
    mode = download_cfg.get("speaker_selection", {}).get("mode", "top_duration")
    if mode == "mixed_segments":
        return "mixed_hindi"

    explicit = str(payload.get("speaker_id") or payload.get("speakerId") or payload.get("speaker") or "").strip()
    if explicit:
        return explicit

    return json_path.stem


def iter_public_archive_examples(extracted_root: Path, download_cfg: dict, transcript_priority: list[str]):
    for json_path in extracted_root.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        audio_path = public_archive_audio_path(extracted_root, json_path, payload, download_cfg["audio_field"])
        if audio_path is None:
            continue

        if isinstance(payload.get("verbatim"), list):
            speaker_id = public_archive_speaker_id(json_path, payload, download_cfg)
            for segment_index, segment in enumerate(payload["verbatim"]):
                text = normalize_text(str(segment.get("text") or "").strip())
                start = float(segment.get("start") or 0.0)
                end = float(segment.get("end") or 0.0)
                duration = max(0.0, end - start)
                if duration <= 0:
                    continue

                yield {
                    "text": text,
                    "duration": duration,
                    "task_name": str(payload.get("task_name") or "").strip(),
                    "scenario": str(payload.get("scenario") or "").strip(),
                    "speaker_id": speaker_id,
                    "gender": str(payload.get("gender") or "").strip(),
                    "state": str(payload.get("state") or "").strip(),
                    "local_audio_path": str(audio_path),
                    "lang": str(payload.get("language") or "hi").strip(),
                    "segment_index": segment_index,
                    "segment_start": start,
                    "segment_end": end,
                    "source_json": str(json_path),
                    "split": str(json_path.parent.name),
                }
            continue

        example = build_public_example(extracted_root, json_path, payload, download_cfg, transcript_priority)
        if example is not None:
            example["split"] = str(json_path.parent.name)
            yield example


def choose_speaker(download_cfg: dict, speaker_stats: dict[str, dict]) -> str:
    selection_cfg = download_cfg["speaker_selection"]
    mode = selection_cfg.get("mode", "top_duration")
    explicit_speaker_id = str(selection_cfg.get("speaker_id") or "").strip()

    if explicit_speaker_id:
        if explicit_speaker_id not in speaker_stats:
            raise ValueError(f"Configured speaker_id not found after filtering: {explicit_speaker_id}")
        return explicit_speaker_id

    if mode == "mixed_segments":
        return "mixed_hindi"

    if mode != "top_duration":
        raise ValueError(f"Unsupported speaker selection mode: {mode}")

    min_hours = float(selection_cfg.get("min_speaker_hours", 0.0))
    candidates = [
        (speaker_id, stats)
        for speaker_id, stats in speaker_stats.items()
        if (stats["duration_seconds"] / 3600.0) >= min_hours
    ]
    if not candidates:
        raise ValueError("No speaker met the minimum usable hours threshold after filtering")

    candidates.sort(key=lambda item: (item[1]["duration_seconds"], item[1]["samples"]), reverse=True)
    return candidates[0][0]


def collect_speaker_stats(examples, download_cfg: dict, transcript_priority: list[str]) -> tuple[dict[str, dict], Counter, Counter]:
    speaker_stats: dict[str, dict] = defaultdict(lambda: {
        "samples": 0,
        "duration_seconds": 0.0,
        "task_names": Counter(),
        "scenarios": Counter(),
        "genders": Counter(),
        "states": Counter(),
    })
    filter_reasons = Counter()
    split_counts = Counter()

    for example in examples:
        keep, reason = matches_filters(example, download_cfg, transcript_priority)
        filter_reasons[reason] += 1
        if not keep:
            continue

        speaker_id = str(example.get("speaker_id") or "unknown_speaker").strip() or "unknown_speaker"
        duration = float(example.get("duration") or 0.0)
        task_name = str(example.get("task_name") or "").strip()
        scenario = str(example.get("scenario") or "").strip()
        gender = str(example.get("gender") or "").strip()
        state = str(example.get("state") or "").strip()
        split = str(example.get("split") or "train")

        stats = speaker_stats[speaker_id]
        stats["samples"] += 1
        stats["duration_seconds"] += duration
        if task_name:
            stats["task_names"][task_name] += 1
        if scenario:
            stats["scenarios"][scenario] += 1
        if gender:
            stats["genders"][gender] += 1
        if state:
            stats["states"][state] += 1
        split_counts[split] += 1

    return speaker_stats, filter_reasons, split_counts


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    base_dir = PROJECT_ROOT

    download_cfg = config.get("download", {})
    dataset_cfg = config.get("dataset", {})

    if not should_run(download_cfg):
        print(json.dumps({"download": "skipped"}, indent=2))
        return

    output_root = resolve_path(download_cfg["output_root"], base_dir)
    preserve_download_cache = bool(download_cfg.get("preserve_download_cache", True))

    if output_root.exists() and not preserve_download_cache:
        shutil.rmtree(output_root)

    ensure_dir(output_root)
    clips_dir = output_root / "clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir = ensure_dir(clips_dir)
    metadata_path = output_root / "metadata.csv"
    summary_path = output_root / "download_summary.json"
    sample_rate = int(dataset_cfg.get("target_sample_rate", 22050))
    audio_field = download_cfg["audio_field"]
    transcript_priority = list(download_cfg.get("transcript_priority") or ["normalized", "text", "verbatim"])

    dataset_name = download_cfg.get("hf_dataset_name", "")
    config_name = download_cfg.get("hf_config_name", "")
    source = download_cfg.get("source")

    if source == "indicvoices_hf":
        token = get_token(download_cfg)
        speaker_stats, filter_reasons, split_counts = collect_speaker_stats(
            (
                {**example, "split": split}
                for split in download_cfg.get("splits", ["train"])
                for example in iter_examples(dataset_name, config_name, split, token)
            ),
            download_cfg,
            transcript_priority,
        )
        source_label = "huggingface"
        public_extracted_root = None
    elif source == "indicvoices_public_archives":
        archives_cfg = download_cfg.get("public_archives", {})
        archives_dir = ensure_dir(output_root / "archives")
        extracted_root = ensure_dir(output_root / "extracted")
        language_name = archives_cfg["language_name"]
        archive_template = archives_cfg["archive_url_template"]

        for version in archives_cfg.get("versions", [1, 2, 3, 4, 5]):
            archive_path = archives_dir / f"v{version}_{language_name}_train.tgz"
            extract_marker = extracted_root / f".extracted_v{version}"
            if not archive_path.exists():
                url = archive_template.format(version=version, language_name=language_name)
                print(f"Downloading {url} -> {archive_path}")
                download_public_archive(url, archive_path)
            if not extract_marker.exists():
                print(f"Extracting {archive_path}")
                extract_public_archive(archive_path, extracted_root)
                extract_marker.write_text("done", encoding="utf-8")
                if archives_cfg.get("delete_archive_after_extract", False) and archive_path.exists():
                    archive_path.unlink()

        speaker_stats, filter_reasons, split_counts = collect_speaker_stats(
            iter_public_archive_examples(extracted_root, download_cfg, transcript_priority),
            download_cfg,
            transcript_priority,
        )
        source_label = "public_archives"
        public_extracted_root = extracted_root
        dataset_name = "IndicVoices public Hindi archives"
        config_name = language_name.lower()
    else:
        raise ValueError(f"Unsupported download source: {source}")

    if not speaker_stats:
        raise ValueError("No Hindi IndicVoices samples remained after filtering")

    selected_speaker_id = choose_speaker(download_cfg, speaker_stats)
    selection_cfg = download_cfg["speaker_selection"]
    max_hours = float(selection_cfg.get("max_hours", 0.0))
    max_samples = int(selection_cfg.get("max_samples", 0))
    max_duration_seconds = max_hours * 3600.0 if max_hours > 0 else 0.0

    top_speakers = sorted(
        (
            {
                "speaker_id": speaker_id,
                "samples": stats["samples"],
                "hours": round(stats["duration_seconds"] / 3600.0, 3),
                "top_task_names": stats["task_names"].most_common(5),
                "top_scenarios": stats["scenarios"].most_common(5),
                "top_genders": stats["genders"].most_common(3),
                "top_states": stats["states"].most_common(5),
            }
            for speaker_id, stats in speaker_stats.items()
        ),
        key=lambda item: (item["hours"], item["samples"]),
        reverse=True,
    )

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_path", "text"],
        )
        writer.writeheader()

        written_samples = 0
        written_seconds = 0.0

        if source == "indicvoices_hf":
            import soundfile as sf
            from tqdm import tqdm

            token = get_token(download_cfg)
            for split in download_cfg.get("splits", ["train"]):
                iterator = iter_audio_examples(dataset_name, config_name, split, token, audio_field, sample_rate)
                for example in tqdm(iterator, desc=f"download-{split}"):
                    keep, _ = matches_filters(example, download_cfg, transcript_priority)
                    if not keep:
                        continue

                    speaker_id = str(example.get("speaker_id") or "unknown_speaker").strip() or "unknown_speaker"
                    if speaker_id != selected_speaker_id:
                        continue

                    if max_samples > 0 and written_samples >= max_samples:
                        break
                    if max_duration_seconds > 0 and written_seconds >= max_duration_seconds:
                        break

                    text = selected_text(example, transcript_priority)
                    audio = example[audio_field]
                    duration = float(example.get("duration") or 0.0)

                    basename = f"{selected_speaker_id}_{written_samples:06d}.wav".replace("/", "_")
                    output_path = clips_dir / basename
                    sf.write(str(output_path), audio["array"], int(audio["sampling_rate"]), subtype="PCM_16")

                    writer.writerow({"audio_path": str(output_path.relative_to(output_root)), "text": text})

                    written_samples += 1
                    written_seconds += duration

                if (max_samples > 0 and written_samples >= max_samples) or (
                    max_duration_seconds > 0 and written_seconds >= max_duration_seconds
                ):
                    break
        else:
            from tqdm import tqdm
            import soundfile as sf

            assert public_extracted_root is not None
            for example in tqdm(iter_public_archive_examples(public_extracted_root, download_cfg, transcript_priority), desc="curate-public"):
                keep, _ = matches_filters(example, download_cfg, transcript_priority)
                if not keep:
                    continue

                speaker_id = str(example.get("speaker_id") or "unknown_speaker").strip() or "unknown_speaker"
                if speaker_id != selected_speaker_id:
                    continue

                if max_samples > 0 and written_samples >= max_samples:
                    break
                if max_duration_seconds > 0 and written_seconds >= max_duration_seconds:
                    break

                source_audio_path = Path(str(example["local_audio_path"]))
                basename = f"{selected_speaker_id}_{written_samples:06d}.wav".replace("/", "_")
                output_path = clips_dir / basename

                if "segment_start" in example and "segment_end" in example:
                    info = sf.info(str(source_audio_path))
                    start_frame = max(0, int(float(example["segment_start"]) * info.samplerate))
                    end_frame = min(info.frames, int(float(example["segment_end"]) * info.samplerate))
                    if end_frame <= start_frame:
                        continue
                    audio, samplerate = sf.read(str(source_audio_path), start=start_frame, stop=end_frame)
                    if len(audio) == 0:
                        continue
                    sf.write(str(output_path), audio, samplerate, subtype="PCM_16")
                else:
                    shutil.copy2(source_audio_path, output_path)

                writer.writerow({"audio_path": str(output_path.relative_to(output_root)), "text": str(example["text"])})

                written_samples += 1
                written_seconds += float(example.get("duration") or 0.0)

    summary = {
        "source": source_label,
        "dataset_name": dataset_name,
        "config_name": config_name,
        "selected_speaker_id": selected_speaker_id,
        "selected_speaker_hours_available": round(speaker_stats[selected_speaker_id]["duration_seconds"] / 3600.0, 3),
        "selected_speaker_top_task_names": speaker_stats[selected_speaker_id]["task_names"].most_common(10),
        "selected_speaker_top_scenarios": speaker_stats[selected_speaker_id]["scenarios"].most_common(10),
        "written_samples": written_samples,
        "written_hours": round(written_seconds / 3600.0, 3),
        "filter_reasons": dict(filter_reasons),
        "split_counts_after_filtering": dict(split_counts),
        "top_speakers": top_speakers[:20],
        "output_root": str(output_root),
        "metadata_path": str(metadata_path),
    }
    write_json(summary_path, summary)

    if source == "indicvoices_public_archives" and download_cfg.get("cleanup_extracted_after_curate", False):
        if public_extracted_root is not None and public_extracted_root.exists():
            shutil.rmtree(public_extracted_root)
        archives_dir = output_root / "archives"
        if archives_dir.exists():
            shutil.rmtree(archives_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
