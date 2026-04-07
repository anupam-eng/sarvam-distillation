from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.vits_pipeline.common import PROJECT_ROOT, ensure_dir, find_latest_matching, load_config, resolve_path


@dataclass
class EvalSample:
    basename: str
    text: str
    audio_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor checkpoint mel-MSE and stop training when it overfits")
    parser.add_argument("--config", type=str, default="config/tts_config.yaml")
    parser.add_argument("--train-pid", type=int, default=0)
    return parser.parse_args()


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_eval_subset(processed_dir: Path, subset_size: int) -> list[EvalSample]:
    manifest = processed_dir / "metadata_eval.csv"
    wavs_dir = processed_dir / "wavs"
    samples: list[EvalSample] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", maxsplit=2)
            if len(parts) < 2:
                continue
            basename = parts[0]
            text = parts[1].strip()
            audio_path = wavs_dir / f"{basename}.wav"
            if audio_path.exists():
                samples.append(EvalSample(basename=basename, text=text, audio_path=audio_path))
            if len(samples) >= subset_size:
                break
    if not samples:
        raise ValueError(f"No eval samples found in {manifest}")
    return samples


def compute_log_mel(audio: np.ndarray, sample_rate: int, audio_cfg: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=int(audio_cfg["fft_size"]),
        hop_length=int(audio_cfg["hop_length"]),
        win_length=int(audio_cfg["win_length"]),
        n_mels=int(audio_cfg["num_mels"]),
        fmin=float(audio_cfg["mel_fmin"]),
        fmax=audio_cfg.get("mel_fmax") or sample_rate / 2,
        power=1.0,
    )
    return np.log10(np.clip(mel, 1e-5, None))


def mel_mse(reference_audio_path: Path, predicted_audio: np.ndarray, predicted_sr: int, audio_cfg: dict) -> float:
    target_sr = int(audio_cfg["sample_rate"])
    reference_audio, _ = librosa.load(str(reference_audio_path), sr=target_sr, mono=True)
    if predicted_sr != target_sr:
        predicted_audio = librosa.resample(predicted_audio.astype(np.float32), orig_sr=predicted_sr, target_sr=target_sr)

    reference_mel = compute_log_mel(reference_audio.astype(np.float32), target_sr, audio_cfg)
    predicted_mel = compute_log_mel(predicted_audio.astype(np.float32), target_sr, audio_cfg)

    frame_count = min(reference_mel.shape[1], predicted_mel.shape[1])
    if frame_count <= 0:
        raise ValueError("Mel spectrogram comparison had zero overlapping frames")

    diff = reference_mel[:, :frame_count] - predicted_mel[:, :frame_count]
    return float(np.mean(np.square(diff)))


def latest_run_dir(output_dir: Path, run_name: str) -> Path | None:
    candidates = [path for path in output_dir.glob(f"{run_name}-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"checkpoint_(\d+)\.pth$", path.name)
    if not match:
        return None
    return int(match.group(1))


def write_jsonl(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def evaluate_checkpoint(checkpoint_path: Path, config_path: Path, eval_samples: list[EvalSample], audio_cfg: dict, use_cuda: bool) -> dict:
    from TTS.utils.synthesizer import Synthesizer

    synthesizer = Synthesizer(tts_checkpoint=str(checkpoint_path), tts_config_path=str(config_path), use_cuda=use_cuda)
    sample_scores = []
    for sample in eval_samples:
        wav = synthesizer.tts(sample.text)
        score = mel_mse(sample.audio_path, np.asarray(wav, dtype=np.float32), synthesizer.output_sample_rate, audio_cfg)
        sample_scores.append({"basename": sample.basename, "text": sample.text, "mel_mse": score})

    mean_score = float(np.mean([item["mel_mse"] for item in sample_scores]))
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step(checkpoint_path),
        "mel_mse_mean": mean_score,
        "sample_scores": sample_scores,
    }


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)

    monitoring_cfg = config.get("monitoring", {})
    if not monitoring_cfg.get("enabled", False):
        print(json.dumps({"monitoring": "disabled"}, indent=2))
        return

    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    audio_cfg = config["audio"]

    output_dir = resolve_path(training_cfg["output_dir"], PROJECT_ROOT)
    processed_dir = resolve_path(dataset_cfg["processed_dir"], PROJECT_ROOT)
    poll_seconds = int(monitoring_cfg.get("poll_seconds", 60))
    min_checkpoint_step = int(monitoring_cfg.get("min_checkpoint_step", 0))
    eval_subset_size = int(monitoring_cfg.get("eval_subset_size", 8))
    min_checkpoints_before_stop = int(monitoring_cfg.get("min_checkpoints_before_stop", 4))
    patience_checkpoints = int(monitoring_cfg.get("patience_checkpoints", 4))
    min_improvement = float(monitoring_cfg.get("min_improvement", 0.05))
    overfit_margin = float(monitoring_cfg.get("overfit_margin", 0.15))
    use_cuda = not bool(monitoring_cfg.get("use_cpu", True))
    metrics_path = resolve_path(monitoring_cfg["metrics_path"], PROJECT_ROOT)
    summary_path = resolve_path(monitoring_cfg["summary_path"], PROJECT_ROOT)

    eval_samples = load_eval_subset(processed_dir, eval_subset_size)

    seen_steps: set[int] = set()
    best_score: float | None = None
    best_step: int | None = None
    non_improving = 0
    evaluated_count = 0
    stop_requested = False

    summary = {
        "status": "running",
        "best_mel_mse": None,
        "best_checkpoint_step": None,
        "last_evaluated_step": None,
        "evaluated_count": 0,
        "non_improving_checkpoints": 0,
        "stop_requested": False,
        "stop_reason": "",
    }
    write_summary(summary_path, summary)

    while True:
        run_dir = latest_run_dir(output_dir, training_cfg["run_name"])
        checkpoints = [] if run_dir is None else sorted(run_dir.glob("checkpoint_*.pth"), key=lambda path: checkpoint_step(path) or -1)

        for checkpoint in checkpoints:
            step = checkpoint_step(checkpoint)
            if step is None or step < min_checkpoint_step or step in seen_steps:
                continue

            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue

            try:
                result = evaluate_checkpoint(checkpoint, config_path, eval_samples, audio_cfg, use_cuda=use_cuda)
            except Exception as exc:
                write_jsonl(
                    metrics_path,
                    {
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_step": step,
                        "error": str(exc),
                        "skipped": True,
                    },
                )
                continue

            step = int(result["checkpoint_step"])
            current_score = float(result["mel_mse_mean"])
            seen_steps.add(step)
            evaluated_count += 1

            improved = best_score is None or current_score < (best_score - min_improvement)
            if improved:
                best_score = current_score
                best_step = step
                non_improving = 0
            else:
                non_improving += 1

            overfit_detected = (
                best_score is not None
                and evaluated_count >= min_checkpoints_before_stop
                and non_improving >= patience_checkpoints
                and current_score > (best_score + overfit_margin)
            )

            record = {
                **result,
                "improved": improved,
                "best_mel_mse_so_far": best_score,
                "best_checkpoint_step_so_far": best_step,
                "non_improving_checkpoints": non_improving,
                "overfit_detected": overfit_detected,
            }
            write_jsonl(metrics_path, record)

            summary = {
                "status": "running",
                "best_mel_mse": best_score,
                "best_checkpoint_step": best_step,
                "last_evaluated_step": step,
                "evaluated_count": evaluated_count,
                "non_improving_checkpoints": non_improving,
                "stop_requested": False,
                "stop_reason": "",
            }

            if overfit_detected:
                stop_requested = True
                if process_is_alive(args.train_pid):
                    os.kill(args.train_pid, signal.SIGTERM)
                summary.update(
                    {
                        "status": "stopped_for_overfit",
                        "stop_requested": True,
                        "stop_reason": f"checkpoint mel_mse degraded to {current_score:.4f} after best {best_score:.4f} at step {best_step}",
                    }
                )

            write_summary(summary_path, summary)

            if stop_requested:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return

        if not process_is_alive(args.train_pid):
            final_status = "training_exited"
            if stop_requested:
                final_status = "stopped_for_overfit"
            summary = {
                "status": final_status,
                "best_mel_mse": best_score,
                "best_checkpoint_step": best_step,
                "last_evaluated_step": max(seen_steps) if seen_steps else None,
                "evaluated_count": evaluated_count,
                "non_improving_checkpoints": non_improving,
                "stop_requested": stop_requested,
                "stop_reason": summary.get("stop_reason", ""),
            }
            write_summary(summary_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
