from __future__ import annotations

import argparse
from pathlib import Path

from src.vits_pipeline.common import PROJECT_ROOT, ensure_dir, load_config, read_nonempty_lines, resolve_path


HINDI_PUNCTUATIONS = "!'(),-.:;? []।॥"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Coqui VITS model from the prepared dataset")
    parser.add_argument("--config", type=str, default="config/tts_config.yaml")
    return parser.parse_args()


def load_samples(manifest_path: Path, wavs_dir: Path, speaker_name: str, language: str) -> list[dict]:
    samples = []
    for line in read_nonempty_lines(manifest_path):
        parts = line.split("|", maxsplit=2)
        if len(parts) < 2:
            continue
        basename = parts[0]
        text = parts[1].strip()
        audio_file = wavs_dir / f"{basename}.wav"
        samples.append(
            {
                "text": text,
                "audio_file": str(audio_file),
                "audio_unique_name": basename,
                "speaker_name": speaker_name,
                "root_path": str(wavs_dir.parent),
                "language": language,
            }
        )
    return samples


def build_characters_config(train_samples: list[dict], eval_samples: list[dict]):
    from TTS.tts.configs.shared_configs import CharactersConfig

    all_text = "".join(sample["text"] for sample in [*train_samples, *eval_samples])
    punctuation_set = set(HINDI_PUNCTUATIONS)
    characters = "".join(sorted({char for char in all_text if char not in punctuation_set and not char.isspace()}))
    if not characters:
        raise ValueError("Could not build a character vocabulary from the prepared dataset")

    return CharactersConfig(
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=characters,
        punctuations=HINDI_PUNCTUATIONS,
        is_unique=True,
        is_sorted=True,
    )


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    base_dir = PROJECT_ROOT

    dataset_cfg = config["dataset"]
    audio_cfg = config["audio"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    inference_cfg = config.get("inference", {})
    project_cfg = config.get("project", {})

    processed_dir = resolve_path(dataset_cfg["processed_dir"], base_dir)
    wavs_dir = processed_dir / "wavs"
    train_manifest = processed_dir / "metadata_train.csv"
    eval_manifest = processed_dir / "metadata_eval.csv"
    output_dir = ensure_dir(resolve_path(training_cfg["output_dir"], base_dir))
    eval_prompts_path = resolve_path(inference_cfg["prompts_file"], base_dir)

    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")
    if not eval_manifest.exists():
        raise FileNotFoundError(f"Eval manifest not found: {eval_manifest}")

    speaker_name = dataset_cfg.get("default_speaker_name") or "speaker"
    language = str(dataset_cfg.get("language") or "")
    train_samples = load_samples(train_manifest, wavs_dir, speaker_name, language)
    eval_samples = load_samples(eval_manifest, wavs_dir, speaker_name, language)

    if not train_samples:
        raise ValueError("No training samples were found in the prepared dataset")

    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseAudioConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=train_manifest.name,
        path=str(processed_dir),
        language=language or None,
    )
    audio_config = BaseAudioConfig(
        sample_rate=int(audio_cfg["sample_rate"]),
        fft_size=int(audio_cfg["fft_size"]),
        win_length=int(audio_cfg["win_length"]),
        hop_length=int(audio_cfg["hop_length"]),
        num_mels=int(audio_cfg["num_mels"]),
        mel_fmin=float(audio_cfg["mel_fmin"]),
        mel_fmax=audio_cfg.get("mel_fmax"),
        resample=False,
        do_trim_silence=False,
    )

    vits_config = VitsConfig(
        output_path=str(output_dir),
        run_name=training_cfg["run_name"],
        batch_size=int(training_cfg["batch_size"]),
        eval_batch_size=int(training_cfg["eval_batch_size"]),
        num_loader_workers=int(training_cfg["num_loader_workers"]),
        num_eval_loader_workers=int(training_cfg["num_eval_loader_workers"]),
        run_eval=bool(eval_samples),
        test_delay_epochs=-1,
        epochs=int(training_cfg["epochs"]),
        print_step=int(training_cfg["print_step"]),
        print_eval=False,
        mixed_precision=bool(training_cfg["mixed_precision"]),
        save_step=int(training_cfg["save_step"]),
        save_n_checkpoints=int(training_cfg["save_n_checkpoints"]),
        save_best_after=int(training_cfg["save_best_after"]),
        training_seed=int(project_cfg.get("seed", 17)),
        lr_gen=float(training_cfg["lr_gen"]),
        lr_disc=float(training_cfg["lr_disc"]),
        min_text_len=int(dataset_cfg["min_text_chars"]),
        max_text_len=int(dataset_cfg["max_text_chars"]),
        min_audio_len=max(1, int(float(dataset_cfg["min_audio_seconds"]) * int(audio_cfg["sample_rate"]))),
        max_audio_len=int(float(dataset_cfg["max_audio_seconds"]) * int(audio_cfg["sample_rate"])),
        text_cleaner=model_cfg["text_cleaner"],
        use_phonemes=bool(model_cfg["use_phonemes"]),
        phoneme_language=model_cfg.get("phoneme_language"),
        phoneme_cache_path=str(output_dir / "phoneme_cache"),
        add_blank=bool(model_cfg.get("add_blank", True)),
        test_sentences_file=str(eval_prompts_path) if eval_prompts_path.exists() else "",
        characters=build_characters_config(train_samples, eval_samples),
        audio=audio_config,
        datasets=[dataset_config],
    )

    audio_processor = AudioProcessor.init_from_config(vits_config)
    tokenizer, vits_config = TTSTokenizer.init_from_config(vits_config)
    model = Vits(vits_config, audio_processor, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(),
        vits_config,
        str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
