import argparse
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VITS student with Coqui TTS")
    parser.add_argument("--config", type=str, default="../../config/tts_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_dir = config["data"]["student_dataset_dir"]
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing Coqui dataset metadata: {metadata_path}")

    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseAudioConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    output_dir = os.path.abspath(config["training"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=os.path.abspath(dataset_dir),
        language=config["training"].get("language"),
    )
    audio_config = BaseAudioConfig(
        sample_rate=config["data"].get("sample_rate", 22050),
        resample=False,
        do_trim_silence=False,
    )
    vits_config = VitsConfig(
        output_path=output_dir,
        run_name="vits_student",
        batch_size=config["training"]["batch_size"],
        eval_batch_size=config["training"]["eval_batch_size"],
        num_loader_workers=config["training"]["num_loader_workers"],
        num_eval_loader_workers=config["training"]["num_eval_loader_workers"],
        epochs=config["training"]["num_epochs"],
        mixed_precision=config["training"]["mixed_precision"],
        print_step=config["training"]["print_step"],
        save_step=config["training"]["save_step"],
        eval_split_size=config["training"]["eval_split_size"],
        run_eval=True,
        use_phonemes=config["training"].get("use_phonemes", False),
        text_cleaner=config["training"].get("text_cleaner", "basic_cleaners"),
        audio=audio_config,
        datasets=[dataset_config],
    )

    audio_processor = AudioProcessor.init_from_config(vits_config)
    tokenizer, vits_config = TTSTokenizer.init_from_config(vits_config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=vits_config.eval_split_size,
        eval_split_max_size=vits_config.eval_split_max_size,
    )
    model = Vits(vits_config, audio_processor, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(),
        vits_config,
        output_dir,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
