import argparse
import os

import jiwer
import torch
import yaml

from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Whisper ASR student from paired manifests")
    parser.add_argument("--config", type=str, default="../../config/asr_config.yaml")
    return parser.parse_args()


def normalize_text(text):
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    return transform(text or "")


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics_factory(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
        references = processor.batch_decode(label_ids, skip_special_tokens=True)
        predictions = [normalize_text(text) for text in predictions]
        references = [normalize_text(text) for text in references]
        return {"wer": jiwer.wer(references, predictions), "cer": jiwer.cer(references, predictions)}

    return compute_metrics


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_name = config["training"]["student_model_name"]
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(
        language=config["training"].get("language"),
        task=config["training"].get("task", "transcribe"),
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = config["training"].get("language")
    model.generation_config.task = config["training"].get("task", "transcribe")
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    dataset = load_dataset(
        "json",
        data_files={
            "train": config["data"]["train_manifest_path"],
            "eval": config["data"]["eval_manifest_path"],
        },
    )
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=config["data"].get("sample_rate", 16000)))

    max_duration = config["training"].get("max_duration_seconds", 30.0)

    def prepare_sample(batch):
        audio = batch["audio_path"]
        duration = len(audio["array"]) / max(audio["sampling_rate"], 1)
        batch["duration_seconds"] = duration
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare_sample, remove_columns=dataset["train"].column_names)
    dataset = dataset.filter(lambda sample: sample["duration_seconds"] <= max_duration)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        max_steps=config["training"].get("max_steps", -1),
        predict_with_generate=True,
        generation_max_length=config["training"].get("generation_max_length", 225),
        fp16=(config["training"].get("mixed_precision") == "fp16"),
        remove_unused_columns=False,
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 2),
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=compute_metrics_factory(processor),
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    processor.save_pretrained(config["training"]["output_dir"])


if __name__ == "__main__":
    main()
