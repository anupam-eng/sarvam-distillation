import argparse
import os
import yaml
import torch
import webdataset as wds
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    TrainingArguments, Trainer
)
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train ASR Student (Distillation)")
    parser.add_argument("--config", type=str, default="../../config/asr_config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    student_model_name = config["training"]["student_model_name"]
    processor = Wav2Vec2Processor.from_pretrained(student_model_name)
    model = Wav2Vec2ForCTC.from_pretrained(
        student_model_name,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True
    )
    
    shard_pattern = os.path.join(config["data"]["shards_dir"], "*.tar")
    dataset = wds.WebDataset(shard_pattern).decode("l").to_tuple("wav", "json")
    
    def transform(sample):
        wav_bytes, json_data = sample
        import io, torchaudio
        waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        inputs = processor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        with processor.as_target_processor():
            labels = processor(json_data["transcript"], return_tensors="pt").input_ids
            
        return {"input_values": inputs.input_values[0], "labels": labels[0]}
        
    mapped_dataset = dataset.map(transform)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        group_by_length=False, # True causes issues with Iterables sometimes without length
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        evaluation_strategy="no", 
        num_train_epochs=30,
        fp16=(config["training"]["mixed_precision"] == "fp16"),
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    class DataCollatorCTCWithPadding:
        def __init__(self, processor, padding=True):
            self.processor = processor
            self.padding = padding

        def __call__(self, features):
            input_values = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_values,
                padding=self.padding,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Note: webdataset doesn't have length out of the box, need to specify max_steps usually
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=mapped_dataset.with_epoch(10000), # Mock epoch length
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

if __name__ == "__main__":
    main()
