#!/usr/bin/env python
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import json
from torch.utils.data import Dataset

class DBpediaDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        self.tokenizer = tokenizer
        with open(json_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        input_text = (
            "Classify the following text into one of these categories: "
            "[Company, Educational Institution, Artist, Athlete, Office Holder, "
            "Mean of Transportation, Building, Natural Place, Village, Animal, "
            "Plant, Album, Film, Written Work].\n\n"
            "Text: " + sample["sentence"] + "\nAnswer:"
        )
        target_text = sample["label"]

        # Tokenize input and label
        input_ids = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )["input_ids"].squeeze()

        target_ids = self.tokenizer(
            target_text, truncation=True, padding="max_length", max_length=10, return_tensors="pt"
        )["input_ids"].squeeze()

        return {"input_ids": input_ids, "labels": target_ids}

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = "baffo32/decapoda-research-llama-7B-hf"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    train_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/train.json"
    test_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json"

    print("Loading custom DBpedia dataset...")
    train_dataset = DBpediaDataset(train_path, tokenizer)
    test_dataset = DBpediaDataset(test_path, tokenizer)

    print("Loading model...")
    # Load model using BF16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model.to("cuda:0")

    # Optional: Enable gradient checkpointing if needed
    # model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
    output_dir="./llama7b_dbpedia",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,  # Adjust batch size if needed
    gradient_accumulation_steps=4,  # Reduce if OOM
    bf16=True,
    fp16=False,
    max_grad_norm=1.0,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate after every epoch
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Add evaluation dataset
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    try:
        print("Starting training...")
        trainer.train()
        print("Evaluating model...")
        trainer.evaluate()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Encountered an OOM error. Reduce batch size, sequence length, or enable gradient checkpointing.")
            torch.cuda.empty_cache()
        else:
            raise e

if __name__ == "__main__":
    main()