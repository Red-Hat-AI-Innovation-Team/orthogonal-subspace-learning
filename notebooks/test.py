# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
# from datasets import load_dataset
# from tqdm import tqdm
# import time
# import json
# from collections import Counter

# from accelerate import Accelerator

# accelerator = Accelerator(mixed_precision="bf16")
# device = accelerator.device

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,                      # or load_in_8bit=True for 8-bit
#     bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 math on H100
#     bnb_4bit_use_double_quant=True,         # slight memory/performance improvement
# )

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ###################################################
# # 1. Define a PyTorch Dataset for DBpedia
# ###################################################
# class DBpediaDataset(Dataset):
#     """
#     PyTorch dataset wrapper for the DBpedia dataset.
#     Each example is converted to a text-to-text format.
#     """
#     def __init__(self, json_file, tokenizer):
#         """
#         hf_dataset: the Hugging Face dataset loaded via load_dataset("dbpedia_14")
#         split: "train" or "test"
#         tokenizer: a LLaMATokenizer instance
#         label_mapping: a dict mapping integer labels to string labels, e.g. {0:"Company", ...}
#         """
#         self.tokenizer = tokenizer

#         # Load data from JSON file
#         with open(json_file, "r", encoding="utf-8") as f:
#             self.dataset = json.load(f)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         sample = self.dataset[idx]
#         input_text = (
#             "Classify the following text into one of these categories: "
#             "[Company, Educational Institution, Artist, Athlete, Office Holder, Mean of Transportation, "
#             "Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work].\n\n"
#             "Text: " + sample["sentence"] + "\nAnswer:"
#         )
#         target_text = sample["label"]
#         return input_text, target_text


# ###################################################
# # 2. Collate Function
# ###################################################
# def collate_fn(batch, tokenizer, max_source_length=512):
#     inputs, targets = zip(*batch)
    
#     # Format as "input + target" since LLaMA is an autoregressive model
#     formatted_texts = [inp + " " + tgt for inp, tgt in zip(inputs, targets)]
    
#     encodings = tokenizer(formatted_texts, padding=True, truncation=True, max_length=max_source_length, return_tensors="pt")
#     encodings["labels"] = encodings["input_ids"].clone()  # Labels should be the same as input_ids for CLMs
    
#     return encodings

# def load_finetuned_model(model_path="llama_finetuned_dbpedia.pt"):
#     """
#     Load the fine-tuned LLaMA model from disk.
#     """
#     model_name = "baffo32/decapoda-research-llama-7B-hf"
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token  # Add this line
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

#     # Load the trained weights
#     model.load_state_dict(torch.load(model_path))

#     # Move to GPU and wrap in DataParallel
#     model = model.to(device)

#     print(f"Loaded fine-tuned model from {model_path}")
#     return model, tokenizer

# ###################################################
# # 3. Training and Evaluation Functions
# ###################################################
# def train_finetune_llama():
#     # Define paths to JSON train and test files
#     train_json_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/train.json"
#     test_json_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json"

#     # Load pretrained LLaMA tokenizer and model (LLaMA)
#     model_name = "baffo32/decapoda-research-llama-7B-hf"
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token  # Add this line
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
#     model.gradient_checkpointing_enable()

#     # Create PyTorch datasets for train and test splits
#     train_dataset = DBpediaDataset(train_json_path, tokenizer)
#     test_dataset = DBpediaDataset(test_json_path, tokenizer)

#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
#                               collate_fn=lambda batch: collate_fn(batch, tokenizer))
#     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
#                              collate_fn=lambda batch: collate_fn(batch, tokenizer))

#     # Prepare optimizer (full fine-tuning; all model parameters are updated)
#     optimizer = optim.AdamW(model.parameters(), lr=5e-5)
#     num_epochs = 1

#     # Prepare dataloader and optimizer
#     train_loader, optimizer, model = accelerator.prepare(train_loader, optimizer, model)

#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)
#         start_time = time.time()

#         for batch in progress_bar:

#             outputs = model(**batch)
#             loss = outputs.loss

#             optimizer.zero_grad()
#             accelerator.backward(loss)
#             optimizer.step()

#             total_loss += loss.item()

#             # Estimate time remaining
#             elapsed_time = time.time() - start_time
#             remaining_time = elapsed_time / (progress_bar.n + 1) * (len(train_loader) - progress_bar.n)
#             progress_bar.set_postfix(loss=loss.item(), eta=f"{remaining_time:.2f}s")

#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

#     # Save the fine-tuned model
#     # torch.save(model.state_dict(), "t5_finetuned_dbpedia.pt")
#     torch.save(accelerator.unwrap_model(model).state_dict(), "llama_finetuned_dbpedia.pt")
#     print("Model saved as 'llama_finetuned_dbpedia.pt'.")

#     return model, tokenizer, train_loader, test_loader


# def evaluate(model, tokenizer, data_loader, dataset_name="Test"):
#     """
#     Evaluate the fine-tuned model on the test set.
#     """
#     model.eval()
#     correct = 0
#     total = 0
#     sample_count = 0
#     prediction_counts = Counter()  # Dictionary to store label prediction counts

#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc=f"Evaluating on {dataset_name} set", unit="batch"):
#             # Generate predictions
#             outputs = model(**batch)
#             logits = outputs.logits  # Get raw logits

#             # Decode the most likely token
#             predictions = tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)

#             # Decode the ground truth labels
#             targets = [tokenizer.decode(t, skip_special_tokens=True).strip() for t in batch["labels"]]

#             for pred, target in zip(predictions, targets):
#                 prediction_counts[pred.lower()] += 1  # Count predictions
#                 if pred.lower() == target.lower():
#                     correct += 1
#                 total += 1
#                 # Print only the first 10 examples
#                 if sample_count < 5:
#                     print(f"[{dataset_name} Set] Target: {target} | Prediction: {pred}")
#                     sample_count += 1

#     accuracy = correct / total if total > 0 else 0.0
#     print(f"{dataset_name} Accuracy: {accuracy*100:.2f}%")

#     # Print the prediction count for each label
#     print(f"Prediction Distribution in {dataset_name} Set:")
#     for label, count in prediction_counts.items():
#         print(f"{label}: {count}")

#     return accuracy


# ###################################################
# # 5. Main: Train, Check, and Evaluate
# ###################################################
# if __name__ == "__main__":
#     # Train and fine-tune LLaMA on DBpedia
#     model1, tokenizer, train_loader, test_loader = train_finetune_llama()

#     # Load the fine-tuned model
#     model1, tokenizer = load_finetuned_model(model_path="llama_finetuned_dbpedia.pt")

#     # Define paths to JSON train and test files
#     train_json_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/train.json"
#     test_json_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json"

#     # Load datasets
#     train_dataset = DBpediaDataset(train_json_path, tokenizer)
#     test_dataset = DBpediaDataset(test_json_path, tokenizer)

#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
#                               collate_fn=lambda batch: collate_fn(batch, tokenizer))
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
#                              collate_fn=lambda batch: collate_fn(batch, tokenizer))
    
#     test_loader, model1 = accelerator.prepare(test_loader, model1)

#     # Evaluate on train set
#     train_accuracy = evaluate(model1, tokenizer, train_loader, dataset_name="Train")

#     # Evaluate on test set
#     test_accuracy = evaluate(model1, tokenizer, test_loader, dataset_name="Test")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from tqdm import tqdm
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Define Dataset
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
            "[Company, Educational Institution, Artist, Athlete, Office Holder, Mean of Transportation, "
            "Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work].\n\n"
            "Text: " + sample["sentence"] + "\nAnswer:"
        )
        target_text = sample["label"]
        return input_text, target_text

# Collate function
def collate_fn(batch, tokenizer, max_length=512):
    inputs, targets = zip(*batch)
    formatted_texts = [inp + " " + tgt for inp, tgt in zip(inputs, targets)]
    encodings = tokenizer(formatted_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encodings["input_ids"] = encodings["input_ids"]
    encodings["attention_mask"] = encodings["attention_mask"]
    encodings["labels"] = encodings["input_ids"].clone()

    return encodings

# Load model and tokenizer
def load_model():
    model_name = "baffo32/decapoda-research-llama-7B-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    return model, tokenizer

# Training function
def train_model(model, tokenizer, train_loader, accelerator):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(1):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")
        for batch in progress_bar:
            # Ensure batch tensors are on the same device as the model
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            # Use accelerator for backward
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        accelerator.print(f"Epoch finished - Avg Loss: {total_loss / len(train_loader):.4f}")

def evaluate(model, tokenizer, data_loader, accelerator):
    model.eval()
    correct, total = 0, 0
    # Same idea: the model/batch are already on GPU thanks to accelerate
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            outputs = model(**batch)
            # shape = (batch_size, sequence_length, vocab_size)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert back to text
            predictions_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            targets_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            
            for pred_str, tgt_str in zip(predictions_texts, targets_texts):
                # Simple string match
                if pred_str.strip().lower() == tgt_str.strip().lower():
                    correct += 1
                total += 1

    accuracy = 100.0 * correct / total
    accelerator.print(f"Accuracy: {accuracy:.2f}%")

# Main execution
if __name__ == "__main__":
    train_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/train.json"
    test_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json"

    # Initialize Accelerator
    accelerator = Accelerator()
    
    model, tokenizer = load_model()
    train_dataset = DBpediaDataset(train_path, tokenizer)
    test_dataset = DBpediaDataset(test_path, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Prepare everything for Accelerate
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    # Training
    train_model(model, optimizer, train_loader, accelerator)

    # Evaluation
    evaluate(model, tokenizer, test_loader, accelerator)