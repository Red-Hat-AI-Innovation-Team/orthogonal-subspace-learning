import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import json
import os
import deepspeed
from torch.utils.data.distributed import DistributedSampler

# Force usage of a single GPU: GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define DeepSpeed configuration
ds_config = {
    "train_micro_batch_size_per_gpu": 2,  # Reduce batch size per GPU
    "zero_optimization": {
        "stage": 2,  # ZeRO-3: Offload optimizer, gradients, and model states
    },
    "gradient_checkpointing": True,  # Save memory
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}

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
            "[Company, Educational Institution, Artist, Athlete, Office Holder, "
            "Mean of Transportation, Building, Natural Place, Village, Animal, "
            "Plant, Album, Film, Written Work].\n\n"
            "Text: " + sample["sentence"] + "\nAnswer:"
        )
        target_text = sample["label"]
        return input_text, target_text

# Collate function with reduced max_length (optional)
def collate_fn(batch, tokenizer, max_length=256):
    inputs, targets = zip(*batch)
    # Create full texts: prompt + " " + target
    full_texts = [inp + " " + tgt for inp, tgt in zip(inputs, targets)]
    encodings = tokenizer(
        full_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    
    # Now create labels, but mask out the prompt tokens.
    # First, get the tokenized version of just the prompt.
    prompt_texts = [inp + " " for inp in inputs]
    prompt_encodings = tokenizer(
        prompt_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    labels = encodings["input_ids"].clone()
    
    # For each sample, set label tokens corresponding to the prompt to -100.
    for i in range(len(full_texts)):
        prompt_length = prompt_encodings["input_ids"][i].ne(tokenizer.pad_token_id).sum()
        # Set all tokens up to prompt_length to -100 so loss isn't computed on them
        labels[i, :prompt_length] = -100
    encodings["labels"] = labels
    return encodings

# Load model and tokenizer (modified for one GPU only)
def load_model():
    model_name = "baffo32/decapoda-research-llama-7B-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with BF16 and enable gradient checkpointing for memory savings
    model = LlamaForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()  # Save memory during backpropagation
    return model, tokenizer

def load_finetuned_model(model_path="llama_finetuned_dbpedia"):
    """
    Correctly load the fine-tuned LLaMA model from a DeepSpeed checkpoint.
    """
    model_name = "baffo32/decapoda-research-llama-7B-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Add padding token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize model
    model = LlamaForCausalLM.from_pretrained(f"/workspace/orthogonal-subspace/notebooks/{model_path}/converted_model")

    # Reinitialize DeepSpeed and Load the Checkpoint
    model = deepspeed.initialize(
        model=model,
        config=ds_config
    )[0]  # Get model from tuple

    model.load_checkpoint(model_path)  # ✅ Load DeepSpeed checkpoint properly
    model.eval()  # Set to evaluation mode

    print(f"✅ Successfully loaded fine-tuned model from {model_path}")
    return model, tokenizer

# Training function (no accelerate)
def train_model(model, tokenizer, train_loader):
    model.train()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, 
        config=ds_config
    )

    # model = torch.compile(model)
    for epoch in range(1):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        for batch in progress_bar:
            model_engine.zero_grad()
            # Move batch to model's device (using the first parameter's device)
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}  # Move batch to correct model device

            outputs = model_engine(**batch)
            loss = outputs.loss

            with open("loss.txt", "a") as f:  # "a" mode appends to the file
                print(f"Loss: {loss}", file=f)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch finished - Avg Loss: {total_loss / len(train_loader):.4f}")
    
    # Save the trained model
    model_path = "llama_finetuned_dbpedia"
    model_engine.save_checkpoint(model_path)
    print(f"Model saved as '{model_path}'")

def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    # Tokenize only the prompt (without the target)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # Generate tokens starting from the prompt
    outputs = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        return_dict_in_generate=True
    )
    
    # Extract only the newly generated tokens (excluding the input)
    new_token_ids = outputs.sequences[:, input_ids.shape[-1]:]
    
    # Decode only the newly generated tokens
    new_text = tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
    
    return new_text.strip()

def evaluate(model, tokenizer, data_loader):
    correct, total = 0, 0
    sample_count = 0
    for inp_list, tgt_list in tqdm(data_loader, desc="Evaluating"):
        inp = inp_list[0] + " "  # unwrap the single item
        tgt = tgt_list[0]
        generated_answer = generate_answer(model, tokenizer, inp)
        if generated_answer.lower() == tgt.lower():
            correct += 1
        if sample_count < 5:
            with open("output.txt", "a") as f:  # "a" mode appends to the file
                print(f"[Target: {tgt.strip()} | Prediction: {generated_answer.strip()}]", file=f)
            sample_count += 1
        total += 1
    print(f"Accuracy: {100.0 * correct / total:.2f}%")

# Main
if __name__ == "__main__":

    deepspeed.init_distributed()

    train_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/train.json"

    model, tokenizer = load_model()

    train_dataset = DBpediaDataset(train_path, tokenizer)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    # Train
    train_model(model, tokenizer, train_loader)

    test_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json"
    
    # Load the fine-tuned model before evaluation
    model, tokenizer = load_finetuned_model("llama_finetuned_dbpedia")

    test_dataset = DBpediaDataset(test_path, tokenizer)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=1,
        # collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    # Evaluate
    evaluate(model, tokenizer, test_loader)