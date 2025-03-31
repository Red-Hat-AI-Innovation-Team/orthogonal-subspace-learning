import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os

# Force usage of a single GPU: GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Dataset
class QualityDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data["messages"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        return messages

# Collate function with reduced max_length (optional)
def collate_fn(batch, tokenizer, max_length=2048):
    prompts = [
        tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        for sample in batch
    ]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)

    labels = encodings["input_ids"].clone()

    # Mask only special tokens (e.g., <|endoftext|>, etc.)
    for special_token_id in tokenizer.all_special_ids:
        if special_token_id == 128009:
            continue
        labels[labels == special_token_id] = -100
    
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(t)
        for t in ["<|start_header_id|>", "<|end_header_id|>"]
        if tokenizer.convert_tokens_to_ids(t) is not None
    ]
    
    for token_id in special_token_ids:
        labels[labels == token_id] = -100

    encodings["labels"] = labels
    return encodings


# Load model and tokenizer (modified for one GPU only)
def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ✅ Add a new pad token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.attention_dropout = 0.2
    config.hidden_dropout = 0.2

    # Load model with BF16 and enable gradient checkpointing for memory savings
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
    # ✅ Resize embeddings to account for new token
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()  # Save memory during backpropagation
    model.to("cuda:0")  # Force model to GPU 0
    return model, tokenizer

def load_finetuned_model(model_path="llama_finetuned_dbpedia.pt"):
    """
    Load the fine-tuned LLaMA model from disk.
    """
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ✅ Add a new pad token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.attention_dropout = 0.2
    config.hidden_dropout = 0.2

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.gradient_checkpointing_enable()  # Enable checkpointing here too
    model.to("cuda:0")  # Force model to GPU 0
    model.eval()  # Set to evaluation mode
    print(f"Loaded fine-tuned model from {model_path}")
    return model, tokenizer

# Training function (no accelerate)
def train_model(model, tokenizer, train_loader):
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
    model.train()
    for epoch in range(1):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        for batch in progress_bar:
            # Move batch to model's device (using the first parameter's device)
            first_param_device = next(model.parameters()).device
            for key in batch:
                batch[key] = batch[key].to(first_param_device)

            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with open("loss.txt", "a") as f:  # "a" mode appends to the file
                print(f"Loss: {loss}", file=f)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()  # Optionally clear cache after each batch

        print(f"Epoch finished - Avg Loss: {total_loss / len(train_loader):.4f}")
    
    # Save the trained model
    model_path = "llama_finetuned_dbpedia.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    # Tokenize only the prompt (without the target)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # Generate tokens starting from the prompt
    outputs = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True
    )
    
    # Extract only the newly generated tokens (excluding the input)
    new_token_ids = outputs.sequences[:, input_ids.shape[-1]:]
    
    # Decode only the newly generated tokens
    new_text = tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
    
    return new_text.strip()

def evaluate(model, tokenizer, dataset):
    correct, total = 0, 0
    for inp, tgt in tqdm(dataset, desc="Evaluating"):
        generated_answer = generate_answer(model, tokenizer, inp)
        if generated_answer.lower() == tgt.lower():
            correct += 1
        with open("output.txt", "a") as f:  # "a" mode appends to the file
            print(f"[Target: {tgt.lower()} | Prediction: {generated_answer.lower()}]", file=f)
        total += 1
    print(f"Accuracy: {100.0 * correct / total:.2f}%")

# Main
if __name__ == "__main__":

    train_path = "/new_data/knowledge_rh/quality/training_mix/train_base_extractive_stack.jsonl"

    model, tokenizer = load_model()

    train_dataset = QualityDataset(train_path, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer))

    for batch in train_loader:
        print("Input IDs:", tokenizer.decode(batch['input_ids'][0]))
        print("Labels:", tokenizer.decode([x for x in batch['labels'][0] if x != -100]))
        # print(batch['input_ids'][0])
        # print(batch['labels'][0])
        break

    # Train
    train_model(model, tokenizer, train_loader)
    
    # Load the fine-tuned model before evaluation
    # model, tokenizer = load_finetuned_model("llama_finetuned_dbpedia.pt")

    # Evaluate
    # evaluate(model, tokenizer, train_dataset)