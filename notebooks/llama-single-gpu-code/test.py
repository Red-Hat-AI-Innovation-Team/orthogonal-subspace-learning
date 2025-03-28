import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from tqdm import tqdm
import json
import os

# Force usage of a single GPU: GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            "Task: TC\n"
            "Dataset: dbpedia\n"
            "What is the topic of the following paragraph? Choose one from the option.\n"
            "Option: Company, Educational Institution, Artist, Athlete, Office Holder, "
            "Mean of Transportation, Building, Natural Place, Village, Animal, Plant, "
            "Album, Film, Written Work\n" + sample['sentence'] + "Assistant:"

            # "Task: SC\n"
            # "Dataset: amazon\n"
            # "What is the sentiment of the following paragraph? Choose one from the option.\n"
            # "Option: very negative, negative, neutral, positive, very positive\n" + sample['sentence'] + "Sentiment:"

            # "Task: TC\n"
            # "Dataset: yahoo\n"
            # "What is the topic of the following paragraph? Choose one from the option.\n"
            # "Option: Society & Culture, Science & Mathematics, Health, Education & Reference, "
            # "Computers & Internet, Sports, Business & Finance, Entertainment & Music, "
            # "Family & Relationships, Politics & Government\n" + sample['sentence'] + "Assistant:"

            # "Task: TC\n"
            # "Dataset: ag news\n"
            # "What is the topic of the following paragraph? Choose one from the option.\n"
            # "Option: World, Sports, Business, Science or Technology\n" + sample['sentence'] + "Assistant:"

        )
        target_text = sample["label"]
        return input_text, target_text

# Collate function with reduced max_length (optional)
def collate_fn(batch, tokenizer, max_length=1024):
    inputs, targets = zip(*batch)
    # Create full texts: prompt + " " + target
    full_texts = [inp + " " + tgt + tokenizer.eos_token for inp, tgt in zip(inputs, targets)]
    encodings = tokenizer(
        full_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    
    # Now create labels, but mask out the prompt tokens.
    # First, get the tokenized version of just the prompt.
    prompt_texts = [inp for inp in inputs]
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
    labels[encodings["attention_mask"] == 0] = -100
    encodings["labels"] = labels
    return encodings

# Load model and tokenizer (modified for one GPU only)
def load_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.attention_dropout = 0.2
    config.hidden_dropout = 0.2

    # Load model with BF16 and enable gradient checkpointing for memory savings
    model = LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()  # Save memory during backpropagation
    model.to("cuda:0")  # Force model to GPU 0
    return model, tokenizer

def load_finetuned_model(model_path="llama_finetuned_dbpedia.pt"):
    """
    Load the fine-tuned LLaMA model from disk.
    """
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.attention_dropout = 0.2
    config.hidden_dropout = 0.2

    model = LlamaForCausalLM.from_pretrained(model_name)
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
    train_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/train.json"
    test_path = "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json"

    model, tokenizer = load_model()

    train_dataset = DBpediaDataset(train_path, tokenizer)
    test_dataset = DBpediaDataset(test_path, tokenizer)

    # Reduce batch size to further alleviate OOM issues (from 8 to 2)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

    # for batch in train_loader:
    #     print("Input IDs:", tokenizer.decode(batch['input_ids'][0]))
    #     print("Labels:", tokenizer.decode([x for x in batch['labels'][0] if x != -100]))
    #     # print(batch['input_ids'][0])
    #     # print(batch['labels'][0])
    #     break

    # Train
    train_model(model, tokenizer, train_loader)
    
    # Load the fine-tuned model before evaluation
    model, tokenizer = load_finetuned_model("llama_finetuned_dbpedia.pt")

    # Evaluate
    evaluate(model, tokenizer, test_dataset)