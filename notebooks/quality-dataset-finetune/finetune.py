import os
import json
import csv
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OUTPUT_MODEL_NAME = "llama_finetune_quality"         # Name for the saved model after fine-tuning.

###################################################
# 4. Dataset Construction
###################################################
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

def collate_fn(batch, tokenizer, max_length=2048):
    prompts = [
        tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        for sample in batch
    ]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)

    labels = encodings["input_ids"].clone()

    # # Generation prompt marker (fixed sequence of tokens)
    # generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # marker_ids = tokenizer(generation_prompt, add_special_tokens=False)["input_ids"]
    # marker_len = len(marker_ids)

    # def find_marker_start(sequence, marker_ids):
    #     for i in range(len(sequence) - len(marker_ids) + 1):
    #         if sequence[i:i + len(marker_ids)] == marker_ids:
    #             print("found it", flush=True)
    #             return i
    #     return None

    # # Mask out everything before (and including) the generation prompt marker
    # for i in range(len(labels)):
    #     seq = encodings["input_ids"][i].tolist()
    #     marker_start = find_marker_start(seq, marker_ids)
    #     if marker_start is not None:
    #         labels[i, :marker_start + marker_len] = -100
    #     else:
    #         print('check some issue', flush=True)

    # Mask only special tokens (e.g., <|endoftext|>, etc.)
    for special_token_id in tokenizer.all_special_ids:
        if special_token_id in [32000, 128256]:
            labels[labels == special_token_id] = -100
    
    # special_token_ids = [
    #     tokenizer.convert_tokens_to_ids(t)
    #     for t in ["<|start_header_id|>", "<|end_header_id|>"]
    #     if tokenizer.convert_tokens_to_ids(t) is not None
    # ]
    
    # for token_id in special_token_ids:
    #     labels[labels == token_id] = -100

    encodings["labels"] = labels
    return encodings

###################################################

# 5. Training and Saving the SVD Model on Amazon Reviews
###################################################
def train_model(output_model_name=OUTPUT_MODEL_NAME):

    train_path = "/new_data/knowledge_rh/quality/training_mix/entigraph_knowledge1.0_phi4_first_24_n_5_5_percent.jsonl"

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ✅ Add a new pad token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.use_cache = False  # if applicable for LLaMA; otherwise remove or adjust
    config.attention_dropout = 0.2
    config.hidden_dropout = 0.2

    # Create datasets and dataloaders
    train_dataset = QualityDataset(train_path, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, tokenizer))

    for batch in train_loader:
        print("Input IDs:", tokenizer.decode(batch['input_ids'][0]))
        print("Labels:", tokenizer.decode([x for x in batch['labels'][0] if x != -100]))
        # print(batch['input_ids'][0])
        # print(batch['labels'][0])
        break

    # Load a standard LLaMA model to generate the SVD config.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16
    ).to(device)

    # Initialize our custom SVD model with target_svd_config.
    model.resize_token_embeddings(len(tokenizer))
    # Ensure pad_token_id is correctly set
    model.config.pad_token_id = tokenizer.pad_token_id

    model.gradient_checkpointing_enable()

    optimizer = optim.AdamW(model.parameters(), lr=5e-6, betas=(0.9, 0.999), weight_decay=0.01)
    num_epochs = 1  # adjust as needed

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)
        start_time = time.time()

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}  # ✅ key fix
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with open("loss.txt", "a") as f:  # "a" mode appends to the file
                print(f"Loss: {loss}", file=f)

            total_loss += loss.item()
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (progress_bar.n + 1) * (len(train_loader) - progress_bar.n)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", eta=f"{remaining_time:.2f}s")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        epoch_model_path = f"{output_model_name}_epoch{epoch+1}.pt"
        # Save the fine-tuned model (with SVD modifications)
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved as '{epoch_model_path}'")

        # Clear memory to avoid OOM in next epoch
        torch.cuda.empty_cache()
        del outputs
        del loss

    return model, tokenizer, train_dataset

###################################################
# 7. Main
###################################################
if __name__ == "__main__":

    # Train the model and save it
    train_model(output_model_name=OUTPUT_MODEL_NAME)