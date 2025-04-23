import os
import json
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
from tqdm import tqdm
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OUTPUT_MODEL_NAME = "llama_svd_quality"         # Name for the saved model after fine-tuning.

###################################################
# 1. Helper Functions for SVD and Parameter Management
###################################################

def decompose_weight_matrix(weight: torch.Tensor, top_k: int):
    """
    Perform SVD on a 2D weight matrix and split into:
      - top_k singular vectors (treated as frozen/buffers)
      - the rest (treated as trainable)
    Returns a dictionary containing:
      {
        "U_high": ...  # buffer
        "S_high": ...  # buffer
        "V_high": ...  # buffer
        "U_low": ...   # parameter
        "S_low": ...   # parameter
        "V_low": ...   # parameter
        "rank_high": top_k
      }
    """
    device_local = weight.device
    W = weight.to(torch.float32)  # ensure float32 for SVD
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    # Ensure we don’t ask for more than available
    k = min(top_k, S.shape[0])

    # High subspace (frozen)
    U_high = U[:, :k].detach().to(dtype=torch.bfloat16, device=device_local)
    S_high = S[:k].detach().to(dtype=torch.bfloat16, device=device_local)
    V_high = Vt[:k, :].detach().to(dtype=torch.bfloat16, device=device_local)

    # Low subspace (trainable)
    U_low = U[:, k:].detach().to(dtype=torch.bfloat16, device=device_local)
    S_low = S[k:].detach().to(dtype=torch.bfloat16, device=device_local)
    V_low = Vt[k:, :].detach().to(dtype=torch.bfloat16, device=device_local)

    return {
        "U_high": U_high,
        "S_high": S_high,
        "V_high": V_high,
        "U_low": nn.Parameter(U_low),
        "S_low": nn.Parameter(S_low),
        "V_low": nn.Parameter(V_low),
        "rank_high": k
    }


def reconstruct_weight_matrix(svd_dict):
    """
    Reconstruct the full weight matrix:
       W = U_high * diag(S_high) * V_high^T + U_low * diag(S_low) * V_low^T
    """
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    if U_high.shape[1] > 0 and S_high.shape[0] > 0:
        high_part = torch.mm(U_high * S_high.unsqueeze(0), V_high)
    else:
        high_part = torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)

    if U_low.shape[1] > 0 and S_low.shape[0] > 0:
        US_low = U_low * S_low.unsqueeze(0)
        low_part = torch.mm(US_low, V_low)
    else:
        low_part = torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)

    return high_part + low_part


def check_reconstruction_error(weight, svd_dict, atol=1e-5):
    # Move the weight to the same device as the U_high buffer
    target_device = svd_dict["U_high"].device
    weight = weight.to(target_device)
    W_recon = reconstruct_weight_matrix(svd_dict)
    # Ensure reconstruction is also on the target device
    W_recon = W_recon.to(target_device)
    error = torch.norm(weight - W_recon) / torch.norm(weight)
    if error > atol:
        print(f"Warning: Reconstruction error {error:.2e} exceeds tolerance {atol}")
    return error


def project_gradient_to_orthogonal_space(svd_dict):
    """
    Remove from the gradients of the low subspace any component that lies
    in the high subspace.
    """
    if (svd_dict["U_low"].grad is None and
        svd_dict["S_low"].grad is None and
        svd_dict["V_low"].grad is None):
        return

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]

    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        proj = U_high @ (U_high.transpose(0,1) @ dU)
        dU.sub_(proj)

    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        proj = (dV @ V_high.transpose(0,1)) @ V_high
        dV.sub_(proj)
    # We leave S_low unchanged


###################################################
# 2. LLaMA Model Subclass with SVD (Only for Selected Parameters)
###################################################

class LlamaWithSVD(LlamaForCausalLM):
    """
    Subclass that, on initialization, decomposes selected weight matrices via SVD.
    Only parameters specified in the svd_config are decomposed.
    For each such 2D weight, we freeze the top singular vectors (50% by default)
    and register the lower half (trainable) as parameters.

    Additionally, we pre-compute the module mapping for faster weight injection.
    """
    def __init__(self, config: LlamaConfig, svd_config=None, initialize_svd=True):
        super().__init__(config)
        # svd_config is a dict mapping full parameter names to top_k values.
        self.svd_config = svd_config if svd_config is not None else {}
        self.name_mapping = {}         # maps original name -> safe name
        self.svd_original_mapping = {} # maps safe name -> original name
        self.svd_params = nn.ModuleDict()
        self.svd_module_mapping = {}   # maps safe name -> (module, attribute_name)
        if initialize_svd:
          self._initialize_svd_parameters()

    def reinitialize_svd(self):
        """
        Reinitialize the SVD decomposition on the current (loaded) weights.
        Before reinitialization, store a copy of the original weights for each target parameter,
        then after reinitialization, check and print the reconstruction error.
        """
        # Save original weights for each parameter to be decomposed.
        self._original_weights = {}
        for orig_name in self.svd_config.keys():
            # Retrieve from the model's state_dict; ensure it is on the correct device.
            self._original_weights[orig_name] = self.state_dict()[orig_name].clone().to(device)

        # Clear previous SVD mappings.
        self.name_mapping = {}
        self.svd_original_mapping = {}
        self.svd_params = nn.ModuleDict()
        self.svd_module_mapping = {}
        # Reinitialize the SVD decomposition using the current weights.
        self._initialize_svd_parameters()

        # Now, for each decomposed parameter, compute and print the reconstruction error.
        for orig_name, safe_name in self.name_mapping.items():
            orig_weight = self._original_weights[orig_name]
            svd_dict = {
                "U_high": getattr(self, f"{safe_name}_U_high"),
                "S_high": getattr(self, f"{safe_name}_S_high"),
                "V_high": getattr(self, f"{safe_name}_V_high"),
                "U_low": self.svd_params[safe_name].U_low,
                "S_low": self.svd_params[safe_name].S_low,
                "V_low": self.svd_params[safe_name].V_low
            }
            error = check_reconstruction_error(orig_weight, svd_dict)
            print(f"Reconstruction error for {orig_name}: {error:.2e}")
        
        del self._original_weights
        torch.cuda.empty_cache()

    def _initialize_svd_parameters(self):
        # Iterate over all parameters
        for name, param in list(self.named_parameters()):
            if len(param.shape) == 2 and name in self.svd_config and self.svd_config[name] > 0:
                top_k = self.svd_config[name]
                print(f"[SVD Init] Decomposing {name} with top_k={top_k}")
                svd_dict = decompose_weight_matrix(param.data, top_k=top_k)
                safe_name = name.replace(".", "_")
                self.name_mapping[name] = safe_name
                self.svd_original_mapping[safe_name] = name

                # Register buffers for the high subspace
                self.register_buffer(f"{safe_name}_U_high", svd_dict["U_high"])
                self.register_buffer(f"{safe_name}_S_high", svd_dict["S_high"])
                self.register_buffer(f"{safe_name}_V_high", svd_dict["V_high"])

                # Create a module to hold the low subspace trainable parameters
                module_svd = nn.Module()
                module_svd.U_low = nn.Parameter(svd_dict["U_low"])
                module_svd.S_low = nn.Parameter(svd_dict["S_low"])
                module_svd.V_low = nn.Parameter(svd_dict["V_low"])
                module_svd.rank_high = svd_dict["rank_high"]
                module_svd.safe_name = safe_name
                self.svd_params[safe_name] = module_svd

                # Pre-compute and store the module and attribute name for quick access
                mod, attr = self._get_module_by_name(name)

                bias = mod.bias
                def make_svd_forward(safe_name, bias):
                    def forward(x, *args, **kwargs):
                        orig = self.svd_original_mapping[safe_name]
                        W = self._reconstruct_weight(orig)
                        return F.linear(x, W, bias)
                    return forward
                mod.forward = make_svd_forward(safe_name, bias)

                # finally, freeze the original weight so only U_low/S_low/V_low train
                param.requires_grad = False

                mod._parameters.pop(attr, None)

                # # Freeze the original parameter
                # param.requires_grad = False

                # # Pre-compute and store the module and attribute name for quick access
                # mod, attr = self._get_module_by_name(name)
                # if mod is not None:
                #     self.svd_module_mapping[safe_name] = (mod, attr)
            # For parameters not in svd_config, leave them trainable (do nothing)


    def _reconstruct_weight(self, original_name):
        safe_name = self.name_mapping[original_name]
        U_high = getattr(self, f"{safe_name}_U_high")
        S_high = getattr(self, f"{safe_name}_S_high")
        V_high = getattr(self, f"{safe_name}_V_high")
        module_svd = self.svd_params[safe_name]
        U_low = module_svd.U_low
        S_low = module_svd.S_low
        V_low = module_svd.V_low
        svd_dict = {"U_high": U_high, "S_high": S_high, "V_high": V_high,
                    "U_low": U_low, "S_low": S_low, "V_low": V_low}
        W = reconstruct_weight_matrix(svd_dict)

        return W

    # def forward(self, *args, **kwargs):
    #     # Instead of recomputing the module mapping each time,
    #     # iterate over the precomputed svd_module_mapping.
    #     for safe_name, (module, attr) in self.svd_module_mapping.items():
    #         original_name = self.svd_original_mapping[safe_name]
    #         W = self._reconstruct_weight(original_name)
    #         # print(module._parameters)
    #         with torch.no_grad():
    #             getattr(module, attr).data.copy_(W)
    #         # assert W.requires_grad, f"W for {safe_name} lost grad"
    #         # module._parameters[attr] = W
    #     return super().forward(*args, **kwargs)

    def _get_module_by_name(self, name):
        """
        Given a full parameter name (e.g. "encoder.block.0.layer.0.SelfAttention.q.weight"),
        return (module, attribute_name) where module.attribute_name is that parameter.
        """
        parts = name.split(".")
        attr = parts[-1]
        mod = self
        for p in parts[:-1]:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            elif p.isdigit():
                mod = mod[int(p)]
            else:
                return None, None
        return mod, attr

    def project_gradients(self):
        for safe_name, module_svd in self.svd_params.items():
            svd_dict = {
                "U_high": getattr(self, f"{safe_name}_U_high"),
                "S_high": getattr(self, f"{safe_name}_S_high"),
                "V_high": getattr(self, f"{safe_name}_V_high"),
                "U_low": module_svd.U_low,
                "S_low": module_svd.S_low,
                "V_low": module_svd.V_low,
            }
            project_gradient_to_orthogonal_space(svd_dict)

###################################################
# 3. Utility: Auto-generate SVD Config for Target Parameters
###################################################
def auto_generate_target_svd_config(model):
    """
    Given a model, generate an SVD configuration dictionary only for parameters that contain one of the
    following substrings:
      - self_attn.q_proj
      - self_attn.k_proj
      - self_attn.v_proj
      - self_attn.o_proj
      - mlp.gate_proj
      - mlp.down_proj
      - mlp.up_proj
    For each such 2D parameter, set:
         top_k = floor(min(dim0, dim1) / 2)
    """
    target_patterns = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        # "self_attn.v_proj",
        # "self_attn.o_proj",
        # "mlp.gate_proj",
        # "mlp.down_proj",
        # "mlp.up_proj"
    ]
    config = {}
    for name, param in model.named_parameters():
        if any(pat in name for pat in target_patterns) and len(param.shape) == 2:
            top_k = int(np.floor(min(param.shape)*0.20))
            full_rank = min(param.shape)
            if top_k >= full_rank:
                top_k = full_rank - 1
            config[name] = top_k
    # save_svd_config(config)
    return config

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
def train_svd_model(output_model_name=OUTPUT_MODEL_NAME):

    train_path = "/new_data/knowledge_rh/quality/training_mix/entigraph_knowledge1.0_phi4_first_24_n_5_5_percent.jsonl"

    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config
    )
    base_model = base_model.to(device)
    target_svd_config = auto_generate_target_svd_config(base_model)
    print("Auto-generated target SVD config:")
    for k, v in target_svd_config.items():
        print(f"  {k}: freeze top {v} singular vectors")

    del base_model
    torch.cuda.empty_cache()

    # Initialize our custom SVD model with target_svd_config.
    model = LlamaWithSVD.from_pretrained(model_name, config=config, svd_config=target_svd_config, initialize_svd=False, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    # Ensure pad_token_id is correctly set
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move the model to the local device.
    model = model.to(device, dtype=torch.bfloat16)

    model.reinitialize_svd()

    model.gradient_checkpointing_enable()

    optimizer = optim.AdamW(model.parameters(), lr=5e-6, betas=(0.9, 0.999), weight_decay=0.01)
    num_epochs = 1  # adjust as needed

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)
        start_time = time.time()

        # initial_low = {}
        # for safe_name, module_svd in model.svd_params.items():
        #     initial_low[safe_name] = {
        #         "U_low": module_svd.U_low.detach().clone(),
        #         "S_low": module_svd.S_low.detach().clone(),
        #         "V_low": module_svd.V_low.detach().clone(),
        #     }
        
        # initial_high = {
        # safe_name: {
        #     "U_high": getattr(model, f"{safe_name}_U_high").detach().clone(),
        #     "S_high": getattr(model, f"{safe_name}_S_high").detach().clone(),
        #     "V_high": getattr(model, f"{safe_name}_V_high").detach().clone(),
        # }
        # for safe_name in model.svd_params
        # }

        for batch in progress_bar:
            for key, val in batch.items():
                batch[key] = val.to(device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            model.project_gradients()  # ensure gradients remain in correct subspace
            optimizer.step()

            with open("loss.txt", "a") as f:  # "a" mode appends to the file
                print(f"Loss: {loss}", file=f)

            total_loss += loss.item()
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (progress_bar.n + 1) * (len(train_loader) - progress_bar.n)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", eta=f"{remaining_time:.2f}s")

            # Clear memory to avoid OOM in next epoch
            torch.cuda.empty_cache()
            del outputs
            del loss
        
        # print("\nChecking low‑rank parameter equality after 1 epoch:")
        # for safe_name, module_svd in model.svd_params.items():
        #     u_eq = torch.allclose(initial_low[safe_name]["U_low"], module_svd.U_low, atol=1e-6)
        #     s_eq = torch.allclose(initial_low[safe_name]["S_low"], module_svd.S_low, atol=1e-6)
        #     v_eq = torch.allclose(initial_low[safe_name]["V_low"], module_svd.V_low, atol=1e-6)
        #     print(f"{safe_name}: U_low_equal={u_eq}, S_low_equal={s_eq}, V_low_equal={v_eq}")
        
        # print("\nChecking high-rank parameter changes after 1 epoch:")
        # for safe_name, module_svd in model.svd_params.items():
        #     U_high   = getattr(model, f"{safe_name}_U_high")
        #     S_high   = getattr(model, f"{safe_name}_S_high")
        #     V_high   = getattr(model, f"{safe_name}_V_high")
        #     uh_eq = torch.allclose(initial_high[safe_name]["U_high"], U_high, atol=1e-6)
        #     sh_eq = torch.allclose(initial_high[safe_name]["S_high"], S_high, atol=1e-6)
        #     vh_eq = torch.allclose(initial_high[safe_name]["V_high"], V_high, atol=1e-6)
        #     print(f"{safe_name}: U_high_equal={uh_eq}, S_high_equal={sh_eq}, V_high_equal={vh_eq}")

        # print("\nChecking orthogonality between low-rank and high-rank subspaces:")
        # for safe_name, module_svd in model.svd_params.items():
        #     U_high = getattr(model, f"{safe_name}_U_high")
        #     V_high = getattr(model, f"{safe_name}_V_high")
        #     U_low  = module_svd.U_low
        #     V_low  = module_svd.V_low
        #     ortho_U = torch.norm(U_high.T @ U_low).item()
        #     ortho_V = torch.norm(V_low @ V_high.T).item()
        #     print(f"{safe_name}: ||U_high^T U_low||={ortho_U:.2e}, ||V_low V_high^T||={ortho_V:.2e}")
        
        # print("\nChecking orthogonality between own high-rank subspaces:")
        # for safe_name, module_svd in model.svd_params.items():
        #     U_high = getattr(model, f"{safe_name}_U_high")
        #     V_high = getattr(model, f"{safe_name}_V_high")
        #     U_low  = module_svd.U_low
        #     V_low  = module_svd.V_low
        #     ortho_U = torch.norm(U_high.T @ U_high).item()
        #     ortho_V = torch.norm(V_high @ V_high.T).item()
        #     print(f"{safe_name}: ||U_high^T U_high||={ortho_U:.2e}, ||V_high V_high^T||={ortho_V:.2e}")
        
        # print("\nChecking orthogonality between own low-rank subspaces:")
        # for safe_name, module_svd in model.svd_params.items():
        #     U_high = getattr(model, f"{safe_name}_U_high")
        #     V_high = getattr(model, f"{safe_name}_V_high")
        #     U_low  = module_svd.U_low
        #     V_low  = module_svd.V_low
        #     ortho_U = torch.norm(U_low.T @ U_low).item()
        #     ortho_V = torch.norm(V_low @ V_low.T).item()
        #     print(f"{safe_name}: ||U_low^T U_low||={ortho_U:.2e}, ||V_low V_low^T||={ortho_V:.2e}")

        # print("\nParameter gradient status before saving:")
        # for name, param in model.named_parameters():
        #     has_grad = param.grad is not None
        #     print(f"{name:60} requires_grad={param.requires_grad:<5} grad_present={has_grad}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        epoch_model_path = f"{output_model_name}_epoch{epoch+1}.pt"

        # Save the fine-tuned model (with SVD modifications)
        # 1) Build a fresh CPU‐only dict   
        cpu_sd = {}
        with torch.no_grad():
            # Copy all existing entries (buffers, low‐rank params, etc.) to CPU
            for name, tensor in model.state_dict().items():
                cpu_sd[name] = tensor.cpu()
            # Reconstruct and insert the “popped” full weights under their original keys
            for orig_name in model.name_mapping:
                W = model._reconstruct_weight(orig_name).cpu()
                cpu_sd[orig_name] = W
        # 2) Save the CPU dict
        torch.save(cpu_sd, epoch_model_path)
        print(f"Model saved as '{epoch_model_path}'")
        torch.cuda.empty_cache()

    return model, tokenizer, train_dataset

###################################################
# 7. Main
###################################################
if __name__ == "__main__":

    # Train the model and save it
    train_svd_model(output_model_name=OUTPUT_MODEL_NAME)