# Configurable parameters:
SOURCE_SVD_DATASET = "dbpedia"       # Dataset to use when computing the adaptive SVD config.
FINE_TUNE_DATASET = "amazon"       # Fine-tuning dataset; options: "agnews", "amazon", "yelp", "dbpedia", "yahoo"
STARTING_CHECKPOINT = "llama_finetuned_dbpedia"  # Path to the checkpoint you want to start from.
OUTPUT_MODEL_NAME = "llama_svd_amazon"         # Name for the saved model after fine-tuning.

import os
import json
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AutoModelForCausalLM

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_config = {
    "train_micro_batch_size_per_gpu": 1,  # Keep your micro-batch size
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {         # Offload optimizer states to CPU to save GPU memory
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,          # Overlap communication with computation (may help memory usage)
        "contiguous_gradients": True,  # Use contiguous gradients to further reduce memory fragmentation
        "reduce_bucket_size": 50000000,       # Reduce bucket sizes (adjust value as needed)
        "allgather_bucket_size": 50000000     # Adjust allgather bucket size for memory savings
    },
    "gradient_checkpointing": True,
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


####################################################################
# Define Dataset Information
####################################################################
DATASET_INFOS = {
    "dbpedia": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json",
        "label_mapping": {0: "Company", 1: "Educational Institution", 2: "Artist",
                           3: "Athlete", 4: "Office Holder", 5: "Mean of Transportation",
                           6: "Building", 7: "Natural Place", 8: "Village",
                           9: "Animal", 10: "Plant", 11: "Album", 12: "Film", 13: "Written Work"}
    },
    "amazon": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/amazon/test.json",
        "label_mapping": {0: "negative", 1: "positive", 2: "neutral", 3: "very positive", 4: "very negative"}
    },
    "yahoo": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/TC/yahoo/test.json",
        "label_mapping": {0: "Society & Culture", 1: "Science & Mathematics", 2: "Health", 3: "Education & Reference",
                           4: "Computers & Internet", 5: "Sports", 6: "Business & Finance", 7: "Entertainment & Music",
                           8: "Family & Relationships", 9: "Politics & Government"}
    },
    "agnews": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/TC/agnews/test.json",
        "label_mapping": {0: "World", 1: "Sports", 2: "Business", 3: "Science or Technology"}
    },
    # "yelp": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/yelp/test.json",
    #     "label_mapping": {0: "negative", 1: "positive", 2: "neutral", 3: "very positive", 4: "very negative"}
    # },
    # "mnli": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/NLI/MNLI/test.json",
    #     "label_mapping": {0: "entailment", 1: "neutral", 2: "contradiction"}
    # }, 
    # "qqp": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/QQP/QQP/test.json",
    #     "label_mapping": {0: "True", 1: "False"}
    # },
    # "rte": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/NLI/RTE/test.json",
    #     "label_mapping": {0: "entailment", 1: "contradiction"}
    # },
    # "sst-2": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/SST-2/test.json",
    #     "label_mapping": {0: "Bad", 1: "Good"}
    # },
    # "wic": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/WiC/WiC/test.json",
    #     "label_mapping": {0: "True", 1: "False"}
    # },
    # "cb": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/NLI/CB/test.json",
    #     "label_mapping": {0: "contradiction", 1: "entailment", 2: "neutral"}
    # },
    # "copa": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/COPA/COPA/test.json",
    #     "label_mapping": {0: "A", 1: "B"}
    # },
    # "multirc": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/MultiRC/MultiRC/test.json",
    #     "label_mapping": {0: "True", 1: "False"}  
    # },
    # "boolqa": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/BoolQA/BoolQA/test.json",
    #     "label_mapping": {0: "True", 1: "False"}
    # },
    # "imdb": {
    #     "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/IMDB/test.json",
    #     "label_mapping": {0: "Bad", 1: "Good"}
    # }
}

def construct_prompt(sample, dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "agnews":
        return (
            "Classify the following text into one of these categories: "
            "[World, Sports, Business, Science or Technology].\n\n"
            "Text: " + sample["text"] + "\nAnswer:"
        )
    elif dataset_name in ["amazon", "yelp"]:
        return (
            "Classify the sentiment of the following text into one of these categories: "
            "[very negative, negative, neutral, positive, very positive].\n\n"
            "Text: " + sample["text"] + "\nAnswer:"
        )
    elif dataset_name == "dbpedia":
        return (
            "Classify the following text into one of these categories: "
            "[Company, Educational Institution, Artist, Athlete, Office Holder, "
            "Mean of Transportation, Building, Natural Place, Village, Animal, "
            "Plant, Album, Film, Written Work].\n\n"
            "Text: " + sample["text"] + "\nAnswer:"
        )
    elif dataset_name == "yahoo":
        return (
            "Classify the following text into one of these categories: "
            "[Sports, Entertainment & Music, Health, Education & Reference, "
            "Family & Relationships, Politics & Government, Science & Mathematics, "
            "Business & Finance, Computers & Internet, Society & Culture].\n\n"
            "Text: " + sample["text"] + "\nAnswer:"
        )
    elif dataset_name == "mnli":
        # return "classify mnli dataset: premise: " + sample["premise"] + " hypothesis: " + sample["hypothesis"]
        return (
            "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the following options [neutral, entailment, contradiction]. "
            + sample["text"]
        )
    elif dataset_name == "qqp":
        # return "classify qqp dataset: question1: " + sample["question1"] + " question2: " + sample["question2"]
        return (
            "Whether the 'first sentence' and the 'second sentence' have the same meaning? "
            "Choose one from the following options [True, False]. "
            + sample["text"]
        )
    elif dataset_name == "rte":
        # return "classify rte dataset: sentence1: " + sample["sentence1"] + " sentence2: " + sample["sentence2"]
        return (
            "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? "
            "Choose one from the following options [entailment, contradiction]. "
            + sample["text"]
        )
    elif dataset_name == "sst-2":
        # return "classify sst2 dataset: sentence: " + sample["sentence"]
        return (
            "What is the sentiment of the following paragraph? "
            "Choose one from the following options [Bad, Good]. "
            + sample["text"]
        )
    elif dataset_name == "wic":
        # return "classify wic dataset: word: " + sample["word"] + " sentence1: " + sample["sentence1"] + " sentence2: " + sample["sentence2"]
        return (
            "Given a word and two sentences, whether the word is used with the same sense in both sentence? "
            "Choose one from the following options [True, False]. "
            + sample["text"]
        )
    elif dataset_name == "cb":
        # return "classify cb dataset: premise: " + sample["premise"] + " hypothesis: " + sample["hypothesis"]
        return (
            "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? "
            "Choose one from the following options [neutral, entailment, contradiction]. "
            + sample["text"]
        )
    elif dataset_name == "copa":
        # return "classify copa dataset: premise: " + sample["premise"] + " choice1: " + sample["choice1"] + " choice2: " + sample["choice2"]
        return (
            "Which choice best explains or follows from the given premise? "
            "Choose one from the following options [A, B]. "
            + sample["text"]
        )
    elif dataset_name == "multirc":
        # return "classify multirc dataset: question: " + sample["question"] + " passage: " + sample["passage"]
        return (
            "According to the following passage and question, is the candidate answer true or false? "
            "Choose one from the following options [True, False]. "
            + sample["text"]
        )
    elif dataset_name == "boolqa":
        # return "classify boolq dataset: question: " + sample["question"] + " passage: " + sample["passage"]
        return (
            "According to the following passage, is the question true or false? "
            "Choose one from the following options [True, False]. "
            + sample["text"]
        )
    elif dataset_name == "imdb":
        # return "classify imdb dataset: " + sample["text"]
        return (
            "What is the sentiment of the following paragraph? "
            "Choose one from the following options [Bad, Good]. "
            + sample["text"]
        )
    else:
        return "classify dataset: " + sample.get("text", sample.get("content", ""))

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
    U_high = U[:, :k].detach().to(device_local)
    S_high = S[:k].detach().to(device_local)
    V_high = Vt[:k, :].detach().to(device_local)

    # Low subspace (trainable)
    U_low = U[:, k:].detach().to(device_local)
    S_low = S[k:].detach().to(device_local)
    V_low = Vt[k:, :].detach().to(device_local)

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


def compute_effective_rank(matrix):
    """
    Compute the effective rank of a matrix based on the definition provided.
    """
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    singular_values = S.cpu().numpy()

    # Compute the singular value distribution (p_k)
    l1_norm = np.sum(np.abs(singular_values))
    p_k = singular_values / l1_norm

    # Compute the Shannon entropy
    H = -np.sum(p_k * np.log(p_k + 1e-10))  # Add a small constant to avoid log(0)

    # Compute the effective rank
    effective_rank = np.exp(H)

    return effective_rank


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
        # # Save original weights for each parameter to be decomposed.
        # self._original_weights = {}
        # for orig_name in self.svd_config.keys():
        #     # Retrieve from the model's state_dict; ensure it is on the correct device.
        #     self._original_weights[orig_name] = self.state_dict()[orig_name].clone().to(device)

        # Clear previous SVD mappings.
        self.name_mapping = {}
        self.svd_original_mapping = {}
        self.svd_params = nn.ModuleDict()
        self.svd_module_mapping = {}
        # Reinitialize the SVD decomposition using the current weights.
        self._initialize_svd_parameters()

        # # Now, for each decomposed parameter, compute and print the reconstruction error.
        # for orig_name, safe_name in self.name_mapping.items():
        #     orig_weight = self._original_weights[orig_name]
        #     svd_dict = {
        #         "U_high": getattr(self, f"{safe_name}_U_high"),
        #         "S_high": getattr(self, f"{safe_name}_S_high"),
        #         "V_high": getattr(self, f"{safe_name}_V_high"),
        #         "U_low": self.svd_params[safe_name].U_low,
        #         "S_low": self.svd_params[safe_name].S_low,
        #         "V_low": self.svd_params[safe_name].V_low
        #     }
        #     error = check_reconstruction_error(orig_weight, svd_dict)
        #     print(f"Reconstruction error for {orig_name}: {error:.2e}")

    def _initialize_svd_parameters(self):
        # Iterate over all parameters
        for name, param in list(self.named_parameters()):
            if len(param.shape) == 2 and name in self.svd_config and self.svd_config[name] > 0:
                top_k = self.svd_config[name]
                print(f"[SVD Init] Decomposing {name} with top_k={top_k}")

                # Move only the parameter data to GPU temporarily
                param_gpu = param.data.to("cuda", non_blocking=True)  # ensure float32 for SVD
                svd_dict = decompose_weight_matrix(param_gpu, top_k=top_k)
                # Move results back to CPU and free GPU memory
                svd_dict = {
                    key: (svd_dict[key] if key == "rank_high" else svd_dict[key].to("cpu", non_blocking=True))
                    for key in svd_dict
                }
                del param_gpu
                torch.cuda.empty_cache()

                # svd_dict = decompose_weight_matrix(param.data, top_k=top_k)

                safe_name = name.replace(".", "_")
                self.name_mapping[name] = safe_name
                self.svd_original_mapping[safe_name] = name

                # Compute the residual: the difference between the original weight and its SVD reconstruction.
                # residual = (param.data - reconstruct_weight_matrix(svd_dict)).detach()
                # Register the residual as a buffer (no gradients).
                # self.register_buffer(f"{safe_name}_residual", residual)

                # Register buffers for the high subspace
                self.register_buffer(f"{safe_name}_U_high", svd_dict["U_high"])
                self.register_buffer(f"{safe_name}_S_high", svd_dict["S_high"])
                self.register_buffer(f"{safe_name}_V_high", svd_dict["V_high"])

                # Create a module to hold the low subspace trainable parameters
                module_svd = nn.Module()
                module_svd.U_low = nn.Parameter(svd_dict["U_low"].contiguous())
                module_svd.S_low = nn.Parameter(svd_dict["S_low"].contiguous())
                module_svd.V_low = nn.Parameter(svd_dict["V_low"].contiguous())
                module_svd.rank_high = svd_dict["rank_high"]
                module_svd.safe_name = safe_name
                self.svd_params[safe_name] = module_svd

                # Freeze the original parameter
                param.requires_grad = False

                # Pre-compute and store the module and attribute name for quick access
                mod, attr = self._get_module_by_name(name)
                if mod is not None:
                    self.svd_module_mapping[safe_name] = (mod, attr)
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

        # Retrieve the residual that was stored during initialization.
        # residual = getattr(self, f"{safe_name}_residual").detach()

        # return W + residual

        return W

    def forward(self, *args, **kwargs):
        # Instead of recomputing the module mapping each time,
        # iterate over the precomputed svd_module_mapping.
        for safe_name, (module, attr) in self.svd_module_mapping.items():
            original_name = self.svd_original_mapping[safe_name]
            W = self._reconstruct_weight(original_name)
            # if attr in module._parameters:
            #     print(type(module._parameters))
            #     print(module._parameters)
            #     print(attr)
            #     module._parameters.pop(attr)
            # setattr(module, attr, W)
            # print(module._parameters)
            with torch.no_grad():
                getattr(module, attr).data.copy_(W)
        return super().forward(*args, **kwargs)

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
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj"
    ]
    config = {}
    for name, param in model.named_parameters():
        if any(pat in name for pat in target_patterns) and len(param.shape) == 2:
            # effective_rank = compute_effective_rank(param.data)
            # top_k = int(np.floor(effective_rank))
            # full_rank = min(param.shape)
            # if top_k > full_rank:
            #     top_k = full_rank
            # config[name] = top_k
            top_k = int(np.floor(max(param.shape)*0.25))
            full_rank = min(param.shape)
            if top_k >= full_rank:
                top_k = full_rank - 1
            elif top_k <= 0:
                continue  # skip because there's no "high" subspace
            config[name] = top_k
    # save_svd_config(config)
    return config

###################################################
# 4. Dataset Construction
###################################################
class GenericClassificationDataset(Dataset):
    """
    A generic dataset that works for multiple classification datasets.
    Expects the HF dataset to have either "text" or "content" as the input field.
    The prompt is constructed as "classify {dataset_name} dataset: <input>"
    """
    def __init__(self, json_file, tokenizer, label_mapping, dataset_name):
        # self.dataset = hf_dataset[split].shuffle(seed=42).select(range(3600))

        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.dataset_name = dataset_name

        # Load data from JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        # Ensure full dataset is used
        self.dataset = [
            {"text": sample["sentence"], "label": str(sample["label"])}  # Store labels as strings
            for sample in self.dataset
        ]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        input_text = construct_prompt(sample, self.dataset_name)
        return input_text, sample["label"]

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

###################################################

# 5. Training and Saving the SVD Model on Amazon Reviews
###################################################
def train_svd_model(fine_tune_dataset=FINE_TUNE_DATASET, starting_checkpoint=STARTING_CHECKPOINT, output_model_name=OUTPUT_MODEL_NAME):

    if fine_tune_dataset in ["agnews", "dbpedia", "yahoo"]:
        train_json_path = f"/workspace/O-LoRA/CL_Benchmark/TC/{fine_tune_dataset}/train.json"
        test_json_path = f"/workspace/O-LoRA/CL_Benchmark/TC/{fine_tune_dataset}/test.json"

    elif fine_tune_dataset in ["amazon", "yelp", "SST-2", "IMDB"]:
        train_json_path = f"/workspace/O-LoRA/CL_Benchmark/SC/{fine_tune_dataset}/train.json"
        test_json_path = f"/workspace/O-LoRA/CL_Benchmark/SC/{fine_tune_dataset}/test.json"
    
    elif fine_tune_dataset in ["QQP", "WiC", "MultiRC", "COPA", "BoolQA"]:
        train_json_path = f"/workspace/O-LoRA/CL_Benchmark/{fine_tune_dataset}/{fine_tune_dataset}/train.json"
        test_json_path = f"/workspace/O-LoRA/CL_Benchmark/{fine_tune_dataset}/{fine_tune_dataset}/test.json"
    
    elif fine_tune_dataset in ["CB", "MNLI", "RTE"]:
        train_json_path = f"/workspace/O-LoRA/CL_Benchmark/NLI/{fine_tune_dataset}/train.json"
        test_json_path = f"/workspace/O-LoRA/CL_Benchmark/NLI/{fine_tune_dataset}/test.json"

    if fine_tune_dataset.lower() == "agnews":
        label_mapping = {"World": 0, "Sports": 1, "Business": 2, "Science or Technology": 3}
    elif fine_tune_dataset.lower() in ["amazon", "yelp"]:
        label_mapping = {"negative": 0, "positive": 1, "neutral": 2, "very positive": 3, "very negative": 4}
    elif fine_tune_dataset.lower() == "dbpedia":
        label_mapping = {"Company": 0, "Educational Institution": 1, "Artist": 2, "Athlete": 3,
            "Office Holder": 4, "Mean of Transportation": 5, "Building": 6, "Natural Place": 7,
            "Village": 8, "Animal": 9, "Plant": 10, "Album": 11, "Film": 12, "Written Work": 13}
    elif fine_tune_dataset.lower() == "yahoo":
        label_mapping = {"Society & Culture": 0, "Science & Mathematics": 1, "Health": 2, "Education & Reference": 3, "Computers & Internet": 4,
            "Sports": 5, "Business & Finance": 6, "Entertainment & Music": 7, "Family & Relationships": 8, "Politics & Government": 9}
    elif fine_tune_dataset.lower() == "mnli":
        label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    elif fine_tune_dataset.lower() == "qqp":
        label_mapping = {0: "True", 1: "False"}
    elif fine_tune_dataset.lower() == "rte":
        label_mapping = {0: "entailment", 1: "contradiction"}
    elif fine_tune_dataset.lower() == "sst2":
        label_mapping = {0: "Bad", 1: "Good"}
    elif fine_tune_dataset.lower() == "wic":
        label_mapping = {0: "True", 1: "False"}
    elif fine_tune_dataset.lower() == "cb":
        label_mapping = {0: "contradiction", 1: "entailment", 2: "neutral"}
    elif fine_tune_dataset.lower() == "copa":
        label_mapping = {0: "A", 1: "B"}
    elif fine_tune_dataset.lower() == "multirc":
        label_mapping = {0: "True", 1: "False"}  
    elif fine_tune_dataset.lower() == "boolqa":
        label_mapping = {0: "True", 1: "False"}
    elif fine_tune_dataset.lower() == "imdb":
        label_mapping = {0: "Bad", 1: "Good"}
    else:
        raise ValueError(f"Unknown fine-tune dataset: {fine_tune_dataset}")
    
    # Use a prompt that indicates the dataset.
    dataset_prompt = fine_tune_dataset.lower()  # e.g., "dbpedia"

    model_name = "baffo32/decapoda-research-llama-7B-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = LlamaConfig.from_pretrained(model_name)
    config.use_cache = False  # if applicable for LLaMA; otherwise remove or adjust

    # Create datasets and dataloaders
    train_dataset = GenericClassificationDataset(train_json_path, tokenizer, label_mapping, dataset_prompt)
    test_dataset = GenericClassificationDataset(test_json_path, tokenizer, label_mapping, dataset_prompt)

    # Load a base LLaMA model to auto-generate the target SVD config.
    base_model = LlamaWithSVD(config, svd_config={}, initialize_svd=False)
    base_model.gradient_checkpointing_enable()
    target_svd_config = auto_generate_target_svd_config(base_model)
    print("Auto-generated target SVD config:")
    for k, v in target_svd_config.items():
        print(f"  {k}: freeze top {v} singular vectors")

    # Initialize our custom SVD model with target_svd_config.
    base_model.svd_config = target_svd_config
    # Load pretrained weights into our SVD model.
    base_model.load_state_dict(AutoModelForCausalLM.from_pretrained("/workspace/orthogonal-subspace/notebooks/llama_finetuned_dbpedia/converted_model").state_dict(), strict=False)
    base_model.reinitialize_svd()

    # Use a distributed sampler for training
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]

    # Prepare model for DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=base_model,
        config=ds_config
    )

    # model_engine.module.reinitialize_svd()

    # optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 1  # adjust as needed

    model_engine.train()
    for epoch in range(num_epochs):
        # Important: set the epoch on sampler for correct shuffling across epochs
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)
        start_time = time.time()

        for batch in progress_bar:
            for key, val in batch.items():
                batch[key] = val.to(model_engine.device)
            outputs = model_engine(**batch, use_cache=False)
            loss = outputs.loss

            with open("loss.txt", "a") as f:  # "a" mode appends to the file
                print(f"Loss: {loss}", file=f)

            model_engine.zero_grad()
            model_engine.backward(loss)
            model_engine.module.project_gradients()  # if your model uses project_gradients
            model_engine.step()

            total_loss += loss.item()
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (progress_bar.n + 1) * (len(train_loader) - progress_bar.n)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", eta=f"{remaining_time:.2f}s")

            del outputs
            del loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model (with SVD modifications)
    # torch.save(model.state_dict(), output_model_name)
    torch.cuda.empty_cache()
    model_engine.save_checkpoint(output_model_name)
    print(f"Model saved as '{output_model_name}'")
    return model_engine, tokenizer, train_dataset, test_dataset

def generate_answer(model, tokenizer, prompt, max_new_tokens=8):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True
    )
    new_token_ids = outputs.sequences[:, input_ids.shape[-1]:]
    return tokenizer.decode(new_token_ids[0], skip_special_tokens=True).strip()

def evaluate(model, tokenizer, dataset):
    # Create a DistributedSampler so each GPU processes a distinct part of the dataset.
    sampler = DistributedSampler(dataset, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=lambda batch: zip(*batch))

    local_correct = 0
    local_total = 0
    sample_print_count = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating", disable=(dist.get_rank() != 0)):
            # With batch_size=1, extract the single prompt and target.
            prompt = list(inputs)[0] + " "
            target = list(targets)[0]
            generated = generate_answer(model, tokenizer, prompt)
            if generated.lower() == target.lower():
                local_correct += 1
            local_total += 1
            if sample_print_count < 5 and dist.get_rank() == 0:
                with open("output.txt", "a") as f:  # "a" mode appends to the file
                    print(f"[Target: {target.lower()} | Prediction: {generated.lower()}]", file=f)
                sample_print_count += 1

    # Synchronize ranks before collective operation.
    dist.barrier()

    # Gather results from all GPUs.
    local_correct_tensor = torch.tensor(local_correct, device=model.device)
    local_total_tensor = torch.tensor(local_total, device=model.device)
    dist.all_reduce(local_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_total_tensor, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        accuracy = 100.0 * local_correct_tensor.item() / local_total_tensor.item() if local_total_tensor.item() > 0 else 0
        print(f"Accuracy: {accuracy:.2f}%")

def evaluate_on_all_tasks(
    model_checkpoint_dir,  # folder where your DS checkpoint is stored
    dataset_infos
):
    """
    Loads the final LLaMA SVD model from `model_checkpoint_dir` (DeepSpeed checkpoint)
    and evaluates it on each task defined in `dataset_infos`.
    Prints the accuracy for each task and the overall average accuracy.
    """
    import torch.distributed as dist

    # Load tokenizer & config
    model_name = "baffo32/decapoda-research-llama-7B-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    config = LlamaConfig.from_pretrained(model_name)
    config.use_cache = False

    # Initialize our model
    model = LlamaWithSVD(config, svd_config={}, initialize_svd=False)
    model.reinitialize_svd()  # If needed. Or do it after load below.

    # Wrap with DeepSpeed
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    # Now load the DS checkpoint from model_checkpoint_dir
    # This effectively loads the model's weights into model_engine.
    model_engine.load_checkpoint(model_checkpoint_dir)
    model_engine.eval()

    # We'll track results in a dictionary
    all_accuracies = {}
    for task_name, info in dataset_infos.items():
        if not os.path.exists(info["json_path"]):
            if dist.get_rank() == 0:
                print(f"Warning: Test file not found for {task_name} at {info['json_path']}")
            continue

        # Build the dataset (like GenericClassificationDataset):
        dataset = GenericClassificationDataset(
            info["json_path"],
            tokenizer,
            info["label_mapping"],
            task_name.lower()
        )

        # Evaluate using the same distributed approach as above
        test_sampler = DistributedSampler(dataset, shuffle=False)
        test_loader = DataLoader(
            dataset,
            sampler=test_sampler,
            batch_size=1,
            collate_fn=lambda b: collate_fn(b, tokenizer)
        )

        # We'll do a local counter again
        correct_local, total_local = 0, 0
        sample_count = 0

        # Helper for generation
        def generate_answer(prompt):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_engine.device)
            with torch.no_grad():
                outputs = model_engine.module.generate(
                    input_ids,
                    max_new_tokens=16,
                    return_dict_in_generate=True
                )
            new_token_ids = outputs.sequences[:, input_ids.shape[-1]:]
            return tokenizer.decode(new_token_ids[0], skip_special_tokens=True).strip()

        # Loop
        for batch in tqdm(test_loader, desc=f"Evaluating {task_name}"):
            # Adjust if your collate_fn has a different structure
            input_text = batch["prompt_text"][0]
            target_text = batch["target_text"][0]

            gen_ans = generate_answer(input_text)
            if gen_ans.lower() == target_text.lower():
                correct_local += 1

            if sample_count < 5 and dist.get_rank() == 0:
                print(f"[{task_name}] Target: {target_text} | Pred: {gen_ans}")
            sample_count += 1
            total_local += 1

        # Sum across ranks
        correct_tensor = torch.tensor(correct_local, device=model_engine.device, dtype=torch.long)
        total_tensor = torch.tensor(total_local, device=model_engine.device, dtype=torch.long)

        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor,  op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            accuracy = float(correct_tensor.item()) / float(total_tensor.item()) if total_tensor.item() > 0 else 0
            all_accuracies[task_name] = accuracy
            print(f"{task_name} accuracy: {accuracy*100:.2f}%")

    # If rank 0, compute overall average
    if dist.get_rank() == 0 and all_accuracies:
        avg_acc = np.mean(list(all_accuracies.values())) * 100
        print(f"\nAverage accuracy across all tasks: {avg_acc:.2f}%")
    return all_accuracies

###################################################
# 7. Main
###################################################
if __name__ == "__main__":

    deepspeed.init_distributed()  # Initialize distributed

    # 1) Train the model (DeepSpeed checkpoint saved in OUTPUT_MODEL_NAME folder)
    model_engine, tokenizer, train_dataset, test_dataset = train_svd_model(
        fine_tune_dataset=FINE_TUNE_DATASET,
        starting_checkpoint=STARTING_CHECKPOINT,
        output_model_name=OUTPUT_MODEL_NAME
    )
    # Note: If your train_svd_model(...) returns the raw model and not the model_engine,
    # that is also fine. Adjust accordingly.

    # 2) Reload for evaluation
    print("Evaluating on the train dataset:")
    evaluate(model_engine.module, tokenizer, train_dataset)
    print("Evaluating on the test dataset:")
    evaluate(model_engine.module, tokenizer, test_dataset)

    # 4) Optionally evaluate on all tasks
    # evaluate_on_all_tasks(OUTPUT_MODEL_NAME, DATASET_INFOS)

    dist.destroy_process_group()