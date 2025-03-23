# Configurable parameters:
SOURCE_SVD_DATASET = "dbpedia"       # Dataset to use when computing the adaptive SVD config.
FINE_TUNE_DATASET = "agnews"       # Fine-tuning dataset; options: "agnews", "amazon", "yelp", "dbpedia", "yahoo"
STARTING_CHECKPOINT = "llama_finetuned_dbpedia.pt"  # Path to the checkpoint you want to start from.
OUTPUT_MODEL_NAME = "llama_svd_agnews.pt"         # Name for the saved model after fine-tuning.

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
from accelerate import load_checkpoint_and_dispatch

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.autograd.set_detect_anomaly(True)

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
            "Task: TC\n"
            "Dataset: ag news\n"
            "What is the topic of the following paragraph? Choose one from the option.\n"
            "Option: World, Sports, Business, Science or Technology\n" + sample['text'] + "Answer: "
        )
    elif dataset_name in ["amazon", "yelp"]:
        return (
            "Classify the sentiment of the following text into one of these categories: "
            "[very negative, negative, neutral, positive, very positive].\n\n"
            "Text: " + sample["text"] + "\nAnswer:"
        )
    elif dataset_name == "dbpedia":
        return (
            "Task: TC\n"
            "Dataset: dbpedia\n"
            "What is the topic of the following paragraph? Choose one from the option.\n"
            "Option: Company, Educational Institution, Artist, Athlete, Office Holder, "
            "Mean of Transportation, Building, Natural Place, Village, Animal, Plant, "
            "Album, Film, Written Work\n" + sample['text'] + "Answer: "
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
                svd_dict = decompose_weight_matrix(param.data, top_k=top_k)
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
                module_svd.U_low = nn.Parameter(svd_dict["U_low"])
                module_svd.S_low = nn.Parameter(svd_dict["S_low"])
                module_svd.V_low = nn.Parameter(svd_dict["V_low"])
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
            if top_k > full_rank:
                top_k = full_rank
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
    full_texts = [inp + tgt + tokenizer.eos_token for inp, tgt in zip(inputs, targets)]
    encodings = tokenizer(
        full_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    
    # # Now create labels, but mask out the prompt tokens.
    # # First, get the tokenized version of just the prompt.
    # prompt_texts = [inp for inp in inputs]
    # prompt_encodings = tokenizer(
    #     prompt_texts, padding=True, truncation=True,
    #     max_length=max_length, return_tensors="pt"
    # )
    labels = encodings["input_ids"].clone()
    
    # # For each sample, set label tokens corresponding to the prompt to -100.
    # for i in range(len(full_texts)):
    #     prompt_length = prompt_encodings["input_ids"][i].ne(tokenizer.pad_token_id).sum()
    #     # Set all tokens up to prompt_length to -100 so loss isn't computed on them
    #     labels[i, :prompt_length] = -100
    labels[encodings["attention_mask"] == 0] = -100
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

    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.vocab_size = 32001
    config.pad_token_id = 32000
    config.use_cache = False  # if applicable for LLaMA; otherwise remove or adjust

    # Create datasets and dataloaders
    train_dataset = GenericClassificationDataset(train_json_path, tokenizer, label_mapping, dataset_prompt)
    test_dataset = GenericClassificationDataset(test_json_path, tokenizer, label_mapping, dataset_prompt)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, tokenizer))
    # Load a base LLaMA model to auto-generate the target SVD config.
    base_model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    target_svd_config = auto_generate_target_svd_config(base_model)
    # target_svd_config = auto_generate_target_svd_config(base_model, tokenizer)
    print("Auto-generated target SVD config:")
    for k, v in target_svd_config.items():
        print(f"  {k}: freeze top {v} singular vectors")
    
    del base_model
    torch.cuda.empty_cache()

    # Initialize our custom SVD model with target_svd_config.
    model = LlamaWithSVD(config, svd_config=target_svd_config, initialize_svd=False)
    # Load pretrained weights into our SVD model.
    model = load_checkpoint_and_dispatch(model, checkpoint=starting_checkpoint, device_map="auto", strict=False)
    model.resize_token_embeddings(len(tokenizer))
    model.reinitialize_svd()
    model.gradient_checkpointing_enable()

    optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    num_epochs = 1  # adjust as needed

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)
        start_time = time.time()

        for batch in progress_bar:
            first_param_device = next(model.parameters()).device
            for key, val in batch.items():
                batch[key] = val.to(first_param_device)
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model (with SVD modifications)
    torch.save(model.state_dict(), output_model_name)
    print(f"Model saved as '{output_model_name}'")
    return model, tokenizer, train_dataset, test_dataset

###################################################
# 6. Evaluation on Test Set
###################################################
def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    # Tokenize only the prompt (without the target)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
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

def evaluate(model, tokenizer, dataset, dataset_name="Test"):
    correct, total = 0, 0
    sample_count = 0
    print(f"Evaluating on {dataset_name} set...")
    for inp, tgt in tqdm(dataset, desc="Evaluating"):
        generated_answer = generate_answer(model, tokenizer, inp)
        if generated_answer.lower() == tgt.lower():
            correct += 1
        if sample_count < 5:
            with open("output.txt", "a") as f:  # "a" mode appends to the file
                print(f"[Target: {tgt.lower()} | Prediction: {generated_answer.lower()}]", file=f)
            sample_count += 1
        total += 1
    print(f"{dataset_name} Accuracy: {100.0 * correct / total:.2f}%")

def evaluate_on_all_tasks(model_checkpoint, dataset_infos):
    """
    Loads the final LLaMA SVD model from `model_checkpoint` and evaluates it on each
    task defined in `dataset_infos` using generation-based evaluation.
    Prints the accuracy for each task and the overall average accuracy.
    """
    model_name = "baffo32/decapoda-research-llama-7B-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    config = LlamaConfig.from_pretrained(model_name)
    config.use_cache = False  # disable cache for training/generation

    # Load the trained model in bfloat16 on GPU 0
    model = LlamaWithSVD(config, svd_config={}, initialize_svd=False)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device), strict=False)
    model.reinitialize_svd()
    model = model.to(device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.eval()

    task_accuracies = {}

    # Loop over each task defined in dataset_infos.
    for task_name, info in dataset_infos.items():
        print(f"\nEvaluating on {task_name} set:")
        json_path = info["json_path"]
        label_mapping = info["label_mapping"]

        if not os.path.exists(json_path):
            print(f"Warning: Test file not found for {task_name} at {json_path}")
            continue

        # Load the evaluation dataset (using your GenericClassificationDataset)
        eval_dataset = GenericClassificationDataset(json_path, tokenizer, label_mapping, task_name.lower())

        correct, total = 0, 0
        sample_count = 0
        for prompt, tgt in tqdm(eval_dataset, desc=f"Evaluating {task_name}"):
            # Generate answer using only the prompt
            generated_answer = generate_answer(model, tokenizer, prompt, max_new_tokens=16)
            if generated_answer.strip().lower() == tgt.strip().lower():
                correct += 1
            if sample_count < 5:
                print(f"[Target: {tgt.strip()} | Prediction: {generated_answer.strip()}]")
                sample_count += 1
            total += 1

        acc = correct / total if total > 0 else 0
        task_accuracies[task_name] = acc
        print(f"{task_name} accuracy: {acc * 100:.2f}%")

    avg_acc = np.mean(list(task_accuracies.values())) if task_accuracies else 0
    print("\nAverage accuracy across all tasks: {:.2f}%".format(avg_acc * 100))
    return task_accuracies

###################################################
# 7. Main
###################################################
if __name__ == "__main__":

    # Train the model and save it
    model, tokenizer, train_dataset, test_dataset = train_svd_model(
        fine_tune_dataset=FINE_TUNE_DATASET,
        starting_checkpoint=STARTING_CHECKPOINT,
        output_model_name=OUTPUT_MODEL_NAME
    )

    # Reload the saved model for evaluation (exactly the same as training)
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.vocab_size = 32001
    config.pad_token_id = 32000

    # Initialize the model with the same SVD config used in training
    model = LlamaWithSVD(config, svd_config={}, initialize_svd=False)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(OUTPUT_MODEL_NAME), strict=False)
    model.reinitialize_svd()
    model.gradient_checkpointing_enable()
    model.eval()
    
    print(f"Loaded saved model from '{OUTPUT_MODEL_NAME}' for evaluation.")

    # Evaluate on both Train and Test sets
    evaluate(model, tokenizer, train_dataset, dataset_name="Train")
    evaluate(model, tokenizer, test_dataset, dataset_name="Test")

    # evaluate_on_all_tasks(OUTPUT_MODEL_NAME, DATASET_INFOS)