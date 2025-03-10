import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_svd_dataset", type=str, required=True)
parser.add_argument("--fine_tune_dataset", type=str, required=True)
parser.add_argument("--starting_checkpoint", type=str, required=True)
parser.add_argument("--output_model_name", type=str, required=True)
parser.add_argument("--scale_factor", type=float, required=True)

args = parser.parse_args()

SOURCE_SVD_DATASET = args.source_svd_dataset
FINE_TUNE_DATASET = args.fine_tune_dataset
STARTING_CHECKPOINT = args.starting_checkpoint
OUTPUT_MODEL_NAME = args.output_model_name
SCALE_FACTOR = args.scale_factor  # This will be passed dynamically from the shell script

import os
import json
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_prompt(sample, dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "agnews":
        # return "classify ag_news dataset: " + sample["text"]
        return (
            "What is the topic of the following paragraph? "
            "Choose one from the following options [World, Sports, Business, Science or Technology]. "
            + sample["text"]
        )
    elif dataset_name in ["amazon", "yelp"]:
        # return "classify amazon dataset: " + sample["content"]
        return (
            "What is the sentiment of the following paragraph? "
            "Choose one from the following options [very negative, negative, neutral, positive, very positive]. "
            + sample["text"]
        )
    elif dataset_name == "dbpedia":
        # return "classify dbpedia dataset: " + sample["content"]
        return (
            "What is the topic of the following paragraph? "
            "Choose one from the following options [Company, Educational Institution, Artist, Athlete, Office Holder, "
            "Mean of Transportation, Building, Natural Place, Village, Animal, Plant, Album, "
            "Film, Written Work]. "
            + sample["text"]
        )
    elif dataset_name == "yahoo":
        # return "classify yahoo dataset: " + sample["question_title"] + " " + sample["question_content"]
        return (
            "What is the topic of the following paragraph? "
            "Choose one from the following options [Sports, Entertainment & Music, Health, Education & Reference, Family & Relationships, Politics & Government, Science & Mathematics, Business & Finance, Computers & Internet, Society & Culture]. "
            + sample["text"]
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
# 2. T5 Model Subclass with SVD (Only for Selected Parameters)
###################################################

class T5WithSVD(T5ForConditionalGeneration):
    """
    Subclass that, on initialization, decomposes selected weight matrices via SVD.
    Only parameters specified in the svd_config are decomposed.
    For each such 2D weight, we freeze the top singular vectors (50% by default)
    and register the lower half (trainable) as parameters.

    Additionally, we pre-compute the module mapping for faster weight injection.
    """
    def __init__(self, config: T5Config, svd_config=None, initialize_svd=True):
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
      - SelfAttention.q.weight
      - SelfAttention.k.weight
      - SelfAttention.v.weight
      - SelfAttention.o.weight
      - DenseReluDense.wi.weight
      - DenseReluDense.wo.weight
    For each such 2D parameter, set:
         top_k = floor(min(dim0, dim1) / 2)
    """
    target_patterns = [
        "SelfAttention.q.weight",
        "SelfAttention.k.weight",
        "SelfAttention.v.weight",
        "SelfAttention.o.weight",
        "DenseReluDense.wi.weight",
        "DenseReluDense.wo.weight"
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
            top_k = int(np.floor(max(param.shape)*SCALE_FACTOR))
            full_rank = min(param.shape)
            if top_k > full_rank:
                top_k = full_rank
            config[name] = top_k
    # save_svd_config(config)
    return config

# def auto_generate_target_svd_config(model, tokenizer, n_samples=128, batch_size=8, num_batches=5, source_dataset=SOURCE_SVD_DATASET):
#     """
#     For each target parameter (matching target_patterns), compute the adaptive retention ratio based on
#     the importance I(W) measured using actual inputs from the AGNews test set.

#     For each target parameter W (shape: (d, m), let d = min(W.shape)).
#     For each such parameter:
#        - Run num_batches of AGNews test data through the model with hooks to capture the input X for
#          the module corresponding to W.
#        - Concatenate the captured X from all batches to form a matrix X of shape (m, total_samples).
#        - Compute I(W) = average cosine similarity between columns of X and Y = W @ X.
#     Then normalize importance by the mean and set:
#        CR(W) = 1 + (I(W)/mean(I(W)))*((d/2) - 1)
#        k = round(CR(W) * d / 2)
#     Clamp k between 1 and d.
#     Return a dictionary mapping parameter names to top_k.
#     """
#     target_patterns = [
#         "SelfAttention.q.weight",
#         "SelfAttention.k.weight",
#         "SelfAttention.v.weight",
#         "SelfAttention.o.weight",
#         "DenseReluDense.wi.weight",
#         "DenseReluDense.wo.weight"
#     ]
#     # Dictionary to store importance for each target parameter.
#     importance_dict = {}
#     # Dictionary to store captured inputs for each target parameter.
#     captured_inputs = {name: [] for name, param in model.named_parameters()
#                          if any(pat in name for pat in target_patterns) and len(param.shape)==2}

#     # Create hooks to capture inputs for each target module.
#     hooks = {}
#     def get_hook(name):
#         def hook(module, input, output):
#             # input[0] might have shape (batch_size, seq_length, in_features)
#             X = input[0]
#             # Flatten the batch and sequence dimensions into one:
#             X = X.reshape(-1, X.shape[-1])  # shape: (batch_size * seq_length, in_features)
#             # Transpose so that columns represent individual samples:
#             captured_inputs[name].append(X.transpose(0, 1).detach())
#         return hook

#     # For each target parameter, register a hook on its parent module.
#     for name, param in model.named_parameters():
#         if any(pat in name for pat in target_patterns) and len(param.shape)==2:
#             mod, attr = model._get_module_by_name(name)
#             if mod is not None:
#                 hooks[name] = mod.register_forward_hook(get_hook(name))

#     # Now run a few batches of test data from the dataset.
#     from datasets import load_dataset

#     # Load the chosen source dataset and build inputs appropriately.
#     if source_dataset.lower() == "ag_news":
#         dataset = load_dataset("ag_news", split="test")
#         inputs = [f"classify ag_news dataset: " + sample["text"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "amazon":
#         dataset = load_dataset("amazon_polarity", split="test")
#         inputs = [f"classify amazon dataset: " + sample["content"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "yelp":
#         dataset = load_dataset("yelp_review_full", split="test")
#         inputs = [f"classify yelp dataset: " + sample["text"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "dbpedia":
#         dataset = load_dataset("dbpedia_14", split="test")
#         inputs = [f"classify dbpedia dataset: " + sample["content"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "yahoo":
#         dataset = load_dataset("yahoo_answers_topics", split="test")
#         inputs = [f"classify yahoo dataset: " + sample["question_title"] + " " + sample["question_content"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "mnli":
#         dataset = load_dataset("glue", "mnli", split="validation_matched")
#         inputs = [f"classify mnli dataset: premise: " + sample["premise"] + " hypothesis: " + sample["hypothesis"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "qqp":
#         dataset = load_dataset("glue", "qqp", split="validation")
#         inputs = [f"classify qqp dataset: question1: " + sample["question1"] + " question2: " + sample["question2"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "rte":
#         dataset = load_dataset("glue", "rte", split="test")
#         inputs = [f"classify rte dataset: sentence1: " + sample["sentence1"] + " sentence2: " + sample["sentence2"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "sst2":
#         dataset = load_dataset("glue", "sst2", split="test")
#         inputs = [f"classify sst2 dataset: " + sample["sentence"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "wic":
#         dataset = load_dataset("super_glue", "wic", split="test")
#         inputs = [f"classify wic dataset: word: " + sample["word"] + " sentence1: " + sample["sentence1"] + " sentence2: " + sample["sentence2"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "cb":
#         dataset = load_dataset("super_glue", "cb", split="test")
#         inputs = [f"classify cb dataset: premise: " + sample["premise"] + " hypothesis: " + sample["hypothesis"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "copa":
#         dataset = load_dataset("super_glue", "copa", split="test")
#         inputs = [f"classify copa dataset: premise: " + sample["premise"] + " choice1: " + sample["choice1"] + " choice2: " + sample["choice2"] for sample in dataset.select(range(n_samples))]
#     # elif source_dataset.lower() == "multirc":
#     #     dataset = load_dataset("super_glue", "multirc", split="test")
#     #     inputs = [f"classify multirc dataset: question: " + sample["question"] + " passage: " + sample["passage"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "boolq":
#         dataset = load_dataset("super_glue", "boolq", split="test")
#         inputs = [f"classify boolq dataset: question: " + sample["question"] + " passage: " + sample["passage"] for sample in dataset.select(range(n_samples))]
#     elif source_dataset.lower() == "imdb":
#         dataset = load_dataset("imdb", split="test")
#         inputs = [f"classify imdb dataset: " + sample["text"] for sample in dataset.select(range(n_samples))]
#     else:
#         raise ValueError(f"Unknown source dataset: {source_dataset}")

#     encodings = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
#     # Wrap the BatchEncoding in a custom Dataset
#     class BatchEncodingDataset(Dataset):
#         def __init__(self, encodings):
#             self.encodings = encodings
#         def __len__(self):
#             return self.encodings["input_ids"].shape[0]
#         def __getitem__(self, idx):
#             return {key: val[idx] for key, val in self.encodings.items()}

#     dataset = BatchEncodingDataset(encodings)
#     loader = DataLoader(dataset, batch_size=batch_size)
#     # agnews_loader = DataLoader(encodings, batch_size=batch_size)

#     model = model.to(device)
#     model.eval()
#     batches = 0
#     with torch.no_grad():
#         for batch in loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             # _ = model(**batch)  # forward pass to trigger hooks
#             # batches += 1
#             # if batches >= num_batches:
#             #     break

#             batch_size = batch["input_ids"].shape[0]
#             # Create a dummy decoder input using the model's decoder_start_token_id.
#             # T5 usually uses 0 or the value from config.decoder_start_token_id.
#             dummy_decoder_input_ids = torch.full(
#                 (batch_size, 1),
#                 model.config.decoder_start_token_id,
#                 device=device,
#                 dtype=batch["input_ids"].dtype
#             )
#             # Forward pass with both encoder and decoder inputs.
#             _ = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 decoder_input_ids=dummy_decoder_input_ids
#             )

#     # Remove hooks.
#     for h in hooks.values():
#         h.remove()

#     # Now compute importance for each target parameter.
#     for name in captured_inputs.keys():
#         # Concatenate captured inputs along last dimension.
#         X = torch.cat(captured_inputs[name], dim=1).to(device)  # shape: (in_features, total_samples)
#         W = model.state_dict()[name].to(device)
#         Y = torch.mm(W, X)

#         # Determine m = min(W.shape) and slice both X and Y to the first m rows.
#         m = min(W.shape)
#         X_mod = X[:m, :]
#         Y_mod = Y[:m, :]

#         X_norm = X_mod / (torch.norm(X_mod, dim=0, keepdim=True) + 1e-8)
#         Y_norm = Y_mod / (torch.norm(Y_mod, dim=0, keepdim=True) + 1e-8)
#         cosine_sim = torch.sum(X_norm * Y_norm, dim=0)
#         I_W = torch.mean(cosine_sim).item()
#         importance_dict[name] = I_W

#     mean_importance = np.mean(list(importance_dict.values()))
#     config = {}
#     for name, param in model.named_parameters():
#         if name in importance_dict:
#             d = min(param.shape)
#             I_W = importance_dict[name]
#             I_n = I_W / (mean_importance + 1e-8)
#             mrr = d / 2.0 # 1.0
#             trr = d # d / 2.0
#             CR = mrr + I_n * (trr - mrr)
#             # As explained: full params of W is 2*d^2 (for square W) and retained params is 2*d*k,
#             # so we set k/d = CR  => k = CR * d.
#             # k = int(round(CR * d))
#             k = int(round(CR))
#             k = max(1, min(k, d))
#             config[name] = k
#     save_svd_config(config)
#     return config

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

def collate_fn_fn(batch, tokenizer, max_source_length=512, max_target_length=16):
    inputs, targets = zip(*batch)
    input_encodings = tokenizer(list(inputs), padding=True, truncation=True, max_length=max_source_length, return_tensors="pt")
    target_encodings = tokenizer([str(t) for t in targets], padding=True, truncation=True, max_length=max_target_length, return_tensors="pt")
    input_encodings["labels"] = target_encodings["input_ids"]
    return input_encodings

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
    elif fine_tune_dataset.lower() == "sst-2":
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

    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.use_cache = False  # disable cache for training

    # Load a base T5 model to auto-generate the target SVD config.
    # base_model = T5ForConditionalGeneration.from_pretrained(model_name)
    base_model = T5WithSVD(config, svd_config={}, initialize_svd=False)
    base_model.load_state_dict(torch.load(starting_checkpoint, map_location=device), strict=False)
    base_model = base_model.to(device)
    target_svd_config = auto_generate_target_svd_config(base_model)
    # target_svd_config = auto_generate_target_svd_config(base_model, tokenizer)
    print("Auto-generated target SVD config:")
    for k, v in target_svd_config.items():
        print(f"  {k}: freeze top {v} singular vectors")

    # Initialize our custom SVD model with target_svd_config.
    model = T5WithSVD(config, svd_config=target_svd_config, initialize_svd=False)
    # Load pretrained weights into our SVD model.
    model.load_state_dict(torch.load(starting_checkpoint, map_location=device), strict=False)
    model.reinitialize_svd()
    model = model.to(device)

    # # Load the original AGNews state dictionary
    # orig_state = torch.load('t5_finetuned_agnews.pt', map_location=device)

    # # For each parameter in the original state, compare with the corresponding effective parameter in model.
    # for name, orig_param in orig_state.items():
    #     # If this parameter was decomposed (present in our svd_config), then use our reconstruction function.
    #     if name in model.svd_config:
    #         # Compute effective weight from SVD (including residual)
    #         effective_weight = model._reconstruct_weight(name)
    #         # Compute relative error
    #         error = torch.norm(orig_param.to(device) - effective_weight) / torch.norm(orig_param.to(device))
    #         print(f"{name} (decomposed): relative error = {error.item():.2e}")
    #     else:
    #         # Otherwise, compare directly.
    #         try:
    #             model_param = model.state_dict()[name]
    #             error = torch.norm(orig_param.to(device) - model_param) / torch.norm(orig_param.to(device))
    #             print(f"{name} (not decomposed): relative error = {error.item():.2e}")
    #         except KeyError:
    #             print(f"{name} is not present in the current model state_dict.")

    # torch.save(model.state_dict(), "t5_svd_amazon.pt")

    # Create datasets and dataloaders
    train_dataset = GenericClassificationDataset(train_json_path, tokenizer, label_mapping, dataset_prompt)
    test_dataset = GenericClassificationDataset(test_json_path, tokenizer, label_mapping, dataset_prompt)


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              collate_fn=lambda batch: collate_fn_fn(batch, tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             collate_fn=lambda batch: collate_fn_fn(batch, tokenizer))

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 1  # adjust as needed

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)
        start_time = time.time()

        for batch in progress_bar:
            for key, val in batch.items():
                batch[key] = val.to(device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            model.project_gradients()  # ensure gradients remain in correct subspace
            optimizer.step()

            total_loss += loss.item()
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (progress_bar.n + 1) * (len(train_loader) - progress_bar.n)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", eta=f"{remaining_time:.2f}s")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model (with SVD modifications)
    torch.save(model.state_dict(), output_model_name)
    print(f"Model saved as '{output_model_name}'")
    return model, tokenizer, train_loader, test_loader

###################################################
# 6. Inference
###################################################
def inference_svd_model(output_model_name=OUTPUT_MODEL_NAME):
    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.use_cache = False
    # Re-generate the same target SVD configuration
    base_model = T5ForConditionalGeneration.from_pretrained(model_name)
    target_svd_config = auto_generate_target_svd_config(base_model)
    model = T5WithSVD(config, svd_config=target_svd_config)
    model.load_state_dict(torch.load(output_model_name), strict=False)
    model = model.to(device)
    model.eval()

    # Try a generation example – here we provide a sample review.
    input_text = "classify: This product exceeded my expectations and works perfectly!"
    input_enc = tokenizer([input_text], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**input_enc, max_length=16)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

###################################################
# 6. Evaluation on Test Set
###################################################
def evaluate_model(model, tokenizer, data_loader, dataset_name="Test"):
    model.eval()
    total, correct = 0, 0
    sample_count = 0
    print(f"Evaluating on {dataset_name} set...")

    for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name} Set", unit="batch"):
        # Move batch tensors to device
        for key, val in batch.items():
            batch[key] = val.to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=16
            )
        # Decode predictions and targets
        predictions = [tokenizer.decode(g, skip_special_tokens=True).strip().lower()
                       for g in generated_ids]
        targets = [tokenizer.decode(label, skip_special_tokens=True).strip().lower()
                   for label in batch["labels"]]
        for pred, target in zip(predictions, targets):
            total += 1
            if pred.lower() == target.lower():
                correct += 1
            if sample_count < 5:
                print(f"[{dataset_name} Set] Target: {target} | Prediction: {pred}")
                sample_count += 1
    accuracy = correct / total if total > 0 else 0
    print(f"{dataset_name} Accuracy: {accuracy * 100:.2f}%")

###################################################
# 7. Main
###################################################
if __name__ == "__main__":

    # Train the model and save it
    model, tokenizer, train_loader, test_loader = train_svd_model(
        fine_tune_dataset=FINE_TUNE_DATASET,
        starting_checkpoint=STARTING_CHECKPOINT,
        output_model_name=OUTPUT_MODEL_NAME
    )

    # Reload the saved model for evaluation (exactly the same as training)
    config = T5Config.from_pretrained("t5-large")
    tokenizer = T5Tokenizer.from_pretrained("t5-large")

    # Initialize the model with the same SVD config used in training
    base_model = T5WithSVD(config, svd_config={}, initialize_svd=False)
    base_model.load_state_dict(torch.load(OUTPUT_MODEL_NAME, map_location=device), strict=False)
    base_model.reinitialize_svd()
    base_model = base_model.to(device)
    base_model.eval()
    
    print(f"Loaded saved model from '{OUTPUT_MODEL_NAME}' for evaluation.")

    # Evaluate on both Train and Test sets
    evaluate_model(base_model, tokenizer, train_loader, dataset_name="Train")
    evaluate_model(base_model, tokenizer, test_loader, dataset_name="Test")