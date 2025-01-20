import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

###################################################
# 1. Define a dataset (dummy example)
###################################################
class DummySeq2SeqDataset(Dataset):
    """
    A trivial dataset that returns (input_text, target_text).
    Replace with your real dataset.
    """
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]["input"], self.data[idx]["target"]

def collate_fn(batch, tokenizer, max_length=128):
    """
    Tokenize and prepare the batch for T5.
    """
    inputs, targets = zip(*batch)
    input_enc = tokenizer(list(inputs), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    target_enc = tokenizer(list(targets), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # T5 uses 'labels' for the decoder
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": target_enc["input_ids"],
    }

###################################################
# 2. Helper function for SVD and param management
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
    # SVD
    # shape of W = (out_features, in_features)
    device = weight.device
    W = weight.to(torch.float32)  # ensure float32 for SVD
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)  # Vt has shape (in_features, in_features)

    # If top_k >= min(U.shape[1], Vt.shape[0]), clamp it
    k = min(top_k, S.shape[0])
    
    # High subspace (frozen)
    U_high = U[:, :k].detach()
    S_high = S[:k].detach()
    V_high = Vt[:k, :].detach()
    
    # Low subspace (trainable)
    U_low = U[:, k:].detach()
    S_low = S[k:].detach()
    V_low = Vt[k:, :].detach()
    
    # Move them to correct device
    U_high = U_high.to(device)
    S_high = S_high.to(device)
    V_high = V_high.to(device)
    U_low = U_low.to(device)
    S_low = S_low.to(device)
    V_low = V_low.to(device)

    # Wrap the "low" parts as parameters; "high" parts as buffers
    return {
        "U_high": U_high,  # no gradient
        "S_high": S_high,  # no gradient
        "V_high": V_high,  # no gradient
        "U_low": nn.Parameter(U_low),  # trainable
        "S_low": nn.Parameter(S_low),  # trainable
        "V_low": nn.Parameter(V_low),  # trainable
        "rank_high": k
    }


def reconstruct_weight_matrix(svd_dict):
    """
    Reconstruct the weight matrix from both high and low subspaces:
        W = U_high * diag(S_high) * V_high^T + U_low * diag(S_low) * V_low^T
    """
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    # Reconstruct high part
    if U_high.shape[1] > 0 and S_high.shape[0] > 0:
        high_part = (U_high * S_high) @ V_high
    else:
        high_part = torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)

    # Reconstruct low part
    if U_low.shape[1] > 0 and S_low.shape[0] > 0:
        low_part = (U_low * S_low) @ V_low
    else:
        low_part = torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)

    return high_part + low_part


def project_gradient_to_orthogonal_space(svd_dict):
    """
    Zero out gradients that lie in the direction of the high subspace for each param in the low subspace.
    In other words, ensure that d(U_low), d(S_low), d(V_low) are orthogonal to the subspace spanned by U_high, V_high.
    
    For example, we can do something like:
       dU_low = dU_low - (U_high * (U_high^T @ dU_low))
    to remove any components in the column space of U_high. Similarly for V_low.
    This is a simplistic approach.
    """
    # If there's no gradient, return
    if svd_dict["U_low"].grad is None and svd_dict["S_low"].grad is None and svd_dict["V_low"].grad is None:
        return
    
    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]

    # Project out from U_low.grad
    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        # Remove the component that lies in col-space of U_high
        # col-space of U_high is spanned by columns of U_high
        # We'll do: dU <- dU - U_high (U_high^T dU)
        proj = U_high @ (U_high.transpose(0,1) @ dU)
        dU.sub_(proj)  # in-place

    # Project out from V_low.grad
    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        # V_high has shape (k, in_features). The row-space is spanned by rows of V_high
        # We want to remove any component of dV that is in row-space of V_high.
        # row-space of V_high is spanned by each row vector -> equivalently col-space of V_high^T
        # So: dV <- dV - ( (dV V_high^T) V_high )
        # But we have to do that carefully with shapes.
        # One simpler approach is: for each row i of dV, project out from row i of V_high.
        # We'll do a matrix approach with unsqueeze expansions. This can get tricky.
        
        # Let's do it in a more direct manner:
        # We can think of dV as (r_low, c) shaped. row-space is dimension r_low, col c
        # row-space of V_high is dimension k. We want to remove the projection onto each row of V_high.
        # A quick hack: project columns in col-space of V_high^T. We'll treat each row as a vector,
        # so we do: dV <- dV - (dV (V_high^T V_high^\top)) ? Let's keep it simpler:

        # We'll treat each row of dV, call it dV_i. Each row of V_high, call it V_high_j
        # We remove for each j: (dV_i dot V_high_j) * V_high_j / (V_high_j dot V_high_j) if needed
        # But if V_high is orthonormal (which it should be from SVD) then we can do:
        #    dV_i <- dV_i - sum_j( dV_i dot V_high_j ) * V_high_j
        # This is effectively: dV <- dV - (dV V_high^T) V_high  (since V_high is orthonormal, V_high V_high^T = I_k)
        
        # We'll assume V_high is orthonormal from SVD (it should be).
        # Then the projection of dV onto row-space of V_high is (dV * V_high^T) * V_high
        # but we must be careful with matrix dims. Let's do it directly:

        proj = (dV @ V_high.transpose(0,1)) @ V_high
        dV.sub_(proj)
    
    # S_low is just diagonal elements (vector). The "direction" for S_high is also a vector, but we typically freeze S_high.
    # If you want to project dS_low to be orthogonal to S_high, that might or might not make sense. Usually you freeze S_high entirely (no param).
    # For simplicity, do nothing to dS_low here (or if you want to zero it if you consider them in same dimension).
    # We'll do nothing as they do not share "direction space" the same way U/V do.


###################################################
# 3. T5 Model subclass with SVD
###################################################

class T5WithSVD(T5ForConditionalGeneration):
    """
    Subclass of T5ForConditionalGeneration that:
      - On init, decomposes each (or selected) weight matrix via SVD.
      - Freezes the top subspace.
      - Registers the bottom subspace as trainable parameters.
      - On forward, reconstructs the full weight.
      - Optionally, does gradient projection to keep updates orthogonal to the frozen subspace.
    """
    def __init__(self, config: T5Config, svd_config=None):
        """
        svd_config: dict specifying how many top singular vectors to freeze
                    for each layer or each matrix name, e.g.:
                    {
                       "encoder.block.0.layer.0.DenseReluDense.wi.weight": 16,
                       "shared.embedding": 0, # skip or no decomposition
                       ...
                    }
        You might parse a CSV to build this dictionary.
        """
        super().__init__(config)
        self.svd_config = svd_config if svd_config is not None else {}

        # A dictionary to store the SVD decomposition for each param we want to handle:
        #   self.svd_params[name] = {
        #       "U_high": buffer,
        #       "S_high": buffer,
        #       "V_high": buffer,
        #       "U_low": Parameter,
        #       "S_low": Parameter,
        #       "V_low": Parameter,
        #       "rank_high": k
        #   }
        self.svd_params = nn.ModuleDict()

        # We run through named_parameters, pick which ones to decompose
        self._initialize_svd_parameters()

    def _initialize_svd_parameters(self):
        for name, param in list(self.named_parameters()):
            # Decide if we want to decompose this param
            # We only do SVD on 2D weight matrices, skip biases or embeddings with dimension > 2
            if len(param.shape) == 2 and name in self.svd_config and self.svd_config[name] > 0:
                top_k = self.svd_config[name]
                print(f"[SVD Init] Decomposing {name} with top_k={top_k}")

                # Decompose
                svd_dict = decompose_weight_matrix(param.data, top_k=top_k)

                # Register buffers + parameters
                self.register_buffer(f"{name}_U_high", svd_dict["U_high"])
                self.register_buffer(f"{name}_S_high", svd_dict["S_high"])
                self.register_buffer(f"{name}_V_high", svd_dict["V_high"])

                # Low rank subspace as parameters
                U_low = svd_dict["U_low"]
                S_low = svd_dict["S_low"]
                V_low = svd_dict["V_low"]
                # We'll store them in a sub-module so that they appear in .parameters()
                module_svd = nn.Module()
                module_svd.register_parameter("U_low", U_low)
                module_svd.register_parameter("S_low", S_low)
                module_svd.register_parameter("V_low", V_low)
                module_svd.rank_high = svd_dict["rank_high"]
                
                self.svd_params[name] = module_svd

                # Remove the original param from the model's param list 
                # (we do that by setting 'requires_grad=False' or something similar).
                param.requires_grad = False
            else:
                # Not decomposing this param
                pass

    def _reconstruct_weight(self, name):
        """
        Reconstruct the full weight matrix from the stored SVD decomposition (if it exists)
        or return the original param if not decomposed.
        """
        if name in self.svd_params:
            # Reconstruct from high + low
            U_high = getattr(self, f"{name}_U_high")
            S_high = getattr(self, f"{name}_S_high")
            V_high = getattr(self, f"{name}_V_high")

            U_low = self.svd_params[name].U_low
            S_low = self.svd_params[name].S_low
            V_low = self.svd_params[name].V_low

            # Build dict for reconstruct
            svd_dict = {
                "U_high": U_high,
                "S_high": S_high,
                "V_high": V_high,
                "U_low": U_low,
                "S_low": S_low,
                "V_low": V_low
            }
            W = reconstruct_weight_matrix(svd_dict)
            return W
        else:
            # Not decomposed, just return the original param
            return dict(self.named_parameters())[name]

    def forward(self, *args, **kwargs):
        """
        Override forward to:
          1) Reconstruct the decomposed weights on-the-fly.
             - We have to inject them into the right modules.
          2) Then call the standard T5 forward.
        """
        # Step 1: inject reconstructed weights
        # We'll do a simple approach: for each decomposed param, find the actual module
        # that uses that param, and set `module.weight.data = reconstructed`.
        # This is somewhat hacky because T5 can rename parameters. We'll do a naive approach
        # that works if the param name has a direct path.

        with torch.no_grad():
            for name in self.svd_params:
                # name might look like: "encoder.block.0.layer.0.DenseReluDense.wi.weight"
                # We need to locate this module. We can do so with a utility function:
                module, param_name = self._get_module_by_name(name)
                if module is not None and hasattr(module, param_name):
                    W = self._reconstruct_weight(name)
                    getattr(module, param_name).data.copy_(W)

        # Step 2: call the original forward
        return super().forward(*args, **kwargs)

    def _get_module_by_name(self, name):
        """
        Utility to retrieve the module object and the final parameter name
        from the "dot" path. For example:
            name="encoder.block.0.layer.0.DenseReluDense.wi.weight"
        We find `self.encoder.block[0].layer[0].DenseReluDense.wi` as the module,
        and return (module, "weight").
        """
        parts = name.split(".")
        # The last part is typically "weight" or "bias"
        param_name = parts[-1]
        module_parts = parts[:-1]

        # Start from 'self'
        mod = self
        for p in module_parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            elif p.isdigit():
                # if it's an index for a list or nn.ModuleList
                mod = mod[int(p)]
            else:
                # can't find the path
                return None, None
        return mod, param_name

    def project_gradients(self):
        """
        After loss.backward(), call this to project gradients in the "low" subspace
        so that no component in the high subspace is updated.
        """
        for name, module_svd in self.svd_params.items():
            # Build an svd_dict for projection
            svd_dict = {
                "U_high": getattr(self, f"{name}_U_high"),
                "S_high": getattr(self, f"{name}_S_high"),
                "V_high": getattr(self, f"{name}_V_high"),
                "U_low": module_svd.U_low,
                "S_low": module_svd.S_low,
                "V_low": module_svd.V_low,
            }
            project_gradient_to_orthogonal_space(svd_dict)


###################################################
# 4. Example usage: training & inference
###################################################

def load_svd_config_from_csv(csv_path):
    """
    Suppose your CSV has columns: [param_name, top_k].
    We'll parse that into a dict: { param_name: top_k, ... }
    """
    svd_config = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            param_name = row["param_name"]
            top_k = int(row["top_k"])
            svd_config[param_name] = top_k
    return svd_config

def train_svd_model():
    ############################################################################
    # 1. Load or define your T5 config, tokenizer, and possibly a pretrained model
    ############################################################################
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    
    # Example: let’s pretend we have a CSV that says which param_name -> top_k
    # Here we skip the real CSV reading for a quick example:
    # For demonstration, we'll pick one linear layer to freeze top 16 singular values
    # Adjust to your real CSV approach:
    svd_config = {
       "encoder.block.0.layer.0.DenseReluDense.wi.weight": 16,  # freeze top-16
       "encoder.block.0.layer.0.DenseReluDense.wo.weight": 8,   # freeze top-8, etc...
       # ...
    }
    # Or do: svd_config = load_svd_config_from_csv("thresholds.csv")

    # Initialize our custom model
    model = T5WithSVD(config, svd_config=svd_config)

    # Optionally load pretrained T5 weights
    pretrained_model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    model.to("cuda")

    ############################################################################
    # 2. Build a dataset & dataloader
    ############################################################################
    # Dummy data
    dummy_data = [
        {"input": "translate English to German: Hello world", "target": "Hallo Welt"},
        {"input": "translate English to German: I love cats", "target": "Ich liebe Katzen"},
        # ...
    ]
    dataset = DummySeq2SeqDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                            collate_fn=lambda x: collate_fn(x, tokenizer, max_length=32))

    ############################################################################
    # 3. Prepare optimizer
    ############################################################################
    # Notice that the high subspace is not in model.parameters(), only low is.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    ############################################################################
    # 4. Training loop
    ############################################################################
    model.train()
    for epoch in range(2):  # small epoch count for demonstration
        for batch in dataloader:
            for k, v in batch.items():
                batch[k] = v.cuda()
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # Project out the gradients w.r.t. the high subspace
            model.project_gradients()

            optimizer.step()

            print(f"Epoch {epoch} - Loss: {loss.item()}")

    # Save model
    torch.save(model.state_dict(), "t5_svd_finetuned.pt")

def inference_svd_model():
    """
    Illustrates how to do inference. 
    """
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    
    # Suppose we have the same svd_config used in training
    svd_config = {
       "encoder.block.0.layer.0.DenseReluDense.wi.weight": 16,
       "encoder.block.0.layer.0.DenseReluDense.wo.weight": 8,
    }
    model = T5WithSVD(config, svd_config=svd_config)
    # Load your fine-tuned weights
    model.load_state_dict(torch.load("t5_svd_finetuned.pt"))
    model.to("cuda")
    model.eval()

    # Let's do a generation example
    input_text = "translate English to German: I really like pizza"
    input_enc = tokenizer([input_text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**input_enc, max_length=40)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    # Example: train then inference
    train_svd_model()
    inference_svd_model()