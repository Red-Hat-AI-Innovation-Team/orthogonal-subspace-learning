import os, json
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, GraniteForCausalLM

class GraniteWithSVD(GraniteForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def save_pretrained(self, save_directory, **kwargs):
        # Save the model weights and SVD components
        super().save_pretrained(save_directory, **kwargs)

        # Save the SVD components manually
        svd_data = {}
        for layer_index, layer in enumerate(self.model.layers):
            for matrix_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                svd_attr = f"{matrix_name}_svd"
                if hasattr(layer, svd_attr):
                    svd_components = getattr(layer, svd_attr)
                    svd_data[f"layer_{layer_index}_{matrix_name}"] = {
                        "U": svd_components["U"].detach().cpu(),
                        "S": svd_components["S"].detach().cpu(),
                        "Vh": svd_components["Vh"].detach().cpu(),
                    }

        # torch.save(svd_data, os.path.join(save_directory, "svd_components.pt"))

        torch.save(svd_data, "test_svd.pt")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the base model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Load the SVD components if available
        svd_path = f"{pretrained_model_name_or_path}/svd_components.pt"
        if os.path.exists(svd_path):
            svd_data = torch.load(svd_path)
            for key, components in svd_data.items():
                layer_index, matrix_name = key.split("_")[1:3]
                layer = model.model.layers[int(layer_index)]
                setattr(layer, f"{matrix_name}_svd", {
                    "U": components["U"].to(model.device),
                    "S": components["S"].to(model.device),
                    "Vh": components["Vh"].to(model.device),
                })
        return model
    
# Load data
with open("/dev/shm/orthogonal-subspace/training/data.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained("/new_data/experiments/ap-8b-p10-rhel13-data-id-2/hf_format/samples_10597250")

# Create DataLoader
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
    labels = [item["labels"] for item in batch]
    padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels]
    return {
        "input_ids": torch.tensor(padded_input_ids),
        "labels": torch.tensor(padded_labels),
    }

data_loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# Function to decompose and store SVD components dynamically
def decompose_and_store_svd(model, layer_number, matrix_name):
    """
    Decomposes the specified matrix into its SVD components and stores them in the model.
    
    Args:
        model: The GraniteForCausalLM model.
        layer_number: The target layer number (0-based index).
        matrix_name: The matrix to decompose ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj').
    """
    # Extract the target layer and matrix
    target_layer = model.model.layers[layer_number]
    if matrix_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        matrix = getattr(target_layer.self_attn, matrix_name).weight
    elif matrix_name in ['gate_proj', 'up_proj', 'down_proj']:
        matrix = getattr(target_layer.mlp, matrix_name).weight
    else:
        raise ValueError(f"Matrix name {matrix_name} is not valid.")

    print(f"Original Matrix Shape ({matrix_name}): {matrix.shape}")

    # Perform SVD
    matrix = matrix.to(dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(matrix)
    
    # Store SVD components in the model
    setattr(target_layer, f"{matrix_name}_svd", {"U": U, "S": S, "Vh": Vh})
    print(f"Stored SVD components for {matrix_name} in layer {layer_number}.")

# Custom forward pass for summing SVD matrices dynamically
class CustomGraniteLayer(nn.Module):
    def __init__(self, original_layer, matrix_name):
        super().__init__()
        self.original_layer = original_layer
        self.matrix_name = matrix_name

    def forward(self, *args, **kwargs):
        # Reconstruct the matrix from SVD components during the forward pass
        if hasattr(self.original_layer, f"{self.matrix_name}_svd"):
            svd_components = getattr(self.original_layer, f"{self.matrix_name}_svd")
            U, S, Vh = svd_components["U"], svd_components["S"], svd_components["Vh"]

            # Retrieve input tensor
            inputs = args[0] if args else kwargs.get("hidden_states")
            norms = []

            print(inputs.size())

            # Compute the matrix-vector product for each singular vector
            for i in range(len(S)):
                singular_matrix = torch.outer(U[:, i], Vh[i, :])
                singular_output = torch.matmul(inputs.to(dtype=torch.float64), singular_matrix)
                norm = torch.norm(singular_output, dim=-1).mean().item()
                norms.append(norm)
                
            with open("norm_logs.jsonl", "a") as f:
                f.write(json.dumps(norms) + "\n")

            # Reconstruct the matrix
            reconstructed_matrix = sum(S[i] * torch.outer(U[:, i], Vh[i, :]) for i in range(len(S)))

            # Replace the weight dynamically in the forward pass
            if self.matrix_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                og = getattr(self.original_layer.self_attn, self.matrix_name).weight.data.clone()  # Use clone
                getattr(self.original_layer.self_attn, self.matrix_name).weight.data.copy_(reconstructed_matrix)  # Use copy_
            elif self.matrix_name in ['gate_proj', 'up_proj', 'down_proj']:
                og = getattr(self.original_layer.mlp, self.matrix_name).weight.data.clone()  # Use clone
                getattr(self.original_layer.mlp, self.matrix_name).weight.data.copy_(reconstructed_matrix)  # Use copy_
            
            difference = torch.norm(og - reconstructed_matrix).item()
            print(f"Difference between original and reconstructed matrix: {difference}")
            
            output = self.original_layer(*args, **kwargs)

            # Restore the original weight
            if self.matrix_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                getattr(self.original_layer.self_attn, self.matrix_name).weight.data.copy_(og)  # Use copy_
            elif self.matrix_name in ['gate_proj', 'up_proj', 'down_proj']:
                getattr(self.original_layer.mlp, self.matrix_name).weight.data.copy_(og)  # Use copy_

            return output

        # Call the original forward pass
        return self.original_layer(*args, **kwargs)

# Replace the layer dynamically with the custom forward pass
def replace_layer_with_custom(model, layer_number, matrix_name):
    """
    Replaces a specific layer in the model with a custom layer
    that computes the sum of SVD matrices during the forward pass.
    
    Args:
        model: The GraniteForCausalLM model.
        layer_number: The target layer number (0-based index).
        matrix_name: The matrix to replace ('q_proj', 'k_proj', etc.).
    """
    target_layer = model.model.layers[layer_number]
    custom_layer = CustomGraniteLayer(target_layer, matrix_name)
    model.model.layers[layer_number] = custom_layer
    print(f"Replaced layer {layer_number} with custom forward logic for {matrix_name}.")

# Function to validate output equivalence
def validate_outputs(model, layer_number, matrix_name, input_text):
    """
    Validates whether the outputs from the original and reconstructed matrices are equivalent.
    
    Args:
        model: The GraniteForCausalLM model.
        layer_number: The target layer number (0-based index).
        matrix_name: The matrix to validate ('q_proj', 'k_proj', etc.).
        input_text: The input text for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained("/new_data/experiments/ap-8b-p10-rhel13-data-id-2/hf_format/samples_10597250")  # Replace with actual tokenizer path
    inputs = tokenizer(input_text, return_tensors="pt")

    # Get the original output
    original_layer = model.model.layers[layer_number]
    if matrix_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        original_matrix = getattr(original_layer.self_attn, matrix_name).weight.clone()
    elif matrix_name in ['gate_proj', 'up_proj', 'down_proj']:
        original_matrix = getattr(original_layer.mlp, matrix_name).weight.clone()
    else:
        raise ValueError(f"Matrix name {matrix_name} is not valid.")

    original_output = model(**inputs).logits

    # Replace the layer with the custom forward logic
    replace_layer_with_custom(model, layer_number, matrix_name)

    # Get the reconstructed output
    reconstructed_output = model(**inputs).logits

    # Decode and print the outputs
    original_text = tokenizer.decode(torch.argmax(original_output[:, -1, :], dim=-1)[0], skip_special_tokens=True)
    reconstructed_text = tokenizer.decode(torch.argmax(reconstructed_output[:, -1, :], dim=-1)[0], skip_special_tokens=True)
    
    print(f"Decoded Output (Original): {original_text}")
    print(f"Decoded Output (Reconstructed): {reconstructed_text}")

    # Compare outputs
    difference = torch.norm(original_output - reconstructed_output).item()
    print(f"Difference between original and reconstructed outputs: {difference}")

    return difference

# Example Usage
model = AutoModelForCausalLM.from_pretrained("/new_data/experiments/ap-8b-p10-rhel13-data-id-2/hf_format/samples_10597250")  # Replace with actual model path
layer_number = 34  # Specify the layer number (0-based index)
matrix_name = "down_proj"  # Specify the matrix name (e.g., 'q_proj', 'k_proj', etc.)
input_text = "Let us go"  # Example input text

# Decompose and store the SVD components
decompose_and_store_svd(model, layer_number, matrix_name)

replace_layer_with_custom(model, layer_number, matrix_name)

# Process the dataset through the model
model.eval()
with torch.no_grad():
    for batch in data_loader:
        outputs = model(**batch, use_cache=False)
        print('Done!')

# # Validate the outputs
# difference = validate_outputs(model, layer_number, matrix_name, input_text)

# if difference < 1e-5:
#     print("Outputs are effectively the same!")
# else:
#     print("Outputs differ. Reconstruction may need adjustment.")

# # Save the model after decomposing and storing SVD components
# save_path = "./modified_model/"  # Specify the save directory
# GraniteWithSVD.save_pretrained(model, save_path)
# print(f"Model with SVD components saved to {save_path}")

# # Reload the model with SVD components
# reloaded_model = GraniteWithSVD.from_pretrained(save_path)
# print("Model reloaded with SVD components.")

# difference = validate_outputs(reloaded_model, layer_number, matrix_name, input_text)

# if difference < 1e-5:
#     print("Outputs are effectively the same!")
# else:
#     print("Outputs differ. Reconstruction may need adjustment.")