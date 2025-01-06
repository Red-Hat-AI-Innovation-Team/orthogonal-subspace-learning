import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

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
layer_number = 10  # Specify the layer number (0-based index)
matrix_name = "k_proj"  # Specify the matrix name (e.g., 'q_proj', 'k_proj', etc.)
input_text = "Let us go"  # Example input text

# Decompose and store the SVD components
decompose_and_store_svd(model, layer_number, matrix_name)

# Validate the outputs
difference = validate_outputs(model, layer_number, matrix_name, input_text)

if difference < 1e-5:
    print("Outputs are effectively the same!")
else:
    print("Outputs differ. Reconstruction may need adjustment.")