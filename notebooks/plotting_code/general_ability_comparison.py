import matplotlib.pyplot as plt
import numpy as np

# Dataset labels
datasets = ['MMLU', 'GSM', 'BBH', 'TydiQA', 'BoolQA', 'PIQA']
x = np.arange(len(datasets))

# Baseline (from image: pick the one to represent 'Base Instruct Model', e.g., Replay)
base = [46.56, 26.08, 40.23, 23.47, 70.55, 76.22]  # You can switch to LoRASeqFT or Replay if needed
ours = [47.69, 7.71, 34.21, 35.81, 76.64, 77.58]      # Your method

# Bar configuration
bar_width = 0.35
offsets = [-bar_width/2, bar_width/2]

plt.figure(figsize=(22, 16))

# Plot only Base vs Ours
plt.bar(x + offsets[0], base, width=bar_width, label='Base Instruct Model', hatch='+', color='#1f78b4', edgecolor='black', alpha=0.8)
plt.bar(x + offsets[1], ours, width=bar_width, label='Ours (Adaptive SVD)', hatch='-', color='#66c2a5', edgecolor='black', alpha=0.8)

# Add value labels
for i, scores in enumerate([base, ours]):
    for j, score in enumerate(scores):
        plt.text(x[j] + offsets[i], score + 0.7, f"{score:.1f}", ha='center', va='bottom', fontsize=24)

# Aesthetics
plt.xticks(x, datasets, fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel('Score', fontsize=32)
plt.title('General Ability Benchmark Comparison', fontsize=32)
plt.ylim(0, 85)
plt.legend(fontsize=28)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("general_ability.pdf", format="pdf", dpi=300, bbox_inches='tight')