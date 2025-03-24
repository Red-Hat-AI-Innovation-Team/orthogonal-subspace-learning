import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(15, 8))

# --- Step 1: Original Weight Matrix ---
ax.text(1, 6, r"$W$", fontsize=16, ha='center', bbox=dict(facecolor='white', edgecolor='black'))
ax.annotate("", xy=(2, 6), xytext=(1.3, 6), arrowprops=dict(arrowstyle="->", lw=2))

# --- Step 2: SVD Block ---
ax.text(2.5, 6.4, "SVD", fontsize=12, ha='center')
ax.text(2.5, 6.0, r"$W = U \Sigma V^\top$", fontsize=14, ha='center')
ax.add_patch(patches.FancyBboxPatch((1.8, 5.7), 1.4, 1.2, boxstyle="round,pad=0.02", 
                                    edgecolor='black', facecolor='whitesmoke', alpha=0.6))

# --- Step 2b: Retention Ratio Block ---
ax.add_patch(patches.FancyBboxPatch((1.8, 4.2), 1.4, 1.2, boxstyle="round,pad=0.02",
                                    edgecolor='black', facecolor='whitesmoke', alpha=0.6))
ax.text(2.5, 4.8, "Use Retention Ratio\n(Importance Metric)", fontsize=11, ha='center', color='dimgray')

# --- Merged output path arrows ---
# From SVD and Retention Ratio to split node
ax.annotate("", xy=(3.2, 6.0), xytext=(3.2, 6.0), arrowprops=dict(arrowstyle="-", lw=0))
ax.annotate("", xy=(3.2, 5.4), xytext=(3.2, 5.7), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(3.2, 5.4), xytext=(3.2, 4.8), arrowprops=dict(arrowstyle="->", lw=2))

# Arrow splits to high and low rank blocks
ax.annotate("", xy=(3.9, 6.0), xytext=(3.2, 5.4), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(3.9, 4.8), xytext=(3.2, 5.4), arrowprops=dict(arrowstyle="->", lw=2))

# --- Step 3: High vs Low Subspace ---
# High-rank (frozen)
ax.text(4.5, 6.6, "High-Rank Subspace", fontsize=12, ha='center', color='blue')
ax.text(4.5, 6.2, r"$U_{high}, \Sigma_{high}, V_{high}$", fontsize=13, ha='center', color='blue')
ax.add_patch(patches.Rectangle((3.9, 5.7), 1.2, 0.6, edgecolor='blue', facecolor='lightblue', alpha=0.5))

# Low-rank (trainable)
ax.text(4.5, 5.2, "Low-Rank Subspace", fontsize=12, ha='center', color='green')
ax.text(4.5, 4.8, r"$U_{low}, \Sigma_{low}, V_{low}$", fontsize=13, ha='center', color='green')
ax.add_patch(patches.Rectangle((3.9, 4.3), 1.2, 0.6, edgecolor='green', facecolor='lightgreen', alpha=0.5))

# Arrows to next stage
ax.annotate("", xy=(5.3, 6.0), xytext=(5.1, 6.0), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(5.3, 4.8), xytext=(5.1, 4.8), arrowprops=dict(arrowstyle="->", lw=2))

# --- Step 4a: Frozen path (No Gradient Update) ---
ax.text(6.5, 6.4, "Frozen (Old Task)", fontsize=12, ha='center', color='blue')
ax.text(6.5, 6.0, "No Gradient Update", fontsize=11, ha='center', color='gray')
ax.add_patch(patches.Rectangle((6, 5.7), 1.0, 0.6, edgecolor='blue', facecolor='lightgray', alpha=0.4))

# --- Step 4b: Trainable path (Projected Gradient) ---
ax.text(6.5, 5.2, "Trainable (New Task)", fontsize=12, ha='center', color='green')
ax.text(6.5, 4.8, "Projected Gradient", fontsize=11, ha='center', color='gray')
ax.text(6.5, 4.5, r"$\nabla_{proj} = (I - UU^T)\nabla$", fontsize=12, ha='center', color='gray')
ax.add_patch(patches.Rectangle((6, 4.1), 1.0, 0.9, edgecolor='green', facecolor='palegreen', alpha=0.6))

# Arrows to reconstruction
ax.annotate("", xy=(7.2, 6.0), xytext=(7.6, 5.6), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(7.2, 4.9), xytext=(7.6, 5.3), arrowprops=dict(arrowstyle="->", lw=2))

# --- Step 5: Reconstruction ---
ax.text(8.5, 5.6, "Reconstructed W", fontsize=13, ha='center')
ax.text(8.5, 5.3, r"$\hat{W} = U_{high} \Sigma_{high} V_{high}^T + U_{low} \Sigma_{low} V_{low}^T$", 
        fontsize=12, ha='center')
ax.add_patch(patches.Rectangle((7.9, 5.1), 1.2, 0.6, edgecolor='black', facecolor='white', alpha=1.0))

# --- Gradient Projection Subspace View (Enhanced) ---
ax.text(12, 6.6, "Gradient Projection in Subspace", fontsize=13, ha='center')

# Old task subspace
ellipse1 = patches.Ellipse((11.5, 5.6), 2.4, 1.2, angle=15, color='lightblue', alpha=0.6)
ax.add_patch(ellipse1)
ax.text(11.0, 5.8, "Old Task\nSubspace", fontsize=10, ha='center', color='blue')

# New task subspace
ellipse2 = patches.Ellipse((12.0, 5.2), 2.4, 1.2, angle=-15, color='lightgreen', alpha=0.6)
ax.add_patch(ellipse2)
ax.text(12.8, 5.2, "New Task\nSubspace", fontsize=10, ha='center', color='green')

# Overlap region
ax.text(11.6, 5.4, "Shared\nSpan", fontsize=9, color='purple', ha='center')

# Projected gradient arrow
ax.arrow(12.0, 4.6, 0, -0.6, head_width=0.1, head_length=0.1, fc='darkgreen', ec='darkgreen')
ax.text(12.0, 3.8, r"$\nabla_{low}$", fontsize=12, ha='center', color='darkgreen')

# Geometry of new task minus shared span
ax.add_patch(patches.Ellipse((12.0, 3.2), 2.4, 1.2, angle=-15, color='lightgreen', alpha=0.6))
ax.add_patch(patches.Ellipse((11.5, 3.2), 1.2, 0.6, angle=0, color='white', alpha=1.0))
ax.text(12.0, 3.2, "New Task\nw/o Shared Span", fontsize=9, ha='center', color='green')

# Optional: Conceptual note at bottom
ax.text(7, 1.0, 
        "Learning happens in low-rank directions; high-rank subspace aligns with old task data", 
        fontsize=11, ha='center', style='italic', color='dimgray')

# Hide axes
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)
ax.axis('off')

# Title
fig.suptitle("Orthogonal Subspace Learning: SVD, Retention Ratio, and Subspace-Aware Gradient Projection", 
             fontsize=16, weight='bold')

# Save and show
plt.tight_layout()
plt.savefig("/mnt/data/orthogonal_subspace_learning_final_diagram_v3.png", dpi=300, bbox_inches='tight')
plt.show()