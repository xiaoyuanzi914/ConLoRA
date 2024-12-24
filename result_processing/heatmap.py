import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example data: Replace with actual data
ConLoRA_result = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

LoRA_result = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# Calculate the min and max values for the common color scale
min_value = min(ConLoRA_result.min(), LoRA_result.min())
max_value = max(ConLoRA_result.max(), LoRA_result.max())

# Create the figure
plt.figure(figsize=(8, 12))

# Plot the first heatmap
plt.subplot(2, 1, 1)
ax1 = sns.heatmap(
    ConLoRA_result, 
    annot=True, 
    fmt=".2f", 
    cmap="Blues", 
    cbar_kws={'label': 'Accuracy'}, 
    vmin=min_value, 
    vmax=max_value
)
ax1.set_xlabel('α (Dirichlet Parameters)')
ax1.set_ylabel('Average Connectivity')
ax1.set_xticklabels(['0.1', '0.15', '0.2', '0.25'])
ax1.set_yticklabels(['3', '4', '5', '6'])

# Plot the second heatmap
plt.subplot(2, 1, 2)
ax2 = sns.heatmap(
    LoRA_result, 
    annot=True, 
    fmt=".2f", 
    cmap="Blues", 
    cbar_kws={'label': 'Accuracy'}, 
    vmin=min_value, 
    vmax=max_value
)
ax2.set_xlabel('α (Dirichlet Parameters)')
ax2.set_ylabel('Average Connectivity')
ax2.set_xticklabels(['0.1', '0.15', '0.2', '0.25'])
ax2.set_yticklabels(['3', '4', '5', '6'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

# Save the plot as a PNG file
plt.savefig("123.png")
