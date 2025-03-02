import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns

# Define left-side electrodes
electrodes = [f'c{i}' for i in range(1, 9)]  # c1 to c8

# Define stimulation levels (from 0.0mA to 5.0mA in 0.5mA steps)
stimulation_levels = [f"{x:.1f}mA" for x in np.arange(0.0, 5.5, 0.5)]

# Define data directory
data_directory = Path(os.getcwd()) / "sensor_data" / "48_MET_2512_PD"

# Preallocate RMS matrix (Electrode × Stimulation Level)
rms_matrix = np.full((len(electrodes), len(stimulation_levels)), np.nan)

# Dictionary to store RMS values grouped by (electrode, stimulation level)
rms_dict = {elec: {mA: [] for mA in stimulation_levels} for elec in electrodes}

# Loop over electrodes and stimulation levels
for elec in electrodes:
    for mA in stimulation_levels:
        pattern = re.compile(rf"stnL_{elec}_{mA}.*dd.*\.csv")  # Dynamic regex for each condition

        # Find matching files for this electrode and stimulation level
        matching_files = [f for f in os.listdir(data_directory) if pattern.match(f)]

        temp_rms_list = []  # Temporary list to store all RMS values
        for file_data in matching_files:
            df = pd.read_csv(data_directory / file_data)
            print(f"Processing {file_data} for {elec} at {mA}")

            # Apply PCA and compute RMS of the first principal component
            pca = PCA(n_components=1)
            principal_component = pca.fit_transform(df[["x", "y", "z"]][50:].dropna().values)[:, 0]
            rms_pc = np.sqrt(np.mean(principal_component ** 2))

            # Store RMS value for this electrode and stimulation level
            temp_rms_list.append({"file": file_data, "rms": rms_pc})

        # Convert temporary list to DataFrame
        temp_df = pd.DataFrame(temp_rms_list)

        # Compute the mean RMS from all files and store in rms_dict
        if not temp_df.empty:
            rms_dict[elec][mA] = temp_df["rms"].mean()  # Store mean RMS in the dictionary

print("RMS computation complete. Matrix ready for heatmap visualization.")

# Define the 1-3-3-1 spatial layout for electrodes
electrode_positions = {
    "c1": (0, 1),  # First row, center
    "c2": (1, 0), "c3": (1, 1), "c4": (1, 2),  # Second row
    "c5": (2, 0), "c6": (2, 1), "c7": (2, 2),  # Third row
    "c8": (3, 1)   # Fourth row, center
}

# Define stimulation levels (0.0mA to 5.0mA in 0.5mA steps)
stimulation_levels = [f"{x:.1f}mA" for x in np.arange(0.0, 5.5, 0.5)]

# Mock data for rms_dict (replace with actual computation)
# np.random.seed(42)  # For reproducibility
# rms_dict = {elec: {mA: np.random.uniform(0.1, 1.0) for mA in stimulation_levels} for elec in electrode_positions}

# Create a 3D matrix for the heatmaps (4 rows × 3 columns × 11 stimulation levels)
rms_grid = np.full((4, 3, len(stimulation_levels)), np.nan)  # Initialize with NaNs

# Fill the matrix with computed RMS values
for i, mA in enumerate(stimulation_levels):
    temp_grid = np.full((4, 3), np.nan)  # 2D temp grid for current stimulation level
    for elec, (row, col) in electrode_positions.items():
        if elec in rms_dict and mA in rms_dict[elec]:  # Ensure data exists
            temp_grid[row, col] = np.nanmean(rms_dict[elec][mA])
            #temp_grid[row, col] = rms_dict[elec][mA]  # Assign RMS value
    rms_grid[:, :, i] = temp_grid  # Store into the 3D matrix

# Set up subplots for heatmaps (3 rows × 4 columns)
fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten for easy indexing

# Plot each stimulation level's heatmap
for i, mA in enumerate(stimulation_levels):
    if i < len(axes):  # Avoid out-of-bounds errors
        sns.heatmap(rms_grid[:, :, i], annot=True, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor="black",
                    xticklabels=[], yticklabels=[], ax=axes[i])

        axes[i].set_title(f"{mA}")  # Set stimulation level as title

# Hide extra subplots if they exist
for i in range(len(stimulation_levels), len(axes)):
    axes[i].axis("off")

# Formatting
fig.suptitle("RMS Heatmaps in 1-3-3-1 Layout Across Stimulation Levels", fontsize=14)
plt.tight_layout()
plt.show()

# Other visualisation

# Define the 1-3-3-1 spatial layout for subplots
electrode_positions_subplot = {
    "c8": (0, 1),  # First row, center
    "c5": (1, 0), "c6": (1, 1), "c7": (1, 2),  # Second row
    "c2": (2, 0), "c3": (2, 1), "c4": (2, 2),  # Third row
    "c1": (3, 1)   # Fourth row, center
}

# Define stimulation levels (0.0mA to 5.0mA in 0.5mA steps)
stimulation_levels = [f"{x:.1f}mA" for x in np.arange(0.0, 5.5, 0.5)]

# Mock RMS dictionary (replace with actual computation)
# np.random.seed(42)
#rms_dict = {elec: {mA: np.random.uniform(0.1, 1.0) for mA in stimulation_levels} for elec in electrode_positions_subplot}

# Create a 4x3 grid for subplots
fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey=True)

# Hide all subplots initially
for ax in axes.flatten():
    ax.axis("off")

# Plot each electrode's response in its respective position
for elec, (row, col) in electrode_positions_subplot.items():
    ax = axes[row, col]
    ax.axis("on")  # Activate only the used subplots

    rms_values = [
        np.nanmean(rms_dict[elec][mA]) if mA in rms_dict[elec] and rms_dict[elec][mA] else np.nan
        for mA in stimulation_levels
    ]

    ax.plot(stimulation_levels, rms_values, marker="o", linestyle="-", label=elec)
    ax.set_title(f"{elec}", fontsize=10)
    ax.set_xticks(range(len(stimulation_levels)))  # Set proper tick locations
    ax.set_xticklabels(stimulation_levels, rotation=45, fontsize=8)  # Apply labels

# Formatting
fig.suptitle("RMS Values Across Stimulation Levels for Each Electrode (1-3-3-1 Layout)", fontsize=14)
fig.supxlabel("Stimulation Level (mA)")
fig.supylabel("RMS Value")
plt.tight_layout()
plt.show()

