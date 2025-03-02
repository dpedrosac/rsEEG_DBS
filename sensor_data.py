import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import PCA
from helper_functions import General

# Define left-side electrodes and stimulation levels
electrodes = [f'c{i}' for i in range(1, 9)]
stimulation_levels = [f"{x:.1f}mA" for x in np.arange(0.0, 5.5, 0.5)]
electrode_positions_subplot = {
    "c8": (0, 1),  # First row, center
    "c5": (1, 0), "c6": (1, 1), "c7": (1, 2),  # Second row
    "c2": (2, 0), "c3": (2, 1), "c4": (2, 2),  # Third row
    "c1": (3, 1)   # Fourth row, center
}

def compute_metric(values, metric='mean'):
    if metric == 'mean':
        return np.nanmean(values)
    elif metric == 'median':
        return np.nanmedian(values)
    elif metric == 'rms':
        return np.sqrt(np.nanmean(np.array(values) ** 2))
    else:
        raise ValueError("Invalid metric specified")


def process_data(subjects, metric='rms'):
    results = {}
    for subject in subjects:
        data_directory = Path(os.getcwd()) / "sensor_data" / subject
        rms_dict = {elec: {mA: [] for mA in stimulation_levels} for elec in electrodes}

        for elec in electrodes:
            for mA in stimulation_levels:
                # pattern = re.compile(rf"stnL_{elec}_{mA}.*dd.*\\.csv")
                pattern = re.compile(rf"stnL_{elec}_{mA}.*dd.*\.csv")  # Dynamic regex for each condition
                matching_files = [f for f in os.listdir(data_directory) if pattern.match(f)]
                temp_rms_list = []

                for file_data in matching_files:
                    # df = pd.read_csv(data_directory / file_data)
                    df = General.load_pkl_csv(data_directory / file_data)
                    print(f"Processing {file_data} for {elec} at {mA}")
                    pca = PCA(n_components=1)
                    principal_component = pca.fit_transform(df[["x", "y", "z"]][50:].dropna().values)[:, 0]
                    metric_value = compute_metric(principal_component, metric)
                    temp_rms_list.append(metric_value)

                if temp_rms_list:
                    rms_dict[elec][mA] = compute_metric(temp_rms_list, metric)

        results[subject] = rms_dict
    return results


# Function to process data for a single subject
def process_subject_parallel(subject_metric_tuple, normalise=True):
    subject, metric = subject_metric_tuple
    data_directory = Path(os.getcwd()) / "sensor_data" / subject
    rms_dict = {elec: {mA: [] for mA in stimulation_levels} for elec in electrodes}

    if normalise:
        tmp_rms_norm_list = []
        pattern_norm = re.compile(rf"stnL.*.0.0mA.*dd.*\.pkl")
        normalising_files = [f for f in os.listdir(data_directory) if pattern_norm.match(f)]
        if normalising_files == []:
            pattern_norm = re.compile(rf"stnL.*.nanmA.*dd.*\.pkl")
            normalising_files = [f for f in os.listdir(data_directory) if pattern_norm.match(f)]

        for normalise_data in normalising_files:
            # df_norm = pd.read_csv(data_directory / normalise_data)
            df_norm = General.load_pkl_csv(data_directory / normalise_data)
            pca = PCA(n_components=1)
            principal_component_norm = pca.fit_transform(df_norm[["x", "y", "z"]][50:].dropna().values)[:, 0]
            metric_value_norm = compute_metric(principal_component_norm, metric)
            tmp_rms_norm_list.append(metric_value_norm)

        metric_value_norm = compute_metric(tmp_rms_norm_list, metric="mean")

    for elec in electrodes:
        for mA in stimulation_levels:
            pattern = re.compile(rf"stnL_{elec}_{mA}.*dd.*\.pkl")
            matching_files = [f for f in os.listdir(data_directory) if pattern.match(f)]
            temp_rms_list = []

            for file_data in matching_files:
                df = General.load_pkl_csv(data_directory / file_data)
                # df = pd.read_csv(data_directory / file_data)
                print(f"Processing {file_data} for {elec} at {mA}")
                pca = PCA(n_components=1)
                principal_component = pca.fit_transform(df[["x", "y", "z"]][50:].dropna().values)[:, 0]
                metric_value = compute_metric(principal_component, metric)
                temp_rms_list.append(metric_value)

            metric_total = compute_metric(temp_rms_list, metric="mean")

            if normalise and matching_files != []:
                    metric_normalised = 1 / metric_value_norm * metric_total
                    rms_dict[elec][mA] = metric_normalised
            else:
                if temp_rms_list:
                    rms_dict[elec][mA] = metric_total

    return subject, rms_dict

# Function to process data in parallel
def process_data_parallel(subjects, metric='rms'):
    with Pool(processes=cpu_count()) as pool:
        results_list = pool.map(process_subject_parallel, [(subject, metric) for subject in subjects])

    return dict(results_list)

def plot_results(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    for subject, rms_dict in results.items():
        for elec in electrodes:
            rms_values = [
                np.nanmean(rms_dict[elec][mA]) if mA in rms_dict[elec] and rms_dict[elec][mA] else np.nan
                for mA in stimulation_levels
            ]
            ax.plot(stimulation_levels, rms_values, marker="o", linestyle="-", label=f"{subject} - {elec}")

    ax.set_title("RMS Values Across Stimulation Levels")
    ax.set_xlabel("Stimulation Level (mA)")
    ax.set_ylabel("Metric Value")
    ax.legend(loc="best", fontsize="small")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Function to plot results in a 1-3-3-1 layout
def plot_results_all(results):
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey=True)
    for ax in axes.flatten():
        ax.axis("off")  # Hide all subplots initially

    all_lines = []
    all_labels = []

    for subject, rms_dict in results.items():
        for elec, (row, col) in electrode_positions_subplot.items():
            ax = axes[row, col]
            ax.axis("on")

            rms_values = [
                np.nanmean(rms_dict[elec][mA]) if mA in rms_dict[elec] and rms_dict[elec][mA] else np.nan
                for mA in stimulation_levels
            ]
            line, = ax.plot(stimulation_levels, rms_values, marker="o", linestyle="-", label=subject)
            ax.set_title(f"{elec}", fontsize=10)
            ax.set_xticks(range(len(stimulation_levels)))
            ax.set_xticklabels(stimulation_levels, rotation=45, fontsize=8)

            if subject not in all_labels:  # Avoid duplicate legend entries
                all_lines.append(line)
                all_labels.append(subject)

    fig.suptitle("RMS Values Across Stimulation Levels for Each Electrode (1-3-3-1 Layout)", fontsize=14)
    fig.supxlabel("Stimulation Level (mA)")
    fig.supylabel("RMS Value")

    # Create a separate legend at the end
    fig.legend(all_lines, all_labels, loc='lower center', ncol=len(all_labels), fontsize='small')

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to fit legend
    plt.show()


# Identify subjects dynamically
subjects = [d for d in os.listdir(Path(os.getcwd()) / "sensor_data") if
            os.path.isdir(Path(os.getcwd()) / "sensor_data" / d)]

# Process and plot results
results = process_data_parallel(subjects, metric='rms')
plot_results_all(results)
