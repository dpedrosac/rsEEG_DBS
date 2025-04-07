# Standard Library
import os
import logging
from pathlib import Path

import mne.io
from mne import export, make_fixed_length_epochs

# Third-Party Libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Helper Functions (Project-Specific)
from helper_functions import General, DBSfilt, MONTAGE


general = General() # load helper functions needed in the script

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("logs/pipelineEpochedTFR.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)  # Create a logger instance

wdir,save_dir  = general.default_folders(os.getenv("USERNAME"))
list_of_subjects = general.list_subjects(wdir, subfolder="results")
DURATION, NYQUIST = 4,125
FLAG_CHECK=False

list_off_conditions_dict = { 'rsEEG_01': 'DBS_Rest_1_cleaned_TFR.fif',
                             'rsEEG_04': 'Rest_Off_cleaned_TFR.fif',
                             'rsEEG_06': 'Rest_cleaned_TFR.fif',
                             'rsEEG_13': 'DBS_13_rest2_cleaned_TFR.fif',
                             'rsEEG_15': 'DBS_15_rest_cleaned_TFR.fif',
                             'rsEEG_16': 'DBS_16_OFF_cleaned_TFR.fif',
                             'rsEEG_18': 'rest_cleaned_TFR.fif',
                             'rsEEG_19': 'DBS_20_rest_cleaned_TFR.fif',
                             'rsEEG_20': 'DBS_21_rest_cleaned_TFR.fif',
                             'rsEEG_21': 'DBS_22_Rest_cleaned_TFR.fif',
                             'rsEEG_22': 'DBS_23_Rest_cleaned_TFR.fif',
                             'rsEEG_23': 'DBS_24_Rest_cleaned_TFR.fif',
                             'rsEEG_24': 'DBS_25_Rest_cleaned_TFR.fif',
                             'rsEEG_25': 'DBS_25_rest_cleaned_TFR.fif' }

listTFR = []
for subj in list_of_subjects:
    subject_path = wdir / Path("results") / Path(subj)
    if not subject_path.exists():
        print(f"Directory does not exist: {subject_path}")
        continue

    try:
        filename_temp = wdir / Path("results") / Path(subj) / list_off_conditions_dict[subj]
    except KeyError:
        continue

    if not filename_temp.exists():  # Check if the output file exists
        continue

    fig, axes = plt.subplots(1, 2, figsize=(7, 4), layout="constrained")
    fig.suptitle('Subj: {}'.format(subj), fontsize=16)
    topomap_kw = dict(
        ch_type="eeg", tmin=0.5, tmax=1.5, baseline=(0, 0.5), mode="logratio", show=False
    )
    plot_dict = dict(Theta=dict(fmin=30, fmax=45), Beta=dict(fmin=13, fmax=30))
    for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
        mne.time_frequency.read_tfrs(filename_temp).plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
        ax.set_title(title)
    plt.show()

    #listTFR.append(mne.time_frequency.read_tfrs(filename_temp))

power = mne.grand_average(listTFR)
power.plot_topo(baseline=(0, 0.5), mode="percent", title="Average power")
power.plot(picks=[82], baseline=(0, 0.5), mode="percent", title=power.ch_names[82])

fig, axes = plt.subplots(1, 2, figsize=(7, 4), layout="constrained")
topomap_kw = dict(
    ch_type="eeg", tmin=0.5, tmax=1.5, baseline=(0, 0.5), mode="logratio", show=False
)
plot_dict = dict(Theta=dict(fmin=30, fmax=45), Beta=dict(fmin=13, fmax=30))
for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
    power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
    ax.set_title(title)