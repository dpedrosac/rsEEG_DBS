import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

import os
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import sklearn
import pyprep

from pyprep.prep_pipeline import PrepPipeline
from pyprep.ransac import find_bad_by_ransac

flag_check = False


# Read data into workspace

sample_data_raw_file = Path.home() / 'Schreibtisch' / 'rsEEG' / 'rsEEG_15' / 'DBS_15_0.vhdr'
montage = mne.channels.make_standard_montage('standard_1005')

# Load the BrainVision data
raw = mne.io.read_raw_brainvision(str(sample_data_raw_file), preload=True)
raw.set_channel_types({'EKG' : 'misc',
                       'AccX' : 'misc',
                       'AccY' : 'misc',
                       'AccZ' : 'misc',
                       'LabJack' : 'misc',
                       'EMG' : 'emg',
                       'O9' : 'misc',
                       'O10' : 'misc'})
raw.set_montage(montage)
sample_freq = raw.info["sfreq"]

if flag_check:
    for title, proj in zip(["Original", "Average"], [False, True]):
        with mne.viz.use_browser_backend("matplotlib"):
            fig = raw.plot(proj=proj, n_channels=len(raw))
        # make room for title
        fig.subplots_adjust(top=0.9)
        fig.suptitle("{} reference".format(title), size="xx-large", weight="bold")

# Plot raw PSD
if flag_check:
    initial_psd = raw.plot_psd(fmin=0, fmax=250, n_fft=2048, spatial_colors=True)

# Remove drift artifacts:
if flag_check:
    for cutoff in (0.2, 0.5):
            raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
            with mne.viz.use_browser_backend("matplotlib"):
                fig = raw_highpass.plot(
                    duration=60, proj=False, n_channels=30, remove_dc=False
                )
            fig.subplots_adjust(top=0.9)
            fig.suptitle(f"High-pass filtered at {cutoff} Hz", size="xx-large", weight="bold")

    filter_params = mne.filter.create_filter(
        raw.get_data(),
        raw.info["sfreq"],
        l_freq=0.5,
        h_freq=None)

    from mne.viz import plot_filter
    plot_filter(filter_params, sfreq=raw.info['sfreq'])

raw.filter(l_freq=.5, h_freq=None)

# Fit prep
prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(50, 220 / 2, 50),
}

prep = PrepPipeline(raw, prep_params, montage)
prep.fit()
data_interpolated = prep.raw

if flag_check:
    for title, proj in zip(["Raw", "Interpolated"], [raw, prep.raw]):
        with mne.viz.use_browser_backend("matplotlib"):
            fig = proj.plot(n_channels=24)
        # make room for title
        fig.subplots_adjust(top=0.9)
        fig.suptitle("{} data".format(title), size="xx-large", weight="bold")

# Re-reference data
data_interpolated.set_eeg_reference("average")

if flag_check:
    initial_psd = raw.plot_psd(fmin=0, fmax=250, n_fft=2048, spatial_colors=True)
    interpolated_psd = data_interpolated.plot_psd(fmin=0, fmax=250, n_fft=2048, spatial_colors=True)

# Check for line noise:
def add_arrows(axes):
    """Add some arrows at 50 Hz and its harmonics."""
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (50, 100, 150, 200):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4) : (idx + 5)].max()
            ax.arrow(
                x=freqs[idx],
                y=y + 18,
                dx=0,
                dy=-12,
                color="red",
                width=0.1,
                head_width=3,
                length_includes_head=True,
            )

freqs = (50, 100, 150, 200)
data_interpolated = data_interpolated.notch_filter(freqs=freqs, picks = 'eeg',  filter_length = 'auto',
                                                   phase = 'zero-double', fir_design = 'firwin')

if flag_check:
    for title, data in zip(["Un", "Notch "], [raw, data_interpolated]):
        fig = data.compute_psd(fmax=250).plot(
            average=True, amplitude=False, picks="data", exclude="bads"
        )
        fig.suptitle(f"{title}filtered", size="xx-large", weight="bold")
        add_arrows(fig.axes[:2])


amount_variance_explain = .99
ica_components = 50
ica = mne.preprocessing.ICA(n_components=ica_components, random_state=97, max_iter=800)#, method='picard')
ica.fit(data_interpolated)

ica.exclude = [4,8]  # exclude components related to artifacts
#raw.load_data(preload=True )
ica.apply(data_interpolated)

preprocessed_data = data_interpolated.copy().set_eeg_reference(ref_channels="average")
# raw_avg_ref.plot()

# Downsample data
preprocessed_data = preprocessed_data.resample(sfreq=220)

