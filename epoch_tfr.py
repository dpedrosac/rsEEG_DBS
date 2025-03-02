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
dbsfilter = DBSfilt()

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("test/pipeline1.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)  # Create a logger instance

wdir,save_dir  = general.default_folders(os.getenv("USERNAME"))
list_of_subjects = general.list_subjects(wdir, subfolder="results")
DURATION, NYQUIST = 4,125
FLAG_CHECK=False

def process_pipelineEpochTFR(cleaned_eeg, subj_ID, flag_check=FLAG_CHECK, duration=DURATION, method='multitaper',
                             freq_spacing='lin', time_window_length=.5, freqs=[3, 90, 50], freq_bandwidth=4):
    """
        Processes EEG data for a given subject and input file.

        This function loads processed EEG data (cf. analysis_pipeline.py for more details, epochs it to the desired
        duration and performs TFR analysis for further steps. All processed data is saved to a new file.

        Parameters
        ----------
        input_filename : Path
            Path to the input `.vhdr` file.
        subj : str
            Subject identifier (e.g., "rsEEG_15").
        flag_check : bool, optional
            Whether to enable debugging checks and plots. Default is False.
        duration : float, optional
            duration of resulting epochs (non-overlapping).
        method : str, optional
            method used to estimate EEGs TFR.
        freq_spacing : str, optional
            spacing of frequencies either linearly or logarithmic
        time_window_length : float, optional
            time window to estimate the cycles in the script
        freqs : float
            frqeuncies on which the TFR will be estimated at

        Returns
        -------
        No return

        Examples
        --------
        >>> from pathlib import Path
        >>> process_pipelineEpochTFR(Path("subject1.vhdr"), "subject1", flag_check=True, duration=4, method="multitaper")

        Notes
        -----
        - The input file must be in BrainVision format.
        - Processed files are saved under the `save_dir` folder.
        """

    output_filenameEpochs = save_dir / subj_ID / f"{cleaned_eeg.stem}_epo.fif"
    if not output_filenameEpochs.exists(): # Check if the output file does not exist, otherwise epoch data
        logger.warning(f"File inexistent: {output_filenameEpochs}. Re-running analysis...")
        logger.info(f"Epoching data to chunks of: {duration} secs. duration")
        logger.info(f"\tLoading data from {cleaned_eeg}")
        raw_eeg, _ = general.load_data_brainvision(filename=cleaned_eeg)
        epoched_rsdata = make_fixed_length_epochs(raw_eeg, duration=duration, preload=False)

        if flag_check:
            logger.debug("Plotting example data for debugging")
            epoched_rsdata.plot_image(picks=["C3"])
        epoched_rsdata.set_montage(MONTAGE)
        epoched_rsdata.save(fname=output_filenameEpochs)
    else:
        epoched_rsdata = mne.read_epochs(output_filenameEpochs)

    output_filenameTFR = save_dir / subj_ID / f"{cleaned_eeg.stem}_TFR.fif"
    if not output_filenameTFR.exists():  # Check if the output file exists
        logger.warning(f"File inexistent: {output_filenameTFR}. Re-running TFR analysis...")
        logger.info(f"Running TFR analysis with {method} method")
        logger.info(f"\tLoading data from {cleaned_eeg}")

        # define hyperparameters
        if freq_spacing == 'lin':
            freq = np.linspace(*freqs)
        elif freq_spacing == 'log': # TODO: not working fior the time being
            freq = np.logspace(*np.log10(freqs[:2]), freqs[2])
        n_cycles = freq * time_window_length  # set n_cycles based on fixed time window length
        time_bandwidth = time_window_length * freq_bandwidth  # must be >= 2

        tfr_epoched = epoched_rsdata.compute_tfr(method=method, freqs=freq, n_cycles=n_cycles,
                                                 time_bandwidth=time_bandwidth, average=True, return_itc=False)  # tfr of epoched data
        mne.time_frequency.write_tfrs(fname=output_filenameTFR, tfr=tfr_epoched)
        logger.info(f"\tEpoching data and TFR analyses completed for subj: {subj_ID}")
    else:
        logger.info(f"\tEpoching data and TFR analyses completed for subj: {subj_ID}")
        # mne.time_frequency.read_tfrs(fname=output_filenameTFR, tfr=tfr_epoched)

for subj in list_of_subjects:
    subject_path = wdir / Path("results") / Path(subj)
    if not subject_path.exists():
        print(f"Directory does not exist: {subject_path}")
        continue

    # Print filtered files
    print(f"Processing data for {subj}:")

    results_path = Path(wdir) / 'results' / subj
    results_path.mkdir(parents=True, exist_ok=True)  # create if it does not exist

    # List files and filter for .vhdr files for this subject
    list_of_files = [f for f in os.listdir(subject_path) if f.endswith('cleaned.vhdr')]

    for recording in list_of_files:
        input_filename = subject_path / recording
        process_pipelineEpochTFR(input_filename, subj)




#
# flag_check = False
#
# df_continuous_OFF = '/media/storage/rsEEG_DBS/results/rsEEG_15/DBS_15_0_cleaned.vhdr'
# df_continuous_ON = '/media/storage/rsEEG_DBS/results/rsEEG_15/DBS_15_4_cleaned.vhdr'
# raw_eegOFF = mne.io.read_raw_brainvision(df_continuous_OFF, preload=True)  # Load BrainVision data
# raw_eegON = mne.io.read_raw_brainvision(df_continuous_ON, preload=True)  # Load BrainVision data
#
# df_epochedOFF = mne.make_fixed_length_epochs(raw_eegOFF, duration=4, preload=True)
# df_epochedOFF = mne.Epochs.drop(df_epochedOFF, indices=[0,len(df_epochedOFF)-1]) # drop first and last epoch to remove artefacts
#
# df_epochedON = mne.make_fixed_length_epochs(raw_eegON, duration=4, preload=True)
# df_epochedON = mne.Epochs.drop(df_epochedON, indices=[0,len(df_epochedON)-1]) # drop first and last epoch to remove artefacts
#
# if flag_check:
#     df_epochedOFF.plot_image(picks=["C3"])
#     df_epochedON.plot_image(picks=["C3"])
#
# tfrOFF = df_epochedOFF.compute_tfr(method="multitaper", freqs=np.arange(1, 100), n_cycles=np.arange(1, 100),
#                                    average=False, return_itc=False) #df_epoched
# tfrON = df_epochedON.compute_tfr(method="multitaper", freqs=np.arange(1, 100), n_cycles=np.arange(1, 100),
#                                    average=False, return_itc=False) #df_epoched
#
#
#
# bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
#          'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
#          'Gamma (30-45 Hz)': (30, 45)}
# montage = mne.channels.make_standard_montage('standard_1005')
# df_epochedOFF.set_channel_types({'EKG' : 'misc',
#                        'AccX' : 'misc',
#                        'AccY' : 'misc',
#                        'AccZ' : 'misc',
#                        'LabJack' : 'misc',
#                        'EMG' : 'emg',
#                        'O9' : 'misc',
#                        'O10' : 'misc'})
# df_epochedOFF.set_montage(montage)
#
# df_epochedOFF.compute_psd().plot_topomap(bands=bands, normalize=True, cmap='coolwarm')
#
# df_epochedON.set_channel_types({'EKG' : 'misc',
#                        'AccX' : 'misc',
#                        'AccY' : 'misc',
#                        'AccZ' : 'misc',
#                        'LabJack' : 'misc',
#                        'EMG' : 'emg',
#                        'O9' : 'misc',
#                        'O10' : 'misc'})
# df_epochedON.set_montage(montage)
#
# df_epochedON.compute_psd().plot_topomap(bands=bands, normalize=True, cmap='coolwarm')
#
#
# extrapolations = ['local']
# fig, axes = plt.subplots(2,3, figsize=(15,10))
# axes = axes.flatten()
# all_band_data = []
#
# for i, (band, (llim, hlim)) in enumerate(freq_bands.items()):
#     raw_band = tfrOFF.copy().filter(l_freq=llim, h_freq=hlim, method='iir', verbose=True)
#     freqs = raw_band.get_data()[:, 1]
#
#     all_band_data.append(freqs)
#     im, _ = mne.viz.plot_topomap(freqs, raw_band.info, axes=axes[i], cmap='Spectral_r', contours=5, show=False) # names=channels missing
#     axes[i].set_title(f'{band}-Band')
#
# vmin = float('inf')
# vmax = float('-inf')
#
# for freq in range(len(freqs)):
#     vmin = min(vmin, np.min(freqs[freq]))
#     vmax = min(vmax, np.max(freqs[freq]))
#
# norma = plt.Normalize(vmin=vmin, vmax=vmax)
# cbar_ax = fig.add_axes([.92, .15, .02, .7])
# cbar = plt.colorbar(im, cax=cbar_ax, norm=norma)
# cbar.ax.set_title('mV²', fontsize=12)