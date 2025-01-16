# Standard Library
import os
import logging
from pathlib import Path

# Third-Party Libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mne import export, preprocessing, filter
from mne.io import RawArray
from mne.viz import use_browser_backend, plot_filter

# Helper Functions (Project-Specific)
from helper_functions import General, DBSfilt, MONTAGE
from pyprep.prep_pipeline import PrepPipeline

general = General() # load helper functions needed in the script
dbsfilter = DBSfilt()

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("test/pipeline.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)  # Create a logger instance

wdir,save_dir  = general.default_folders(os.getenv("USERNAME"))
list_of_subjects = general.list_subjects(wdir)
LPF, HPF = 1,90
FLAG_CHECK=False


def process_pipelineDBS(unprocessed_eed, subj_ID, flag_check=FLAG_CHECK, hpf=HPF, lpf=LPF):
    """
    Processes EEG data for a given subject and input file.

    This function loads EEG data, applies bandpass filtering, and performs spike detection
    and removal. The processed data is saved to a new file.

    Parameters
    ----------
    input_filename : Path
        Path to the input `.vhdr` file.
    subj : str
        Subject identifier (e.g., "rsEEG_15").
    flag_check : bool, optional
        Whether to enable debugging checks and plots. Default is False.
    hpf : float, optional
        High-pass filter frequency in Hz. Default is 90.0 Hz.
    lpf : float, optional
        Low-pass filter frequency in Hz. Default is 1.0 Hz.

    Returns
    -------
    Path
        Path to the saved processed file.

    Examples
    --------
    >>> from pathlib import Path
    >>> process_pipelineDBS(Path("subject1.vhdr"), "subject1", flag_check=True, hpf=100, lpf=0.5)

    Notes
    -----
    - The input file must be in BrainVision format.
    - Processed files are saved under the `save_dir` folder.
    """

    output_filename = save_dir / subj_ID / f"{unprocessed_eed.stem}_processed.vhdr"
    if output_filename.exists(): # Check if the output file exists
        logger.warning(f"Output file already exists: {output_filename}. Skipping...")
        return  # Skip further processing for this file

    try:
        logger.info(f"Loading data from {unprocessed_eed}")
        raw_eeg, sfreq = general.load_data(filename=unprocessed_eed)
    except FileNotFoundError:
        logger.warning(f"Error: Input file {unprocessed_eed} not found.")
        return

    logger.info(f"Applying filter: lpf={lpf}, hpf={hpf}")
    raw_filtered = raw_eeg.copy().filter(l_freq=lpf, h_freq=hpf)

    if flag_check:
        logger.debug("Generating PSD plots for debugging")
        raw_eeg.plot_psd(fmin=0, fmax=250, n_fft=2048, spatial_colors=True) # plot psd for debugging
        raw_filtered.plot_psd(fmin=0, fmax=250, n_fft=2048, spatial_colors=True) # plot psd for debugging

        filter_params = filter.create_filter( # mne toolbox
            raw_eeg.get_data(),
            raw_eeg.info["sfreq"],
            l_freq=lpf,
            h_freq=hpf)
        plot_filter(filter_params, sfreq=sfreq) # mne toolbox

    logger.info("Starting spike detection and removal")
    spikes, f, Ym1 = dbsfilter.prepare_spike_detection(raw_data=raw_filtered, sampling_rate=sfreq)
    spikes, nb_spikes = dbsfilter.spike_detection(spikes)
    raw_eeg_noDBS = dbsfilter.spike_removal(raw_eeg_data=raw_filtered.get_data(), sampling_rate=sfreq, spikes=spikes)

    raw_eeg_processed = RawArray(data=raw_eeg_noDBS, info=raw_eeg.info)

   # Plot PSD in two subplots
    logger.info("Generating PSD comparison plot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    raw_filtered.compute_psd(fmin=0, fmax=250, n_fft=2048, picks='eeg').plot(axes=axes[0])
    raw_eeg_processed.compute_psd(fmin=0, fmax=250, n_fft=2048, picks='eeg').plot(axes=axes[1])

    fig.suptitle(f"PSD Comparison: {unprocessed_eed.stem}", fontsize=16)
    ymin_total = min([
    value for pair in [axes[0].get_ylim(),axes[1].get_ylim()] for value in pair
    ])
    ymax_total = max([
    value for pair in [axes[0].get_ylim(),axes[1].get_ylim()] for value in pair
    ])

    axes[0].set_ylim([ymin_total, ymax_total])
    axes[1].set_ylim([ymin_total, ymax_total])

    plt.show()  # Execution pauses here until the plots are closed

    plt.pause(0.001)  # Allow plots to render
    logger.info("Please close the plot window to proceed.")

    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  # Pause loop until the plot window is closed

    # Ask user for confirmation
    logger.info("Prompting user for confirmation on data accuracy")
    user_input = input(f"Is the processed data for {unprocessed_eed.stem} accurate? (yes/no): ").strip().lower()
    if user_input != 'yes':
        logger.warning(f"User rejected processed data for {unprocessed_eed.stem}. Skipping save.")
        return
    else:
        export.export_raw(fname=output_filename, raw=raw_eeg_processed) # mne toolbox

    return output_filename

def process_pipelineArtefacts(filtered_eeg, subj_ID, flag_check=FLAG_CHECK):
    """
     Processes (filtered and DBS artefact freed) EEG data by removing bad channels and IC using PrepPipeline and ICA.

     Parameters
     ----------
     filtered_eeg : Path
         Path to the input `.vhdr` file. Assumes the file is preprocessed (output from `process_pipelineDBS`).
     subj_ID : str
         Subject identifier (e.g., "rsEEG_15").
     flag_check : bool, optional
         If True, enables intermediate visualizations for debugging. Default is False.

     Returns
     -------
     Path
         Path to the saved artifact-cleaned file.

     Notes
     -----
     - This function depends on `PrepPipeline` for artifact removal and `mne.preprocessing.ICA` for identifying
       and excluding components related to artifacts.
     - Downsamples the cleaned data to 220 Hz before saving.

     Examples
     --------
     >>> from pathlib import Path
     >>> process_pipelineArtefacts(Path("subject1_processed.vhdr"), "subject1", flag_check=True)
     """

    output_filename = save_dir / subj_ID / f"{filtered_eeg.stem}_cleaned.vhdr"
    if output_filename.exists(): # Check if the output file exists
        logger.warning(f"Output file already exists: {output_filename}. Skipping...")
        return  # Skip further processing for this file

    input_filenam = save_dir / subj_ID / f"{filtered_eeg.stem}_processed.vhdr" # output from (process_pipelineDBS)
    raw_noDBS, sfreq = general.load_data(filename=input_filenam)

    # Fit prep
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, 300 / 2, 50),
        "montage": MONTAGE
    }

    prep = PrepPipeline(raw_noDBS, prep_params, MONTAGE)
    prep.fit()

    if flag_check:
        matplotlib.use('TkAgg')
        for title, proj in zip(["Raw", "Interpolated"], [raw_noDBS, prep.raw]):
            with use_browser_backend("matplotlib"): # mne.viz toolbox
                fig = proj.plot(n_channels=24)
            # make room for title
            fig.subplots_adjust(top=0.9)
            fig.suptitle("{} data".format(title), size="xx-large", weight="bold")

    data_interpolated = prep.raw
    data_interpolated = data_interpolated.filter(l_freq=.75, h_freq=None)
    amount_variance_explain = .99
    ica_components = 40
    ica = preprocessing.ICA(n_components=ica_components, random_state=97, method='fastica', max_iter=800)
    ica.fit(data_interpolated)
    # Plot components
    fig = plt.figure()
    ica.plot_sources(data_interpolated)
    ica.plot_components()
    plt.show(block=False)  # Non-blocking show

    # Wait until all figure windows are closed
    while plt.get_fignums():  # Check if there are any open figure windows
        plt.pause(0.1)  # Small delay to allow event processing

    logger.info("All plot windows closed. Proceeding to user confirmation.")

    # Ask user for confirmation on components to be removed
    logger.info("Prompting user for confirmation on data accuracy")
    user_input = input("Please indicate the components to be removed (comma-separated values, e.g., 1,4,9): ").strip()

    # Loop until user confirms the input
    while True:
        confirm_input = input(f"You entered: {user_input}. Are you sure this is correct? (yes/no): ").strip().lower()
        if confirm_input == 'yes':
            logger.info(f"User confirmed the input: {user_input}")
            break
        elif confirm_input == 'no':
            user_input = input(
                "Please re-enter the components to be removed (comma-separated values, e.g., 1,4,9): ").strip()
        else:
            print("Invalid response. Please type 'yes' or 'no'.")
            continue

    if user_input is not None and user_input != 'None':         # At this point, user_input is confirmed
        logger.info(f"Final confirmed components to remove: {user_input}")
        ica.exclude = list(map(int, user_input.split(',')))  # Convert input to a list of integers
        ica.apply(data_interpolated)

    # Set EEG reference and downsample data
    preprocessed_data = data_interpolated.copy().set_eeg_reference(ref_channels="average")
    preprocessed_data = preprocessed_data.resample(sfreq=220)
    crop_factor = 4
    preprocessed_data = preprocessed_data.crop(tmin=crop_factor,
                                               tmax=preprocessed_data.times[preprocessed_data.n_times - 1] - crop_factor)

    # Save the processed data
    export.export_raw(fname=output_filename, raw=preprocessed_data, overwrite=True)
    logger.info(f"Processed data saved to {output_filename}")


for subj in list_of_subjects:
    subject_path = wdir / Path("raw") / Path(subj)
    if not subject_path.exists():
        print(f"Directory does not exist: {subject_path}")
        continue

    # Print filtered files
    print(f"Processing data for {subj}:")

    results_path = Path(wdir) / 'results' / subj
    results_path.mkdir(parents=True, exist_ok=True)  # create if it does not exist

    # List files and filter for .vhdr files for this subject
    list_of_files = [f for f in os.listdir(subject_path) if f.endswith('.vhdr')]

    for recording in list_of_files:
        input_filename = subject_path / recording

        output_preprocessDBS = process_pipelineDBS(input_filename, subj)
        output_preprocessArtefacts = process_pipelineArtefacts(input_filename, subj)

