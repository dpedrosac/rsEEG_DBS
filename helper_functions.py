import mne
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# import matplotlib.pyplot as plt
from pathlib import Path

MONTAGE = mne.channels.make_standard_montage('standard_1005')
FLAG_CHECK = False
SCALE_FACTOR = 2

class General:
    def __init__(self, _flag_check=False):
        pass

    @staticmethod
    def default_folders(username):
        if username == 'dpedrosac':
            wdir = "/media/storage/rsEEG_DBS/"
        elif username == 'nahuel':
            wdir = "C:/Users/Nahuel/Desktop/rsEEG_DBS"

        save_dir = Path(wdir) / 'results'
        save_dir.mkdir(parents=True, exist_ok=True)  # create if it does not exist

        return wdir, save_dir

    @staticmethod
    def list_subjects(wdir):
        """
        Returns a list of folder names in the 'raw' directory of the specified working directory.

        Parameters
        ----------
        wdir : str or Path
            The path to the working directory.

        Returns
        -------
        list
            A list of folder names (last layer) in the 'raw' directory.
        """
        raw_dir = Path(wdir) / "raw"
        list_of_subjects = [f.name for f in raw_dir.iterdir() if f.is_dir()]

        return list_of_subjects

    @staticmethod
    def load_data(filename, montage=MONTAGE):
        raw_eeg = mne.io.read_raw_brainvision(filename, preload=True)  # Load BrainVision data
        raw_eeg.set_channel_types({'EKG': 'misc',
                                   'AccX': 'misc',
                                   'AccY': 'misc',
                                   'AccZ': 'misc',
                                   'LabJack': 'misc',
                                   'EMG': 'emg',
                                   'O9': 'misc',
                                   'O10': 'misc'})
        raw_eeg.set_montage(montage)
        sample_freq = raw_eeg.info["sfreq"]

        return raw_eeg, sample_freq

    @staticmethod
    def compute_fft(data_array: object,
                    sampling_rate: float):
        """Compute FFT and frequency vector.

        Parameters
        ----------
        data : object or np.ndarray
            The raw input data, as a NumPy array. Expected shape: (n_channels, n_samples).
        sampling_rate : float
            The sampling rate of the signal in Hz.

        Returns
        -------
        fft_results : np.ndarray
            An n_channels x n_samples (the FFT magnitude spectrum).
        frequencies : np.ndarray
            The frequency vector corresponding to the FFT results.


        """
        fft_result = SCALE_FACTOR * np.abs(np.fft.fft(data_array, axis=1))
        fft_length = data_array.shape[1]
        frequencies = sampling_rate / 2 * np.linspace(0, 1, fft_length // 2 + 1)
        return fft_result, frequencies

class DBSfilt:
    # Variables for the entire script
    SPIKE_ROWS = 8
    TYPE = 2            # Hampel identifier and refined spike identification
    HAMPEL_T = 2.5      # Hampel threshold for automatic spike detection.
    HAMPEL_L = 1        # windows size for automatic spike detection [in Hz]

    def __init__(self, _flag_check=False):
        pass

    def spike_detection(self, spikes,
                        type : float = TYPE,
                        HampelL : float = HAMPEL_L,
                        HampelT: float = HAMPEL_T,
                        FdbsL :float = 130,
                        FdbsR : float = 130,
                        nmax :float = 5,
                        eps : float = .01):

        # Mean spectrum and frequency resolution
        Y = spikes[1, :].copy()  # Mean spectrum
        Fres = spikes[0, 1]  # Frequency resolution

        # Window length and number of windows
        WL = round(HampelL / Fres)  # Window length in frequency space
        nbW = len(Y) // WL  # Number of full-length windows
        L = nbW * WL  # Length of reduced data

        # Partition data into windows
        Ya = Y[:L]  # Data for full windows
        Yb = Y[-WL:]  # Remaining data

        # Epoch the data
        Ya = Ya.reshape(nbW, WL)

        # Calculate the Hampel identifier for each epoch
        YaMedian = np.median(Ya, axis=1)
        YaThres = HampelT * 1.4286 * YaMedian

        # Transform thresholds to a linear array
        YaMedianLin = np.repeat(YaMedian, WL)
        YaThresLin = np.repeat(YaThres, WL)

        # Process last epoch
        YbMedian = np.median(Yb)
        YbThres = HampelT * 1.4286 * YbMedian

        YbMedianLin = np.full(WL, YbMedian)
        YbThresLin = np.full(WL, YbThres)

        # Merge epochs
        Ymedian = np.zeros_like(Y)
        Ythres = np.zeros_like(Y)

        Ymedian[:L] = YaMedianLin
        Ythres[:L] = YaThresLin

        Ymedian[-WL:] = YbMedianLin
        Ythres[-WL:] = YbThresLin

        # Update spikes matrix
        spikes[2, :] = Ymedian
        spikes[3, :] = Ythres
        spikes[4, :] = (Y > Ythres).astype(int)  # Spike detection
        spikes[5, :] = (Y > Ythres).astype(int)

        # Count spikes
        nb_spikes = np.sum(spikes[4, :])

        # Optional: Spike identification
        if type == 2 and FdbsL is not None and FdbsR is not None:
            SpikesIndex = np.where(spikes[4, :] == 1)[0]
            for spk in SpikesIndex:
                Fs = spikes[0, spk]
                if FdbsL - 0.6 < Fs < FdbsL + 0.5:
                    pass  # Placeholder for debugging

                dbs_induced, n, h = self.test_spike(Fs, FdbsL, FdbsR, nmax, eps, 0)

                spikes[5, spk] = dbs_induced
                spikes[6, spk] = n
                spikes[7, spk] = h

            # Recount spikes
            nb_spikes = np.sum(spikes[5, :])

        return spikes, nb_spikes

    def prepare_spike_detection(self, raw_data: object,
                                sampling_rate : float = None,
                                plot_fft : bool = FLAG_CHECK):
        """
        Prepares data for spike detection by processing FFT and organizing relevant outputs.

        Parameters
        ----------
        raw_data : object
            The raw input data, as an MNE object that supports `get_data()`.
            If an object, it should have a `get_data()` method to extract the signal matrix.
            Expected shape: (n_channels, n_samples).
        sampling_rate : float
            The sampling rate of the signal in Hz.
        plot_fft : bool, optional
            If True, plots the FFT magnitude spectrum for the first channel.

        Returns
        -------
        spikes : np.ndarray
            An 8xN matrix where the first row contains frequency values,
            and the second row contains the FFT magnitude spectrum.
        frequencies : np.ndarray
            The frequency vector corresponding to the FFT results.
        averaged_spectrum : np.ndarray
            The averaged FFT magnitude spectrum (mean across channels).

        Notes
        -----
        - If the number of samples in `raw_data` is odd, the last sample is discarded to
          ensure compatibility with FFT operations.
        - This function is designed for preprocessing EEG-like signals for spike detection.

        Examples
        --------
        >>> raw_data = np.random.randn(32, 1000)  # Example EEG data (32 channels, 1000 samples)
        >>> sampling_rate = 1000.0  # Sampling rate in Hz
        >>> spikes, frequencies, spectrum = prepare_spike_detection(raw_data, sampling_rate)
        >>> spikes.shape
        (8, 501)
        """
        logging.info("Preparing spike detection...")
        eeg_data_input = raw_data.get_data()
        if sampling_rate is None:
            sampling_rate = raw_data.info['sfreq']
        data_length = eeg_data_input.shape[1]

        # Parity check for fft estimation later
        fft_length = data_length - (data_length % 2)
        eeg_data_input = eeg_data_input[:, :fft_length]

        # Compute FFT using helper function
        Y, f = General.compute_fft(eeg_data_input, sampling_rate)
        logging.info(f"FFT result shape: {Y.shape}, Frequency vector length: {len(f)}.")

        if plot_fft:
            plt.figure(figsize=(10, 6))
            plt.plot(f, Y[0, :fft_length // 2 + 1], label='FFT Magnitude')
            plt.title('FFT Magnitude Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend()
            plt.show()

        # Split iff necessary
        Ym = Y if Y.ndim == 1 else np.mean(Y, axis=0)
        Ym1 = Ym[:fft_length // 2 + 1]

        # Prepare Spikes matrix
        spikes = np.zeros((8, len(f)))
        spikes[0, :] = f
        spikes[1, :] = Ym1
        logging.info("Spike detection preparation completed.")

        return spikes, f, Ym1

    @staticmethod
    def test_spike(
            Fs: float,
            FdbsL: float,
            FdbsR: float,
            nmax: int,
            eps: float,
            display: bool = False,
            dbs_induced: int = 1,
            n: int = -1,
            h: int = -1
    ):
        """
        Check if a detected spike at frequency `Fs` is a DBS aliased frequency.

        Parameters
        ----------
        Fs : float
            Detected spike frequency.
        FdbsL : float
            Left DBS frequency.
        FdbsR : float
            Right DBS frequency.
        nmax : int
            Maximum sub-multiple of the DBS frequency.
        eps : float
            Tolerance for h to be a positive integer.
        display : bool, optional
            Whether to display results (default is False).
        dbs_induced : int, optional
            Initial state for DBS induced flag (default is 1).
        n : int, optional
            Initial sub-multiple of the DBS frequency (default is -1).
        h : int, optional
            Initial harmonic of the aliased frequency (default is -1).

        Returns
        -------
        dbs_induced : int
            1 if the spike is a DBS aliased frequency, 0 otherwise.
        n : int
            Sub-multiple of the DBS frequency, -1 if not DBS induced.
        h : int
            Harmonic of the aliased frequency, -1 if not DBS induced.
        """

        def compute_harmonic_and_stop(ratio: float, n: int) -> tuple:
            """Calculate harmonic and stop condition for a given ratio and n."""
            h1 = n / ratio
            h = round(h1)
            stop = abs(h1 - h)
            return h, stop

        # Determine whether single or dual DBS frequencies are provided
        ratios = [FdbsL / Fs] if FdbsL == FdbsR else [FdbsL / Fs, FdbsR / Fs]

        for ratio in ratios:
            stop = 1
            n = 0
            while stop > eps:
                n += 1
                if n >= nmax:
                    dbs_induced, n, h = 0, -1, -1
                    break
                h, stop = compute_harmonic_and_stop(ratio, n)

            if stop <= eps:
                # Valid harmonic found
                if display:
                    logging.info(f"[testspike] n={n}, h={h}, DBS induced={dbs_induced} (Freq = {Fs / ratio:.2f})")
                break

        return dbs_induced, n, h

    def spike_removal(self, raw_eeg_data : object, sampling_rate: float, spikes):
        """
        Process signal by performing FFT, spike detection, and signal reconstruction.

        Parameters
        ----------
        x : np.ndarray
            Input signal, shape (channels, samples).
        sr : float
            Sampling rate of the signal in Hz.
        spikes : np.ndarray
            Spikes matrix with spike information. Row 6 should indicate detected spikes.

        Returns
        -------
        x : np.ndarray
            Reconstructed signal after spike processing and interpolation.
        """

        if not sampling_rate:
            sampling_rate = raw_eeg_data.info['sfreq']
        data_length = raw_eeg_data.shape[1]

        # Parity check for fft estimation later
        fft_length = data_length - (data_length % 2)
        raw_eeg_data = raw_eeg_data[:, :fft_length]

        # Compute FFT using helper function
        Y, f = General.compute_fft(raw_eeg_data, sampling_rate)
        logging.info(f"FFT result shape: {Y.shape}, Frequency vector length: {len(f)}.")

        # Split FFT results
        Y1 = Y[:, :fft_length // 2 + 1]
        Y2 = np.hstack((Y1[:, :1], np.fliplr(Y)[:, :fft_length // 2]))

        # Step 3: Spike detection and interpolation
        detected_spikes = np.flatnonzero(spikes[5, :])
        wl_samples = round((1 / f[1] - 1) / 2)  # Frequency resolution is f[1], window length = 1 Hz

        for spike_idx in detected_spikes:
            start_idx = max(spike_idx - wl_samples, 0)
            end_idx = min(spike_idx + wl_samples + 1, Y1.shape[1])

            Y1_window = Y1[:, start_idx:end_idx]
            Y2_window = Y2[:, start_idx:end_idx]

            # Interpolate spikes with median value
            Y1[:, spike_idx] = np.median(Y1_window, axis=1)
            Y2[:, spike_idx] = np.median(Y2_window, axis=1)

        # Step 4: Signal reconstruction
        Y2 = np.fliplr(Y2)
        Y_combined = np.hstack((Y1[:, :-1], Y2[:, :-1]))
        data_restored = np.fft.ifft(Y_combined, axis=1)  # Use irfft for real-valued reconstruction

        # Restore original data length if adjusted
        if data_length > fft_length:
            last_sample = data_restored[:, -1] + (data_restored[:, -1] - data_restored[:, -2])  # Linear extrapolation
            data_restored = np.c_[data_restored, last_sample]  # Append last_sample as a new column

        return data_restored


    def compare_spectra(raw_eeg: float,
                        data_restored : float,
                        sampling_rate: float = None):
        """
        Compare the spectra of raw and processed data.

        Parameters
        ----------
        raw_data : np.ndarray
            The raw input signal (channels x samples).
        processed_data : np.ndarray
            The processed signal (channels x samples).
        sr : float
            Sampling rate of the signals in Hz.

        Returns
        -------
        None
        """

        if not sampling_rate:
            sampling_rate = raw_eeg.info['sfreq']

        # Compute FFT
        raw_fft = np.fft.rfft(raw_eeg, axis=1)
        data_restored = np.array(data_restored, dtype=float)
        processed_fft = np.fft.rfft(data_restored, axis=1)

        # Compute magnitude spectra
        raw_magnitude = np.abs(raw_fft)
        processed_magnitude = np.abs(processed_fft)

        # Frequency axis
        freqs = np.fft.rfftfreq(raw_eeg.shape[1], d=1 / sampling_rate)

        # Plot spectra for the first channel
        plt.figure(figsize=(12, 6))
        plt.plot(freqs, raw_magnitude[47], label="Raw Data", alpha=0.7)
        plt.plot(freqs, processed_magnitude[47], label="Processed Data", alpha=0.7)
        plt.title("Frequency Spectrum Comparison (Channel 1)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Quantify difference (mean squared error)
        mse = np.mean((raw_magnitude - processed_magnitude) ** 2, axis=1)
        print(f"Mean Squared Error (MSE) between spectra: {mse}")