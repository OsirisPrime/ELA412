import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import mne

def load_results(filename):
    return np.loadtxt(filename, delimiter=",")

def plot_components(components, title_prefix):
    time = np.arange(components.shape[0])
    num_components = components.shape[1] if len(components.shape) > 1 else 1

    if num_components == 1:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(time, components)
        ax.set_title(f'{title_prefix} Component 1')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
    else:
        fig, axes = plt.subplots(num_components, 1, figsize=(12, 3 * num_components), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(time, components[:, i])
            ax.set_title(f'{title_prefix} Component {i + 1}')
            ax.set_ylabel('Amplitude')
            ax.grid(True)

    plt.xlabel('Time (samples)')
    plt.tight_layout()
    plt.show()

def flip_ecg_signal(ecg_signal):
    return -ecg_signal

def plot_explained_variance(pca_results):
    explained_variance = np.var(pca_results, axis=0)  # Compute variance of each component
    explained_variance_ratio = explained_variance / np.sum(explained_variance)  # Normalize to get ratio

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title("Explained Variance of PCA Components")
    plt.grid(True)
    plt.show()

def apply_lowpass_filter(ecg_signal, fs=1000):
    info = mne.create_info(ch_names=['ECG'], sfreq=fs, ch_types=['ecg'])
    raw = mne.io.RawArray(ecg_signal[np.newaxis, :], info)  # Convert to MNE RawArray
    raw.filter(l_freq=1, h_freq=50, method='fir', fir_design='firwin', phase='zero', picks=[0])  # 👈 Explicitly pick channel index 0
    return raw.get_data()[0]  # Return filtered ECG signal

# Compute Power Spectral Density (PSD)
def compute_psd(components, fs=1000):
    plt.figure(figsize=(12, 6))
    for i in range(components.shape[1]):
        freqs, psd = welch(components[:, i], fs=fs, nperseg=fs * 2)
        plt.semilogy(freqs, psd, label=f'ICA Component {i + 1}')

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Power Spectral Density of ICA Components")
    plt.legend()
    plt.grid()
    plt.show()

# R-Peak detection function (detects R-waves in the fetal ECG signal)
def detect_rr_peaks(ecg_signal, distance, height):
    peaks, properties = find_peaks(ecg_signal, distance=distance, height=height, prominence=0.6) # Detect peaks in the signal (R-peaks)
    return peaks

# Function to plot the maternal and fetal ECG with detected R-peaks as subplots
def plot_ecg_with_rr_peaks(ecg_signal_1, rr_peaks_1, ecg_signal_2, rr_peaks_2):
    # Create a figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot the maternal ECG with RR peaks in the first subplot
    axes[0].plot(ecg_signal_1, label="Maternal ECG", color='r')
    axes[0].plot(rr_peaks_1, ecg_signal_1[rr_peaks_1], "rx", label="Maternal R-peaks", markersize=8)
    axes[0].set_title("Maternal ECG with R-peaks")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    # Plot the fetal ECG with RR peaks in the second subplot
    axes[1].plot(ecg_signal_2, label="Fetal ECG", color='b')
    axes[1].plot(rr_peaks_2, ecg_signal_2[rr_peaks_2], "bx", label="Fetal R-peaks", markersize=8)
    axes[1].set_title("Fetal ECG with R-peaks")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    # Plot both maternal and fetal ECG signals overlapped in the third subplot
    axes[2].plot(ecg_signal_1, label="Maternal ECG", color='r', alpha=0.6)
    axes[2].plot(ecg_signal_2, label="Fetal ECG", color='b', alpha=0.6)
    axes[2].set_title("Overlap of Maternal and Fetal ECG")
    axes[2].set_xlabel("Time (samples)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

def main():
    pca_results = load_results("pca_results.csv")
    ica_results = load_results("ica_results.csv")

    plot_components(pca_results, 'PCA')
    plot_components(ica_results, 'ICA')

    print("Plotting explained variance for PCA components...")
    plot_explained_variance(pca_results)

    print("Analyzing frequency content to identify fetal ECG...")
    compute_psd(ica_results)

    # Manually select the maternal and fetal ECG component
    maternal_ecg_component_index = int(input("Enter the index of the maternal ECG component (Normally 0): "))
    fetal_ecg_component_index = int(input("Enter the index of the fetal ECG component (Normally 1): "))

    # Extract fetal ECG signal
    maternal_ecg = ica_results[:, maternal_ecg_component_index]
    fetal_ecg = ica_results[:, fetal_ecg_component_index]

    # Flip fetal ECG if necessary
    fetal_ecg = flip_ecg_signal(fetal_ecg)

    # Apply low-pass filtering (1–50 Hz) to remove high-frequency noise
    maternal_ecg = apply_lowpass_filter(maternal_ecg)
    fetal_ecg = apply_lowpass_filter(fetal_ecg)

    # Detect R-peaks (RR peaks) in the fetal ECG signal
    maternal_rr_peaks = detect_rr_peaks(maternal_ecg, distance=400, height=2)
    fetal_rr_peaks = detect_rr_peaks(fetal_ecg, distance=1, height=1.25)

    # Plot the cleaned fetal ECG with RR peaks
    plot_ecg_with_rr_peaks(maternal_ecg, maternal_rr_peaks, fetal_ecg, fetal_rr_peaks)
    print("Fetal ECG analysis complete!")

if __name__ == "__main__":
    main()
