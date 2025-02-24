from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np
import mne

# Load the EDF file using MNE
def load_edf_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    return raw

# Normalize the signals
def normalize_signals(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.T).T

# Perform PCA
def perform_pca(data, n_components=4):
    n_components = min(n_components, min(data.shape))  # Ensure valid n_components
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data)

    # Sort PCA components based on explained variance
    sorted_idx = np.argsort(pca.explained_variance_ratio_)[::-1]
    pca_components_sorted = pca_components[:, sorted_idx]

    # Correct polarity if needed
    for i in range(pca_components_sorted.shape[1]):
        if np.mean(pca_components_sorted[:, i]) > 0:
            pca_components_sorted[:, i] = -pca_components_sorted[:, i]  # Flip the signal if its mean is positive

    plot_pca_variance(pca)
    return pca_components_sorted

# Perform ICA
def perform_ica(data, n_components=4):
    n_components = min(n_components, min(data.shape))  # Ensure valid n_components
    ica = FastICA(n_components=n_components)
    ica_components = ica.fit_transform(data)

    # Sort ICA components by the sum of absolute values (most contribution first)
    sorted_idx = np.argsort(np.sum(np.abs(ica_components), axis=0))[::-1]
    ica_components_sorted = ica_components[:, sorted_idx]

    # Correct polarity if needed
    for i in range(ica_components_sorted.shape[1]):
        if np.mean(ica_components_sorted[:, i]) > 0:
            ica_components_sorted[:, i] = -ica_components_sorted[:, i]  # Flip the signal if its mean is positive

    return ica_components_sorted

# Plot the explained variance ratio of PCA components
def plot_pca_variance(pca):
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.7, label="Individual Variance")

    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.show()

# Compute Power Spectral Density (PSD)
def compute_psd(components, fs=500):
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

# Save results to a file
def save_results(filename, data):
    np.savetxt(filename, data, delimiter=",")

# Plot each component in a separate figure
def plot_components(components, title_prefix):
    time = np.arange(components.shape[0])
    num_components = components.shape[1]
    fig, axes = plt.subplots(num_components, 1, figsize=(12, 3 * num_components), sharex=True)

    if num_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, components[:, i])
        ax.set_title(f'{title_prefix} Component {i + 1}')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

    plt.xlabel('Time (samples)')
    plt.tight_layout()
    plt.show()

# Plot original signal
def plot_original_signal(data, channel_names):
    time = np.arange(data.shape[1])
    num_channels = len(channel_names)

    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels), sharex=True)

    # If only one channel, make axes a list for uniform handling
    if num_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, data[i, :])
        ax.set_title(f'{channel_names[i]}')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

    plt.xlabel('Time (samples)')
    plt.tight_layout()
    plt.show()


def main(file_path, n_components=4):
    # Load the EDF file
    raw_data = load_edf_file(file_path)

    # Select channels: Assuming fetal ECG is first and maternal ECG follows
    channel_names = raw_data.ch_names
    fetal_ecg_channel = [channel_names[0]]  # Assuming first channel is fECG
    abdominal_ecg_channels = channel_names[1:5]  # Assuming next four channels are mECG

    # Apply filtering specific to ECG
    raw_data.filter(l_freq=0.5, h_freq=40.0, picks="all", method="fir")

    # Extract data separately
    fetal_data = raw_data.copy().pick(fetal_ecg_channel).get_data()
    abdominal_data = raw_data.copy().pick(abdominal_ecg_channels).get_data()

    # Plot original signals
    plot_original_signal(fetal_data, fetal_ecg_channel)
    plot_original_signal(abdominal_data, abdominal_ecg_channels)

    # Normalize the signals before PCA and ICA
    abdominal_data = normalize_signals(abdominal_data)

    # Perform PCA and ICA on maternal ECG
    pca_maternal = perform_pca(abdominal_data.T, n_components)
    ica_maternal = perform_ica(abdominal_data.T, n_components)
    print("PCA and ICA on abdominal ECG completed")

    # Save PCA and ICA results
    save_results("pca_results.csv", pca_maternal)
    save_results("ica_results.csv", ica_maternal)

    # Plot PCA and ICA components
    plot_components(pca_maternal, 'PCA - Abdominal ECG')
    plot_components(ica_maternal, 'ICA - Abdominal ECG')

    print("Analyzing frequency content to identify fetal ECG...")
    compute_psd(ica_maternal)

    print("Fetal ECG extraction complete!")

if __name__ == "__main__":
    # Replace with your EDF file path
    file_path = "C:/Users/kimsv/OneDrive - Mälardalens universitet/Desktop/r01.edf"
    main(file_path)
