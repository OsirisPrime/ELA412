import mne
from sklearn.decomposition import PCA, FastICA
import numpy as np
import matplotlib.pyplot as plt

# Load the EDF file using MNE
def load_edf_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    return raw

# Perform PCA
def perform_pca(data, n_components=4):
    n_components = min(n_components, min(data.shape))  # Ensure valid n_components
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data)
    return pca_components

# Perform ICA
def perform_ica(data, n_components=4):
    n_components = min(n_components, min(data.shape))  # Ensure valid n_components
    ica = FastICA(n_components=n_components)
    ica_components = ica.fit_transform(data)
    return ica_components

# Plot each component in a separate figure
def plot_components(components, title_prefix):
    time = np.arange(components.shape[0])

    for i in range(components.shape[1]):
        plt.figure(figsize=(12, 4))
        plt.plot(time, components[:, i])
        plt.title(f'{title_prefix} Component {i + 1}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

# Plot original signal
def plot_original_signal(data, channel_names):
    time = np.arange(data.shape[1])
    plt.figure(figsize=(12, 6))
    for i in range(data.shape[0]):
        plt.plot(time, data[i, :], label=channel_names[i])
    plt.title('Direct Fetal ECG Signals')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(file_path, n_components=4):
    # Load the EDF file
    raw_data = load_edf_file(file_path)

    # Select channels: Assuming fetal ECG is first and maternal ECG follows
    channel_names = raw_data.ch_names
    fetal_ecg_channel = [channel_names[0]]  # Assuming first channel is fECG
    maternal_ecg_channels = channel_names[1:5]  # Assuming next four channels are mECG

    # Apply filtering specific to ECG
    #raw_data.filter(l_freq=0.5, h_freq=100.0)

    # Extract data separately
    fetal_data = raw_data.copy().pick_channels(fetal_ecg_channel).get_data()
    maternal_data = raw_data.copy().pick_channels(maternal_ecg_channels).get_data()

    # Plot original signals
    plot_original_signal(fetal_data, fetal_ecg_channel)
    plot_original_signal(maternal_data, maternal_ecg_channels)

    # Perform PCA and ICA on maternal ECG
    pca_maternal = perform_pca(maternal_data.T, n_components)
    ica_maternal = perform_ica(maternal_data.T, n_components)
    print("PCA and ICA on maternal ECG completed")

    # Perform PCA and ICA on fetal ECG
    pca_fetal = perform_pca(fetal_data.T, n_components=1)
    ica_fetal = perform_ica(fetal_data.T, n_components=1)
    print("PCA and ICA on fetal ECG completed")

    # Plot PCA and ICA components
    plot_components(pca_maternal, 'PCA - Maternal ECG')
    plot_components(ica_maternal, 'ICA - Maternal ECG')
    plot_components(pca_fetal, 'PCA - Fetal ECG')
    plot_components(ica_fetal, 'ICA - Fetal ECG')

if __name__ == "__main__":
    # Replace with your EDF file path
    file_path = "C:/Users/kimsv/OneDrive - Mälardalens universitet/Desktop/r01.edf"
    main(file_path)
