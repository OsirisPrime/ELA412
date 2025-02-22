import numpy as np
import matplotlib.pyplot as plt

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

def main():
    pca_results = load_results("pca_results.csv")
    ica_results = load_results("ica_results.csv")

    plot_components(pca_results, 'PCA')
    plot_components(ica_results, 'ICA')

if __name__ == "__main__":
    main()
