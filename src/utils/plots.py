import matplotlib.pyplot as plt
import numpy as np


def plot_symbols(symbol, save_fig: str = None):
    # Assuming x_signal is a 2D array
    # Display the array as an image
    plt.imshow(symbol, cmap="viridis", aspect="auto")

    # Set labels for the axes with increased fontsize
    plt.xlabel("OFDM Symbols", fontsize=20)
    plt.ylabel("Subcarriers", fontsize=20)

    # Adjust tick parameters to increase the size of tick labels
    plt.tick_params(axis="both", which="major", labelsize=20)

    if save_fig:
        plt.savefig(save_fig)

    # Show the plot
    plt.show()
    plt.close()


def plot_complex_gain(complex_gains, save_fig: str = None):
    num_lines = complex_gains[:, 0].shape[0]
    for i in range(num_lines):
        plt.plot(complex_gains[i, :], label=f"Cluster {i + 1}")

    plt.xlabel("Time step")
    plt.ylabel("CIR")

    # Scale y-axis to make the plot clearer
    # plt.ylim(-0.1, 0.8)  # Adjust the limits based on your data

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Clusters")
    plt.tight_layout()  # Adjust layout to make room for the legend

    if save_fig:
        plt.savefig(save_fig)

    # Show the plot
    plt.show()
    plt.close()


def plot_channel_tap_tim_domain(time_tap, l_min, l_max, save_fig: str = None):
    plt.figure()
    plt.title("Channel taps at the first time step 1st")
    plt.stem(np.arange(l_min, l_max + 1), np.abs(time_tap))
    plt.xlabel("Time Lag Step")
    plt.ylabel("Channel Tap")

    if save_fig:
        plt.savefig(save_fig)

    plt.show()
    plt.close()


def plot_channel_time_domain(h_time, save_fig: str = None):
    num_lines = h_time.shape[1]
    for i in range(num_lines):
        plt.plot(h_time[:, i], label=f"Time lag {i + 1}")

    plt.xlabel("Time step")
    plt.ylabel("Channel Tap")

    # Scale y-axis to make the plot clearer
    # plt.ylim(-0.1, 0.8)  # Adjust the limits based on your data

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Channel Lines")
    plt.tight_layout()  # Adjust layout to make room for the legend

    if save_fig:
        plt.savefig(save_fig)

    plt.show()
    plt.close()


def plot_frequency_of_each_subcarrier(frequencies, save_fig: str = None):
    plt.title("Frequency of each subcarriers")
    plt.stem(np.arange(len(frequencies)), frequencies)
    plt.xlabel("Subcarrier")
    plt.ylabel("Frequency")

    if save_fig:
        plt.savefig(save_fig)

    plt.show()
    plt.close()


def plot_channel_frequency_complex_values(h_freq, save_fig: str = None):
    plt.plot(np.real(h_freq))
    plt.plot(np.imag(h_freq))
    plt.xlabel("Subcarrier index")
    plt.ylabel("Channel frequency response")
    plt.legend(
        [
            "Ideal (real part)",
            "Ideal (imaginary part)",
            "Estimated (real part)",
            "Estimated (imaginary part)",
        ]
    )
    plt.title("Comparison of channel frequency responses")

    if save_fig:
        plt.savefig(save_fig)

    plt.show()
    plt.close()


def plot_channel_frequency_domain(h_freq, save_fig: str = None):
    # Assuming x_signal is a 2D array
    # Display the array as an image
    im = plt.imshow(h_freq, cmap="viridis", aspect="auto")

    plt.colorbar(im)

    # Set labels for the axes with increased fontsize
    plt.xlabel("OFDM Symbols", fontsize=20)
    plt.ylabel("Subcarriers", fontsize=20)

    # Adjust tick parameters to increase the size of tick labels
    plt.tick_params(axis="both", which="major", labelsize=20)

    if save_fig:
        plt.savefig(save_fig)

    # Show the plot
    plt.show()
    plt.close()
