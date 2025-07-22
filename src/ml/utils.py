import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from src.settings.config import (
    num_bs_ant,
    num_effective_subcarriers,
    num_ofdm_symbols,
    num_rx,
    num_tx,
    num_ut_ant,
)
from src.settings.ml import noise_variance


def add_complex_awgn(input_matrix, noise_variance=noise_variance):
    # Generate noise for real and imaginary parts
    real_part = np.random.normal(
        loc=0, scale=np.sqrt(noise_variance / 2), size=input_matrix.shape
    )
    imag_part = np.random.normal(
        loc=0, scale=np.sqrt(noise_variance / 2), size=input_matrix.shape
    )
    noise = real_part + 1j * imag_part

    # Add noise to the input matrix
    noisy_matrix = input_matrix + noise
    return noisy_matrix


def reshape_data(data):
    # First convert complex to real
    data_real = np.stack((data.real, data.imag), axis=1)

    # Reshape and combine channel dimensions with real/imag
    num_channels = num_tx * num_bs_ant * num_rx * num_ut_ant
    data = data_real.reshape(
        -1, 2, num_ofdm_symbols * num_channels, num_effective_subcarriers
    )

    return data


def reversed_reshape_data(data):
    # Calculate number of channels
    num_channels = num_tx * num_bs_ant * num_rx * num_ut_ant

    # Reshape to separate real/imag dimension
    data = data.reshape(
        -1, 2, num_ofdm_symbols * num_channels, num_effective_subcarriers
    )

    # Split real and imaginary parts and combine to complex
    real_part = data[
        :, 0, ...
    ]  # shape: (batch_size, num_channels, num_ofdm_symbols, num_effective_subcarriers)
    imag_part = data[
        :, 1, ...
    ]  # shape: (batch_size, num_channels, num_ofdm_symbols, num_effective_subcarriers)
    data_complex = real_part + 1j * imag_part

    # Reshape back to original dimensions
    data_complex = data_complex.reshape(
        -1,
        num_tx,
        num_bs_ant,
        num_rx,
        num_ut_ant,
        num_ofdm_symbols,
        num_effective_subcarriers,
    )

    return data_complex


def compute_mean_std(h_freqs):
    # Compute the mean and std across the first dimension (axis=0)
    mu_h = h_freqs.mean(axis=0).reshape(-1)  # Shape: (2*8*32,)
    std_h = h_freqs.std(axis=0).reshape(-1)  # Shape: (2*8*32,)

    mu_h = torch.from_numpy(mu_h).float()
    std_h = torch.from_numpy(std_h).float()

    return mu_h, std_h


def save_checkpoint(model, file_name):
    model_to_save = {
        "state_dict": model.state_dict(),
    }
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    torch.save(model_to_save, file_name)


def save_npy(file_name, np_array):
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    np.save(file_name, np_array)


def nmse_func(pred, target):
    # Compute the Mean Squared Error (MSE)
    mse = torch.mean((pred - target) ** 2)

    # Compute the norm of the target tensor
    norm = torch.mean(target**2)

    # NMSE is the ratio of MSE to the norm of the target
    nmse_value = mse / norm

    return nmse_value.item()


def ssim_func(pred, target, eps=1e-8):
    """
    SSIM for batched complex-valued or real-valued channel matrices.
    Input:
        pred, target: shape (B, C, H, W)
    Output:
        mean SSIM across the batch
    """
    B = pred.size(0)
    pred = pred.view(B, -1)  # flatten (C*H*W)
    target = target.view(B, -1)

    # Compute means
    mu_pred = pred.mean(dim=1)
    mu_target = target.mean(dim=1)

    # Compute variances
    sigma_pred_sq = ((pred - mu_pred.unsqueeze(1)) ** 2).mean(dim=1)
    sigma_target_sq = ((target - mu_target.unsqueeze(1)) ** 2).mean(dim=1)

    # Compute covariance
    covariance = (
        (pred - mu_pred.unsqueeze(1)) * (target - mu_target.unsqueeze(1))
    ).mean(dim=1)

    # Compute c1 and c2
    c1 = (0.01 * target.max(dim=1).values) ** 2 + eps
    c2 = (0.03 * target.max(dim=1).values) ** 2 + eps

    # SSIM formula
    numerator = (2 * mu_pred * mu_target + c1) * (2 * covariance + c2)
    denominator = (mu_pred**2 + mu_target**2 + c1) * (
        sigma_pred_sq + sigma_target_sq + c2
    )
    ssim = numerator / (denominator + eps)

    return ssim.mean().item()


def plot_generated_image(pilot_matrix, h_freq, h_freq_hat, save_path):
    display_list = [
        pilot_matrix,
        h_freq,
        h_freq_hat,
    ]
    title = ["Input Y", "Target H", "Prediction H"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def replicate_to_shape(matrix: np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Replicates a numpy matrix to match the given new_shape.

    Parameters:
    matrix (np.ndarray): The input matrix to be expanded.
    new_shape (tuple): The desired target shape.

    Returns:
    np.ndarray: The expanded matrix with the new shape.
    """
    # Ensure matrix is a numpy array
    matrix = np.array(matrix)

    # Validate shape compatibility
    if len(matrix.shape) != len(new_shape):
        raise ValueError("Shapes must have the same number of dimensions.")

    # Compute the tiling factor
    tile_factors = np.array(new_shape) // np.array(matrix.shape)

    # Ensure new shape is a multiple of the original shape
    if not np.all(np.array(new_shape) % np.array(matrix.shape) == 0):
        raise ValueError(
            "New shape must be a multiple of the original shape along expandable dimensions."
        )

    # Replicate using np.tile
    expanded_matrix = np.tile(matrix, tile_factors)

    return expanded_matrix


def plot_generated_image(pilot_matrix, h_freq, h_freq_hat, save_path):
    display_list = [
        pilot_matrix,
        h_freq,
        h_freq_hat,
    ]
    title = ["Input Y", "Target H", "Prediction H"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_generator_losses(train_g_losses, val_g_losses, save_path):
    plt.figure()
    plt.plot(train_g_losses, label="Train Generator Loss")
    plt.plot(val_g_losses, label="Val Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator Loss")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # Save data as .npy files
    np.save(
        os.path.join(save_dir, "train_generator_losses.npy"), np.array(train_g_losses)
    )
    np.save(os.path.join(save_dir, "val_generator_losses.npy"), np.array(val_g_losses))


def plot_critic_losses(train_c_losses, val_c_losses, save_path):
    plt.figure()
    plt.plot(train_c_losses, label="Train Critic Loss")
    plt.plot(val_c_losses, label="Val Critic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Critic Loss")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # Save data as .npy files
    np.save(os.path.join(save_dir, "train_critic_losses.npy"), np.array(train_c_losses))
    np.save(os.path.join(save_dir, "val_critic_losses.npy"), np.array(val_c_losses))


def plot_nmses(train_nmses, val_nmses, save_path):
    plt.figure()
    plt.plot(train_nmses, label="Train NMSE")
    plt.plot(val_nmses, label="Val NMSE")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.legend()
    plt.title("NMSE")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # Save data as .npy files
    np.save(os.path.join(save_dir, "train_nmses.npy"), np.array(train_nmses))
    np.save(os.path.join(save_dir, "val_nmses.npy"), np.array(val_nmses))


def plot_skewness_func(skewness_scores, epochs, img_path):
    # # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, skewness_scores, label="Target")
    plt.xlabel("Epoch")
    plt.ylabel("Skewness distances")
    plt.title("Skewness Loss")
    plt.legend()
    if not os.path.exists(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    # plt.show()


def replicate_to_shape(matrix: np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Replicates a numpy matrix to match the given new_shape.

    Parameters:
    matrix (np.ndarray): The input matrix to be expanded.
    new_shape (tuple): The desired target shape.

    Returns:
    np.ndarray: The expanded matrix with the new shape.
    """
    # Ensure matrix is a numpy array
    matrix = np.array(matrix)

    # Validate shape compatibility
    if len(matrix.shape) != len(new_shape):
        raise ValueError("Shapes must have the same number of dimensions.")

    # Compute the tiling factor
    tile_factors = np.array(new_shape) // np.array(matrix.shape)

    # Ensure new shape is a multiple of the original shape
    if not np.all(np.array(new_shape) % np.array(matrix.shape) == 0):
        raise ValueError(
            "New shape must be a multiple of the original shape along expandable dimensions."
        )

    # Replicate using np.tile
    expanded_matrix = np.tile(matrix, tile_factors)

    return expanded_matrix


def plot_distribution_vae(
    pilot_matrix,
    estimated_channel,
    latent_vector,
    encoder_output,
    real_channel,
    save_path=None,
):
    # Convert to numpy arrays
    encoder_input_np = pilot_matrix.flatten()
    encoder_output_np = encoder_output.flatten()
    decoder_input_np = latent_vector.flatten()
    decoder_output_np = estimated_channel.flatten()
    true_output_np = real_channel.flatten()

    # Create a figure with five subplots in a single row
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor("white")  # Set figure background to white

    # Define the data and labels for each subplot
    data = [
        encoder_input_np,
        encoder_output_np,
        decoder_input_np,
        decoder_output_np,
        true_output_np,
    ]
    titles = [
        "Encoder Input",
        "Encoder Output",
        "Decoder Input (Latent Space)",
        "Decoder Output",
        "True Output",
    ]
    shapes = [
        pilot_matrix.shape,
        encoder_output.shape,
        latent_vector.shape,
        estimated_channel.shape,
        real_channel.shape,
    ]
    colors = ["cyan", "magenta", "orange", "green", "red"]
    axes = [ax1, ax2, ax3, ax4, ax5]

    for ax, d, title, shape, color in zip(axes, data, titles, shapes, colors):
        kde = gaussian_kde(d)
        x_range = np.linspace(d.min(), d.max(), 200)

        ax.plot(x_range, kde(x_range), color=color)
        ax.fill_between(x_range, kde(x_range), alpha=0.5, color=color)

        # Set axis background and text colors
        ax.set_facecolor("white")
        ax.set_title(f"{title}", color="black", fontsize=16)
        ax.set_xlabel("Value", color="black", fontsize=12)
        ax.set_ylabel("Density", color="black", fontsize=12)

        ax.grid(color="gray", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.tick_params(colors="black")

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor="white")  # Ensure saving with white background
        plt.close()
    else:
        plt.show()
