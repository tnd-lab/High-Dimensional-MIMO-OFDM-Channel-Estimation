import numpy as np
import torch

from src.ml.explaination import GradCam
from src.channels.channel_est.ml_channel import VAE, Critic, weights_init
from src.ml.dataloader import ChannelDataloader
from src.ml.loss import SkewCalculator, gradient_penalty_loss, l2_loss, vae_loss
from src.ml.transform import MinMaxScaler4D
from src.ml.utils import (
    add_complex_awgn,
    compute_mean_std,
    nmse_func,
    plot_generated_image,
    replicate_to_shape,
    reshape_data,
    save_checkpoint,
    plot_distribution_vae,
    plot_generator_losses,
    plot_critic_losses,
    plot_nmses,
)
from src.settings.config import (
    ebno_db,
    num_bs_ant,
    num_effective_subcarriers,
    num_ofdm_symbols,
    num_rx,
    num_tx,
    num_ut_ant,
    speed,
)
from src.settings.ml import (
    batch_size,
    betas,
    c_lambda,
    critic_repeats,
    device,
    epochs,
    lr,
    n_splits,
    num_workers,
    number_of_samples,
)


def train_process(generator, critic, train_loader, g_opt, c_opt):
    """Training process for one epoch"""
    generator.train()
    critic.train()

    g_epoch_loss = 0.0
    c_epoch_loss = 0.0
    nmse_epoch_score = 0.0

    for batch_idx, (h_freqs, pilot_matrices) in enumerate(train_loader):
        h_freqs = h_freqs.to(device)
        pilot_matrices = pilot_matrices.to(device)

        # Update critic
        total_critic_loss = 0.0
        for _ in range(critic_repeats):
            c_opt.zero_grad()
            h_freqs_hat, latent_vector, *values = generator(pilot_matrices)
            critic_real = critic(h_freqs)
            critic_fake = critic(h_freqs_hat)

            c_w_loss = torch.mean(critic_fake) - torch.mean(critic_real)
            gp_loss = gradient_penalty_loss(
                critic=critic, real=h_freqs, fake=h_freqs_hat
            )
            total_c_loss = c_w_loss + c_lambda * gp_loss

            total_c_loss.backward(retain_graph=True)
            c_opt.step()
            total_critic_loss += total_c_loss.detach().cpu().item()

        c_epoch_loss += total_critic_loss / critic_repeats

        # Update generator
        g_opt.zero_grad()
        h_freqs_hat_2, latent_vector, *values = generator(pilot_matrices)
        v_loss, *_ = vae_loss(h_freqs_hat_2, h_freqs, latent_vector, std_h, mu_h)
        critic_fake_2 = critic(h_freqs_hat_2)

        skew_loss = l2_loss(
            skew_cal(h_freqs.view(h_freqs.size(0), -1)),
            skew_cal(h_freqs_hat_2.view(h_freqs_hat_2.size(0), -1)),
        )

        g_loss = -torch.mean(critic_fake_2) + v_loss + skew_loss
        nmse_score = nmse_func(
            pred=h_freqs_hat_2.detach().cpu(), target=h_freqs.detach().cpu()
        )

        g_loss.backward()
        g_opt.step()

        g_epoch_loss += g_loss.detach().cpu().item()
        nmse_epoch_score += nmse_score

    avg_g_loss = g_epoch_loss / len(train_loader)
    avg_c_loss = c_epoch_loss / len(train_loader)
    avg_nmse = nmse_epoch_score / len(train_loader)

    return avg_g_loss, avg_c_loss, avg_nmse


def eval_process(generator, critic, val_loader):
    """Evaluation process"""
    generator.eval()
    critic.eval()

    val_g_loss = 0.0
    val_c_loss = 0.0
    val_nmse_score = 0.0

    # Store first sample data for potential plotting
    first_sample_data = None

    with torch.no_grad():
        for batch_idx, (h_freqs, pilot_matrices) in enumerate(val_loader):
            h_freqs = h_freqs.to(device)
            pilot_matrices = pilot_matrices.to(device)

            h_freqs_hat, latent_vector, output_encoder = generator(pilot_matrices)
            critic_real = critic(h_freqs)
            critic_fake = critic(h_freqs_hat)

            c_w_loss = torch.mean(critic_fake) - torch.mean(critic_real)
            v_loss, *_ = vae_loss(h_freqs_hat, h_freqs, latent_vector, std_h, mu_h)
            skew_loss = l2_loss(
                skew_cal(h_freqs.view(h_freqs.size(0), -1)),
                skew_cal(h_freqs_hat.view(h_freqs_hat.size(0), -1)),
            )
            g_loss = -torch.mean(critic_fake) + v_loss + skew_loss
            nmse_score = nmse_func(
                pred=h_freqs_hat.detach().cpu(), target=h_freqs.detach().cpu()
            )

            val_g_loss += g_loss.item()
            val_c_loss += c_w_loss.item()
            val_nmse_score += nmse_score

            # Store data from first sample of first batch
            if batch_idx == 0:
                first_sample_data = {
                    "pilot_matrix": pilot_matrices[0, 0].cpu(),
                    "h_freq": h_freqs[0, 0].cpu(),
                    "h_freq_hat": h_freqs_hat[0, 0].cpu(),
                    "latent_vector": latent_vector[0].view(*h_freqs.shape[1:])[0].cpu(),
                    "encoder_output": output_encoder[0]
                    .view(*h_freqs.shape[1:])[0]
                    .cpu(),
                    "h_freqs_hat_sample": h_freqs_hat[0].unsqueeze(0),
                }

    num_batches = len(val_loader)
    metrics = {
        "generator_loss": val_g_loss / num_batches,
        "critic_loss": val_c_loss / num_batches,
        "nmse": val_nmse_score / num_batches,
        "first_sample_data": first_sample_data,
    }
    return metrics


def train_and_eval(
    generator, critic, train_loader, val_loader, g_opt, c_opt, save_dir, fold
):
    best_nmse = float("inf")
    train_g_losses = []
    train_c_losses = []
    train_nmses = []
    val_g_losses = []
    val_c_losses = []
    val_nmses = []

    for epoch in range(epochs):
        # Training
        train_g_loss, train_c_loss, train_nmse = train_process(
            generator, critic, train_loader, g_opt, c_opt
        )
        train_g_losses.append(train_g_loss)
        train_c_losses.append(train_c_loss)
        train_nmses.append(train_nmse)

        # Evaluation
        val_metrics = eval_process(generator, critic, val_loader)
        val_g_losses.append(val_metrics["generator_loss"])
        val_c_losses.append(val_metrics["critic_loss"])
        val_nmses.append(val_metrics["nmse"])

        # Print progress
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(
            f"Train - G Loss: {train_g_loss:.4f}, C Loss: {train_c_loss:.4f}, NMSE: {train_nmse:.4f}"
        )
        print(
            f"Val   - G Loss: {val_metrics['generator_loss']:.4f}, "
            f"C Loss: {val_metrics['critic_loss']:.4f}, NMSE: {val_metrics['nmse']:.4f}"
        )

        # Check if this is the best model
        if val_metrics["nmse"] < best_nmse:
            best_nmse = val_metrics["nmse"]
            # save_path = f"results/checkpoints/{save_dir}/vae_fold_{fold+1}_best.pth"
            save_checkpoint(generator, f"results/checkpoints/{save_dir}/vae.pth")
            save_checkpoint(critic, f"results/checkpoints/{save_dir}/critic.pth")

            # Plot generated image
            plot_generated_image(
                h_freq=val_metrics["first_sample_data"]["h_freq"].numpy().T,
                h_freq_hat=val_metrics["first_sample_data"]["h_freq_hat"].numpy().T,
                pilot_matrix=val_metrics["first_sample_data"]["pilot_matrix"].numpy().T,
                save_path=f"results/images/{save_dir}/fold_{fold + 1}_best.png",
            )

            # Plot distribution for best model
            plot_distribution_vae(
                pilot_matrix=val_metrics["first_sample_data"]["pilot_matrix"],
                estimated_channel=val_metrics["first_sample_data"]["h_freq_hat"],
                real_channel=val_metrics["first_sample_data"]["h_freq"],
                latent_vector=val_metrics["first_sample_data"]["latent_vector"],
                encoder_output=val_metrics["first_sample_data"]["encoder_output"],
                save_path=f"results/distribution/{save_dir}/fold_{fold + 1}_best_distribution.png",
            )

            # Visualize Grad-CAM for best model
            gradcam_conv_1.visualize_cam(
                val_metrics["first_sample_data"]["h_freqs_hat_sample"],
                save_path=f"results/explaination/{save_dir}/fold_{fold + 1}_best_gradcam_conv_1.png",
            )
            gradcam_conv_2.visualize_cam(
                val_metrics["first_sample_data"]["h_freqs_hat_sample"],
                save_path=f"results/explaination/{save_dir}/fold_{fold + 1}_best_gradcam_conv_2.png",
            )

            print(f"New best model saved with NMSE: {best_nmse:.4f}")

        print("-" * 80)

    # Save losses and NMSE for the fold
    plot_generator_losses(
        train_g_losses,
        val_g_losses,
        f"results/losses/{save_dir}/fold_{fold + 1}_generator_losses.png",
    )
    plot_critic_losses(
        train_c_losses,
        val_c_losses,
        f"results/losses/{save_dir}/fold_{fold + 1}_critic_losses.png",
    )
    plot_nmses(
        train_nmses, val_nmses, f"results/nmses/{save_dir}/fold_{fold + 1}_nmses.png"
    )

    return best_nmse


if __name__ == "__main__":
    save_dir = f"txant_{num_ut_ant}_rxant_{num_bs_ant}_speed_{speed}_samples_{number_of_samples}_ebno_{ebno_db}"
    pilot_matrices = np.load(f"data/{save_dir}/pilot_matrices.npy")
    h_freqs = np.load(f"data/{save_dir}/h_freqs.npy")

    pilot_matrices = replicate_to_shape(pilot_matrices, h_freqs.shape)

    # pilot_matrices = add_complex_awgn(pilot_matrices)

    pilot_matrices = reshape_data(pilot_matrices)
    h_freqs = reshape_data(h_freqs)

    scaler = MinMaxScaler4D()
    pilot_matrices = scaler.fit_transform(pilot_matrices)
    h_freqs = scaler.fit_transform(h_freqs)

    dataloader = ChannelDataloader(
        pilot_matrices=pilot_matrices,
        h_freqs=h_freqs,
        batch_size=batch_size,
        num_worker=num_workers,
        n_splits=n_splits,
    )
    fold_loaders = dataloader.get_fold_dataloaders()

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        mu_h, std_h = compute_mean_std(train_loader.dataset.dataset.h_freqs)
        mu_h = mu_h.to(device)
        std_h = std_h.to(device)
        vae = VAE(
            h_freqs.shape[1:],
            mu_h=mu_h,
            std_h=std_h,
        ).to(device)
        critic = Critic(h_freqs.shape[1:]).to(device)

        vae = vae.apply(weights_init)
        critic = critic.apply(weights_init)
        skew_cal = SkewCalculator(device)

        gradcam_conv_1 = GradCam(critic, critic.critic[-4])
        gradcam_conv_2 = GradCam(critic, critic.critic[-3])

        vae_opt = torch.optim.Adam(vae.parameters(), lr=lr, betas=betas)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

        print(f"\nFold {fold + 1}/{n_splits}")
        print("=" * 80)
        best_nmse = train_and_eval(
            vae, critic, train_loader, val_loader, vae_opt, critic_opt, save_dir, fold
        )
        break
