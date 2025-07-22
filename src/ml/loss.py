import numpy as np
import torch
from scipy import stats


def l2_loss(fake, real):
    return torch.norm(fake - real, p=2)


# Assuming the VAE loss function is defined as follows:
def vae_loss(predicted_h_freqs, true_h_freqs, latent_vector, std_h, mu_h):
    # L2 loss between predicted and true h
    recon_loss = l2_loss(predicted_h_freqs, true_h_freqs)

    # KL divergence formula between two Gaussians
    var_l = latent_vector.var(axis=0)
    mu_l = latent_vector.mean(axis=0)
    var_h = std_h**2
    mu_h = mu_h
    kl_div = 0.5 * torch.sum(
        torch.log(var_h / var_l) + (var_l + (mu_l - mu_h).pow(2)) / var_h - 1
    )
    return recon_loss + 0.01 * kl_div, recon_loss, kl_div


# before compute the gradient penalty, we need compute the gradient of interpolated image
def get_gradient(critic, real, fake, epsilon, *args):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        critic: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = critic(mixed_images, *args)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


# then we compute the gradient penalty
def gradient_penalty_loss(critic, real, fake, *args):
    # get epsilon value
    epsilon = torch.rand(len(real), 1, 1, 1, device=real.device, requires_grad=True)

    # Compute the gradient
    gradient = get_gradient(critic, real, fake, epsilon, *args)

    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


class SkewCalculator:
    def __init__(self, device=None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def __call__(self, tensor):
        """
        Compute skewness of a PyTorch tensor with flexible shape.
        If the tensor is multi-dimensional, computes skewness along the first dimension.

        :param tensor: PyTorch tensor of any shape
        :return: PyTorch tensor containing skewness value(s)
        """
        # Ensure input is a PyTorch tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32, device=self.device)

        # Move tensor to CPU and convert to numpy for scipy
        numpy_array = tensor.detach().cpu().numpy()

        # Compute skewness along the first dimension
        if numpy_array.ndim == 1:
            skewness = stats.skew(numpy_array, axis=0, bias=False)

            if np.isnan(skewness):
                skewness = 0.0
        else:
            skewness = stats.skew(numpy_array, axis=1, bias=False)

        # Convert back to PyTorch tensor and move to original device
        return torch.tensor(skewness, dtype=torch.float32, device=self.device)
