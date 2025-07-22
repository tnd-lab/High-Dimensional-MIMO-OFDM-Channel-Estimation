from itertools import chain
from math import prod

import torch
import torch.nn as nn
import torchsummary


class VAE(nn.Module):
    def __init__(
        self,
        input_shape: tuple = (2, 16, 64),
        mu_h=None,
        std_h=None,
    ):
        super(VAE, self).__init__()

        self.mu_h = mu_h  # Pre-defined tensor or None
        self.std_h = std_h  # Pre-defined tensor or None
        self.channels, self.height, self.width = input_shape  # (n, 14*k, 12*m)

        self._dims = [64, 128, 256, 512]

        self.stride_sizes, self.k, self.m = self._calculate_strides(
            self.height, self.width
        )

        # Encoder
        self.encoder = self._build_encoder()

        self.last_encoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.channels * prod(chain(*self.stride_sizes)) * self.k * self.m,
                prod(input_shape),
            ),
        )

        self.initial_decoder_layer = nn.Linear(
            prod(input_shape),
            self.channels * prod(chain(*self.stride_sizes)) * self.k * self.m,
        )

        # Decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Builds the encoder with calculated stride steps."""
        encoder_layers = []
        in_channels = self.channels

        # Initial block with stride=1
        # encoder_layers.append(self._make_encoder_block(in_channels, in_channels))

        for dim, stride_size in zip(self._dims, self.stride_sizes):
            out_channels = dim

            encoder_layers.append(
                self._make_encoder_block(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = out_channels

        encoder_layers.append(
            self._make_encoder_block(in_channels, in_channels, 3, 1, 1)
        )

        return nn.Sequential(*encoder_layers)

    def _build_decoder(self):
        """Builds the decoder as the inverse of the encoder."""
        decoder_layers = []

        in_channels = int(
            self.channels * self.height * self.width / (self.k * self.m)
        )  # Starting from encoded representation

        for dim, stride_size in zip(self._dims[::-1], self.stride_sizes[::-1]):
            out_channels = dim

            decoder_layers.append(
                self._make_decoder_block(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = out_channels

        decoder_layers.append(
            self._make_decoder_block(in_channels, self.channels, 3, 1, 1, True)
        )

        return nn.Sequential(*decoder_layers)

    def _calculate_strides(self, k, m):
        stride_sizes = []
        while k % 2 == 0 and m % 2 == 0:
            k //= 2
            m //= 2
            stride_sizes.append((2, 2))
            if (k % 2 != 0 or m % 2 != 0) or len(stride_sizes) == len(self._dims):
                break

        return stride_sizes, k, m

    def _make_encoder_block(
        self, input_channels, output_channels, kernel_size=3, stride=1, padding=1
    ):
        """Creates a single encoder block."""
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _make_decoder_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        final_layer=False,
    ):
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride,
                    padding,
                ),
                nn.Tanh(),
            )
        else:
            """Creates a single decoder block."""
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride,
                    padding,
                ),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def encode(self, x):
        """Encodes input to the latent representation."""
        output_encoder = self.encoder(x)  # (batch, n*k*m, 7, 6)

        output_encoder = output_encoder.view(
            output_encoder.size(0), -1
        )  # (batch, n*k*m*14*12)
        return output_encoder

    def reparameterize(self, output_encoder):
        """Reparameterizes using pre-defined mu_h and std_h if provided."""
        if isinstance(self.mu_h, torch.Tensor) and isinstance(self.std_h, torch.Tensor):
            # output_encoder = self.last_encoder_layer(output_encoder)
            mu_h = self.mu_h.to(output_encoder.device)
            std_h = self.std_h.to(output_encoder.device)
            z_latent = mu_h + output_encoder * std_h  # Element-wise operation
            # z_latent = self.initial_decoder_layer(z_latent)
        else:
            z_latent = output_encoder  # No transformation if mu_h or std_h is None

        return z_latent

    def decode(self, z):
        """Decodes latent vector z to reconstruct input."""
        z = z.view(z.size(0), -1, self.k, self.m)
        decoder_output = self.decoder(z)

        return decoder_output

    def forward(self, x):
        """Forward pass: encode, reparameterize, decode."""
        output_encoder = self.encode(x)
        z = self.reparameterize(output_encoder)
        recon_x = self.decode(z)
        return recon_x, z, output_encoder


class Critic(nn.Module):
    def __init__(self, input_shape=(2, 7, 6)):
        super(Critic, self).__init__()
        self.channels, self.height, self.width = input_shape  # e.g., (2, 14*k, 12*m)
        self.stride_sizes, self.k, self.m = self._calculate_strides(
            self.height, self.width
        )
        self._dims = [32, 64, 128, 256]
        if len(self.stride_sizes) > len(self._dims):
            self._dims.extend(
                [max(self._dims)] * (len(self.stride_sizes) - len(self._dims))
            )
        else:
            self._dims = self._dims[: len(self.stride_sizes)]

        # Build the critic network
        self.critic = self._build_critic()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final feature size depends on the last conv layer's output channels
        self.fc = nn.Linear(
            self._dims[-1],
            1,
        )  # Adjust based on hidden_dim scaling

    def _calculate_strides(self, k, m):
        stride_sizes = []
        while k % 2 == 0 and m % 2 == 0:
            k //= 2
            m //= 2
            stride_sizes.append((2, 2))
            if k % 2 != 0 or m % 2 != 0:
                break

        return stride_sizes, k, m

    def _make_critic_block(
        self, input_channels, output_channels, kernel_size=3, stride=1, padding=1
    ):
        """Creates a single critic block."""
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _build_critic(self):
        """Builds the encoder with calculated stride steps."""
        critic_layers = []
        in_channels = self.channels

        # Initial block with stride=1
        # critic_layers.append(self._make_critic_block(in_channels, in_channels))

        for dim, stride_size in zip(self._dims, self.stride_sizes[::-1]):
            k, m = stride_size
            out_channels = dim

            critic_layers.append(
                self._make_critic_block(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=max(stride_size),
                    padding=1,
                )
            )
            in_channels = out_channels

        return nn.Sequential(*critic_layers)

    def forward(self, image):
        """Forward pass through the critic."""
        critic_pred = self.critic(image)  # (batch, channels, h', w')
        critic_pred = self.avg_pool(critic_pred)  # (batch, channels, 1, 1)
        critic_pred = critic_pred.view(critic_pred.size(0), -1)  # (batch, channels)
        critic_pred = self.fc(critic_pred)  # (batch, 1)
        return critic_pred


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Test the model
    input_shape = (1, 2, 64, 64)
    # Test with mu_h and std_h
    x = torch.randn(*input_shape).cuda()
    b, n, h, w = input_shape
    mu_h = torch.zeros(1, n * h * w).cuda()  # Example mu_h
    std_h = torch.ones(1, n * h * w).cuda()  # Example std_h

    model = VAE(input_shape[1:], mu_h=mu_h, std_h=std_h).to("cuda").eval()
    torchsummary.summary(model, input_shape[1:])

    critic = Critic(input_shape[1:]).to("cuda").eval()
    torchsummary.summary(critic, input_shape[1:])

    from src.ml.explaination import GradCam

    gradcam = GradCam(critic, critic.critic[-4])

    gradcam.visualize_cam(x, save_path="images/test_gradcam.png")
