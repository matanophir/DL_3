import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        #similar to DC_gan
        modules.append(nn.Conv2d(in_channels, 128, 4, 2,1))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.BatchNorm2d(128))
        
        modules.append(nn.Conv2d(128, 256, 4, 2, 1))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.BatchNorm2d(256))

        modules.append(nn.Conv2d(256, 512, 4, 2, 1))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.BatchNorm2d(512))

        modules.append(nn.Conv2d(512, out_channels, 8, 2, 1))

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)
        # for layer in self.cnn:
        #     x = layer(x)
        #     print(x.shape)
        # return x


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======

        modules.append(nn.ConvTranspose2d(in_channels, 512, 8, 2, 1,))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.LeakyReLU(0.2))

        modules.append(nn.ConvTranspose2d(512, 256, 4, 2, 1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.LeakyReLU(0.2))

        modules.append(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.LeakyReLU(0.2))

        modules.append(nn.ConvTranspose2d(128, out_channels, 4, 2, 1))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))
        # for layer in self.cnn:
        #     h = layer(h)
        #     print(h.shape)
        # return torch.tanh(h)


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.features_to_mu = nn.Linear(n_features, z_dim)
        self.features_to_logvar = nn.Linear(n_features, z_dim)

        self.z_to_features = nn.Linear(z_dim, n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        # get mu and logvar
        h = self.features_encoder(x).view(x.shape[0], -1)
        mu = self.features_to_mu(h)
        log_sigma2 = self.features_to_logvar(h)

        # sampling
        std = torch.exp(0.5 * log_sigma2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h = self.z_to_features(z).view(-1, *self.features_shape) # reshape back to the shape of the features
        x_rec = self.features_decoder(h)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======

            z = torch.randn(n, self.z_dim, device=device)

            samples = self.decode(z)
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    d_z = z_mu.shape[1]
    d_x = torch.numel(x) // x.shape[0]
    z_sigma = torch.exp(z_log_sigma2)
    N = x.shape[0]

    #point-wise calculation
    kldiv_loss = torch.sum(z_sigma, dim= 1) + torch.sum(z_mu**2, dim= 1) - d_z - torch.sum(z_log_sigma2, dim= 1)
    data_loss =  torch.sum((x.view(N,-1) - xr.view(N,-1))**2, dim= 1) / (x_sigma2 * d_x)

    #loss
    kldiv_loss = torch.mean(kldiv_loss)
    data_loss = torch.mean(data_loss)

    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
