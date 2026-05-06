import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 128
N_ATTRS = 6  # matches the 6 attributes defined in dataset.py


class CVAEEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_attrs=N_ATTRS):
        super().__init__()

        self.attr_embed = nn.Linear(n_attrs, 64 * 64)

        # input is now 4 channels (3 rgb + 1 condition map) this way the encoder sees both the image and what we want at every spatial location
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2, padding=1),    # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x, attrs):
        # reshape the embedded condition into a spatial map and concatenate with the image before encoding
        c = self.attr_embed(attrs).view(-1, 1, 64, 64)
        x = torch.cat([x, c], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_log_var(x)


class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_attrs=N_ATTRS):
        super().__init__()

        # concatenating the condition with the latent code here too so the decoder knows what kind of face it's supposed to generate
        self.fc = nn.Linear(latent_dim + n_attrs, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32 -> 64
            nn.Tanh(),  # tanh keeps output in [-1, 1] matching our normalization
        )

    def forward(self, z, attrs):
        x = torch.cat([z, attrs], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        return self.deconv(x)


class CVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_attrs=N_ATTRS):
        super().__init__()
        self.encoder = CVAEEncoder(latent_dim, n_attrs)
        self.decoder = CVAEDecoder(latent_dim, n_attrs)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        # during training we sample from the distribution at inference we just use the mean (more stable output)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, attrs):
        mu, log_var = self.encoder(x, attrs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, attrs)
        return x_hat, mu, log_var

    def elbo_loss(self, x, x_hat, mu, log_var, beta=1.0):
        # reconstruction loss 
        recon = F.mse_loss(x_hat, x, reduction='mean')

        # kl divergence 
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

        return recon + beta * kl, recon, kl

    def generate(self, attrs, device, n=1):
        # at generation time there's no input image so we sample z from the prior and decode with the given attributes
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(device)
            attrs = attrs.to(device)
            return self.decoder(z, attrs)

    def encode(self, x, attrs):
        return self.encoder(x, attrs)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}")