import torch
import gradio as gr
import numpy as np
import os
from huggingface_hub import hf_hub_download

# download checkpoint from hf hub on startup
ckpt_path = hf_hub_download(
    repo_id='moisturicer/cvae-face-generation',
    filename='cvae_best.pth',
    repo_type='space'
)

# need to inline the model since hf spaces runs app.py standalone
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 128
N_ATTRS = 6
DEVICE = torch.device('cpu')  # hf free tier is cpu only


class CVAEEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_attrs=N_ATTRS):
        super().__init__()
        self.attr_embed = nn.Linear(n_attrs, 64 * 64)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x, attrs):
        c = self.attr_embed(attrs).view(-1, 1, 64, 64)
        x = torch.cat([x, c], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_log_var(x)


class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_attrs=N_ATTRS):
        super().__init__()
        self.fc = nn.Linear(latent_dim + n_attrs, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh(),
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

    def forward(self, x, attrs):
        mu, log_var = self.encoder(x, attrs)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z, attrs), mu, log_var

    def generate(self, attrs, device, n=1):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(device)
            return self.decoder(z, attrs.to(device))


# load model
model = CVAE().to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()


def denorm(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def generate_face(smiling, young, eyeglasses, male, bald, heavy_makeup):
    attrs = torch.tensor(
        [[smiling, young, eyeglasses, male, bald, heavy_makeup]],
        dtype=torch.float32
    ).to(DEVICE)
    with torch.no_grad():
        img = model.generate(attrs, DEVICE)
    return denorm(img[0]).cpu().permute(1, 2, 0).numpy()


def interpolate_faces(
    smiling_a, young_a, eyeglasses_a, male_a, bald_a, makeup_a,
    smiling_b, young_b, eyeglasses_b, male_b, bald_b, makeup_b,
    steps
):
    attrs_a = torch.tensor([[smiling_a, young_a, eyeglasses_a, male_a, bald_a, makeup_a]],
                            dtype=torch.float32).to(DEVICE)
    attrs_b = torch.tensor([[smiling_b, young_b, eyeglasses_b, male_b, bald_b, makeup_b]],
                            dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        z_a = torch.randn(1, model.latent_dim).to(DEVICE)
        z_b = torch.randn(1, model.latent_dim).to(DEVICE)
        frames = []
        for t in np.linspace(0, 1, int(steps)):
            z = (1 - t) * z_a + t * z_b
            attrs = (1 - t) * attrs_a + t * attrs_b
            img = model.decoder(z, attrs)
            frame = denorm(img[0]).cpu().permute(1, 2, 0).numpy()
            frames.append(frame)

    return np.concatenate(frames, axis=1)


generate_tab = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Slider(0, 1, value=0.5, step=0.1, label="smiling"),
        gr.Slider(0, 1, value=0.5, step=0.1, label="young"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="eyeglasses"),
        gr.Slider(0, 1, value=0.5, step=0.1, label="male"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="bald"),
        gr.Slider(0, 1, value=0.5, step=0.1, label="heavy makeup"),
    ],
    outputs=gr.Image(label="generated face"),
    title="generate face",
    description="adjust sliders to control facial attributes and generate a new face",
    flagging_mode="never"
)

interpolate_tab = gr.Interface(
    fn=interpolate_faces,
    inputs=[
        gr.Slider(0, 1, value=0.8, step=0.1, label="[face a] smiling"),
        gr.Slider(0, 1, value=0.8, step=0.1, label="[face a] young"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="[face a] eyeglasses"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="[face a] male"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="[face a] bald"),
        gr.Slider(0, 1, value=0.8, step=0.1, label="[face a] makeup"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="[face b] smiling"),
        gr.Slider(0, 1, value=0.2, step=0.1, label="[face b] young"),
        gr.Slider(0, 1, value=0.8, step=0.1, label="[face b] eyeglasses"),
        gr.Slider(0, 1, value=0.8, step=0.1, label="[face b] male"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="[face b] bald"),
        gr.Slider(0, 1, value=0.0, step=0.1, label="[face b] makeup"),
        gr.Slider(3, 10, value=6, step=1, label="interpolation steps"),
    ],
    outputs=gr.Image(label="interpolation sequence"),
    title="interpolate faces",
    description="set attributes for two faces and see the smooth transition between them",
    flagging_mode="never"
)

demo = gr.TabbedInterface(
    [generate_tab, interpolate_tab],
    ["generate", "interpolate"]
)

demo.launch()
