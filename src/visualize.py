import torch
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import ATTRIBUTES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _denorm(tensor):
    # reverse the normalization so pixel values are back in [0, 1] for display
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def plot_reconstructions(model, loader, n=8, save_path=None):
    model.eval()
    imgs, attrs = next(iter(loader))
    imgs, attrs = imgs.to(DEVICE), attrs.to(DEVICE)

    with torch.no_grad():
        x_hat, _, _ = model(imgs, attrs)

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    fig.suptitle("cvae — reconstructions", fontsize=12)

    for i in range(n):
        orig = _denorm(imgs[i]).cpu().permute(1, 2, 0).numpy()
        recon = _denorm(x_hat[i]).cpu().permute(1, 2, 0).numpy()

        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("original", fontsize=8)

        axes[1, i].imshow(recon)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("reconstructed", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attribute_generation(model, base_attrs, device, save_path=None):
    # main visualization (for the presentation)
    # for each attribute, generate a face with it off and another with it on side by side so the difference is immediately obvious
    model.eval()
    n_attrs = len(ATTRIBUTES)
    fig, axes = plt.subplots(2, n_attrs, figsize=(n_attrs * 2.5, 5))
    fig.suptitle("attribute control — off vs on", fontsize=12)

    with torch.no_grad():
        for i, attr_name in enumerate(ATTRIBUTES):
            attrs_off = base_attrs.clone()
            attrs_off[0, i] = 0.0
            img_off = model.generate(attrs_off, device)
            img_off = _denorm(img_off[0]).cpu().permute(1, 2, 0).numpy()

            attrs_on = base_attrs.clone()
            attrs_on[0, i] = 1.0
            img_on = model.generate(attrs_on, device)
            img_on = _denorm(img_on[0]).cpu().permute(1, 2, 0).numpy()

            axes[0, i].imshow(img_off)
            axes[0, i].set_title(f"{attr_name}\noff", fontsize=7)
            axes[0, i].axis("off")

            axes[1, i].imshow(img_on)
            axes[1, i].set_title(f"{attr_name}\non", fontsize=7)
            axes[1, i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_interpolation(model, loader, device, steps=8, save_path=None):
    # interpolating between two faces shows that the latent space is smooth
    model.eval()
    imgs, attrs = next(iter(loader))
    imgs, attrs = imgs[:2].to(device), attrs[:2].to(device)

    with torch.no_grad():
        mu1, _ = model.encode(imgs[0:1], attrs[0:1])
        mu2, _ = model.encode(imgs[1:2], attrs[1:2])

        fig, axes = plt.subplots(1, steps + 2, figsize=((steps + 2) * 2, 3))
        fig.suptitle("latent space interpolation", fontsize=11)

        axes[0].imshow(_denorm(imgs[0]).cpu().permute(1, 2, 0).numpy())
        axes[0].set_title("face a", fontsize=8)
        axes[0].axis("off")

        for i, t in enumerate(np.linspace(0, 1, steps)):
            z = (1 - t) * mu1 + t * mu2
            # interpolating attributes alongside latent code so the condition smoothly transitions too
            attr_interp = (1 - t) * attrs[0:1] + t * attrs[1:2]
            img = model.decoder(z, attr_interp)
            img = _denorm(img[0]).cpu().permute(1, 2, 0).numpy()
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"t={t:.1f}", fontsize=7)
            axes[i + 1].axis("off")

        axes[-1].imshow(_denorm(imgs[1]).cpu().permute(1, 2, 0).numpy())
        axes[-1].set_title("face b", fontsize=8)
        axes[-1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_loss_curves(history, save_path=None):
    # tracking all three separately because kl collapsing is the main thing to watch for during training
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(history["loss"], color="#ff6b6b", linewidth=2)
    axes[0].set_title("total loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.2)

    axes[1].plot(history["recon"], color="#a8ff78", linewidth=2)
    axes[1].set_title("reconstruction loss")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.2)

    axes[2].plot(history["kl"], color="#ffd700", linewidth=2)
    axes[2].set_title("kl divergence")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()