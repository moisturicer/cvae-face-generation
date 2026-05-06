import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_reconstruction(model, loader, n_batches=10):
    # checking reconstruction quality on the test set, limit to n_batches so it doesn't take forever
    model.eval()
    total_mse = 0.0
    count = 0

    with torch.no_grad():
        for i, (imgs, attrs) in enumerate(loader):
            if i >= n_batches:
                break
            imgs, attrs = imgs.to(DEVICE), attrs.to(DEVICE)
            x_hat, _, _ = model(imgs, attrs)
            mse = torch.nn.functional.mse_loss(x_hat, imgs).item()
            total_mse += mse
            count += 1

    avg_mse = total_mse / count
    print(f"avg reconstruction mse: {avg_mse:.6f}")
    return avg_mse


def evaluate_attribute_accuracy(model, loader, n_batches=20):
    # checks whether the generated faces actually reflect
    model.eval()
    all_diffs = {attr: [] for attr in range(6)}

    with torch.no_grad():
        for i, (imgs, attrs) in enumerate(loader):
            if i >= n_batches:
                break
            imgs, attrs = imgs.to(DEVICE), attrs.to(DEVICE)

            for attr_idx in range(6):
                # generate with attribute on vs off
                attrs_on = attrs.clone()
                attrs_on[:, attr_idx] = 1.0
                attrs_off = attrs.clone()
                attrs_off[:, attr_idx] = 0.0

                out_on = model.generate(attrs_on, DEVICE, n=attrs.size(0))
                out_off = model.generate(attrs_off, DEVICE, n=attrs.size(0))

                # average pixel difference when toggling this attribute
                diff = (out_on - out_off).abs().mean().item()
                all_diffs[attr_idx].append(diff)

    from src.dataset import ATTRIBUTES
    print("average image change when toggling each attribute:")
    for idx, attr in enumerate(ATTRIBUTES):
        avg_diff = np.mean(all_diffs[idx])
        print(f"  {attr:20s}: {avg_diff:.4f}")