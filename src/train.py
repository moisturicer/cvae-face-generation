import os
import torch
import torch.optim as optim

CHECKPOINT_DIR = "/content/drive/MyDrive/3RDYEAR/CS346/cvae-face-generation/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cvae(model, train_loader, epochs=30, lr=1e-3, beta=1.0):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "recon": [], "kl": []}
    best_loss = float("inf")

    print(f"training on: {DEVICE}")
    print(f"epochs: {epochs} | lr: {lr} | beta: {beta}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_recon = total_kl = 0.0

        for imgs, attrs in train_loader:
            imgs = imgs.to(DEVICE)
            attrs = attrs.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mu, log_var = model(imgs, attrs)
            loss, recon, kl = model.elbo_loss(imgs, x_hat, mu, log_var, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        n = len(train_loader)
        avg_loss = total_loss / n
        avg_recon = total_recon / n
        avg_kl = total_kl / n

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        # save the best checkpoint so even if training crashes, progress is not lost
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(model, "cvae_best.pth")

        # logging every 5 epochs is enough to track progress without flooding the output
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:03d}/{epochs} | loss: {avg_loss:.4f} | recon: {avg_recon:.4f} | kl: {avg_kl:.4f}")

        if avg_kl < 0.01 and epoch > 5:
            print(f"  warning: kl is very low ({avg_kl:.4f}) — consider increasing beta")

    print(f"\ntraining complete. best loss: {best_loss:.6f}")
    return history


def _save_checkpoint(model, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, filename))


def load_checkpoint(model, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint found at {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"loaded: {filename}")
    return model