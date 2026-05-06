import torch
import gradio as gr
import numpy as np
from src.cvae import CVAE
from src.train import CHECKPOINT_DIR, DEVICE
import os

model = CVAE().to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(CHECKPOINT_DIR, "cvae_best.pth"),
    map_location=DEVICE
))
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

    combined = np.concatenate(frames, axis=1)
    return combined

# generation
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

# interpolation
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

demo.launch(share=True)