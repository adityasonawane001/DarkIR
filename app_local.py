import os
import time
import numpy as np
from PIL import Image

import gradio as gr
import torch

from archs import create_model
from options.options import parse


CONFIG_PATH = "./options/inference/LOLBlur.yml"


def _clean_state_dict_keys(state_dict):
    """Handle checkpoints saved with or without DDP 'module.' prefixes."""
    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_weights(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "params" in checkpoint:
            weights = checkpoint["params"]
        elif "model_state_dict" in checkpoint:
            weights = checkpoint["model_state_dict"]
        else:
            weights = checkpoint
    else:
        weights = checkpoint

    # Try direct load first, then fallback by cleaning DDP prefix.
    try:
        model.load_state_dict(weights, strict=False)
    except RuntimeError:
        model.load_state_dict(_clean_state_dict_keys(weights), strict=False)


def _prepare_input(image_pil, max_side=1280):
    image_pil = image_pil.convert("RGB")
    w, h = image_pil.size
    longest = max(h, w)

    # Keep latency manageable for laptop GPUs by downscaling very large images.
    if longest > max_side:
        scale = max_side / float(longest)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image_pil = image_pil.resize((new_w, new_h), Image.BICUBIC)

    arr = np.array(image_pil).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x, image_pil.size


def build_model_and_device():
    opt = parse(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = create_model(opt["network"], rank=0)
    model = model.to(device)

    ckpt_path = opt["save"]["path"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    _load_weights(model, ckpt_path, device)
    model.eval()

    return model, device


MODEL, DEVICE = build_model_and_device()


def _auto_brightness_boost(img_np, boost):
    if boost <= 0:
        return img_np

    mean_luma = float(img_np.mean())
    if mean_luma >= 0.25:
        return img_np

    # Gamma-based lift for extremely dark outputs; keeps highlights in range.
    gamma = max(0.35, 1.0 - 0.55 * boost)
    boosted = np.power(np.clip(img_np, 0.0, 1.0), gamma)
    return np.clip(boosted, 0.0, 1.0)


def process_img(image, boost):
    if image is None:
        return None, "No input image provided."

    start = time.perf_counter()

    try:
        x, resized_size = _prepare_input(image)
        x = x.to(DEVICE)

        with torch.no_grad():
            if DEVICE.type == "cuda":
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    out = MODEL(x, side_loss=False)
            else:
                out = MODEL(x, side_loss=False)

        out = out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        out = _auto_brightness_boost(out, boost)

        elapsed = time.perf_counter() - start
        out_u8 = (out * 255.0).round().astype(np.uint8)

        h, w = out_u8.shape[:2]
        msg = (
            f"Done in {elapsed:.2f}s | output: {w}x{h} | "
            f"device: {DEVICE.type} | boost: {boost:.2f} | resized_input: {resized_size[0]}x{resized_size[1]}"
        )
        return Image.fromarray(out_u8), msg

    except Exception as exc:
        return None, f"Inference error: {exc}"


demo = gr.Interface(
    fn=process_img,
    inputs=[
        gr.Image(type="pil", label="Input low-light image"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.35, step=0.05, label="Visibility boost (optional)")
    ],
    outputs=[
        gr.Image(type="pil", label="Enhanced output"),
        gr.Textbox(label="Status", interactive=False)
    ],
    title="DarkIR Local Demo",
    description="Upload a low-light image and get the restored output using your local checkpoint.",
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
