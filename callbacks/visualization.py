import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.inference import CLASSIFIER
import matplotlib.cm as cm

def make_gradcam_overlay(crop: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Superpone el mapa de calor GradCAM en la imagen recortada."""
    w, h = crop.size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]  # RGB from colormap
    heatmap = (heatmap * 255).astype(np.uint8)

    crop_np = np.array(crop.convert("RGB")).astype(np.float32)
    overlay = (1 - alpha) * crop_np + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def compute_gradcam_dual(
    image_tensor: torch.Tensor, part_id_tensor: torch.Tensor, target_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Computa los mapas de calor GradCAM en backbone.stage2 y backbone.stage3 en una sola pasada."""
    model = CLASSIFIER
    model.eval()

    activations_s2: list[torch.Tensor] = []
    gradients_s2: list[torch.Tensor] = []
    activations_s3: list[torch.Tensor] = []
    gradients_s3: list[torch.Tensor] = []

    def fwd_s2(module, input, output):
        activations_s2.append(output)

    def bwd_s2(module, grad_input, grad_output):
        gradients_s2.append(grad_output[0])

    def fwd_s3(module, input, output):
        activations_s3.append(output)

    def bwd_s3(module, grad_input, grad_output):
        gradients_s3.append(grad_output[0])

    fh2 = model.backbone.stage2.register_forward_hook(fwd_s2)
    bh2 = model.backbone.stage2.register_full_backward_hook(bwd_s2)
    fh3 = model.backbone.stage3.register_forward_hook(fwd_s3)
    bh3 = model.backbone.stage3.register_full_backward_hook(bwd_s3)

    try:
        image_tensor = image_tensor.clone().requires_grad_(True)
        logits = model(image_tensor, part_id_tensor)
        model.zero_grad()
        logits[0, target_class].backward()

        cams = []
        for act_list, grad_list in [(activations_s2, gradients_s2), (activations_s3, gradients_s3)]:
            act = act_list[0].detach()
            grad = grad_list[0].detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
            cams.append(cam)
    finally:
        fh2.remove()
        bh2.remove()
        fh3.remove()
        bh3.remove()

    return cams[0], cams[1]


def make_gradcam_overlay(crop: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Superpone el mapa de calor GradCAM en la imagen recortada."""
    w, h = crop.size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]  # RGB from colormap
    heatmap = (heatmap * 255).astype(np.uint8)

    crop_np = np.array(crop.convert("RGB")).astype(np.float32)
    overlay = (1 - alpha) * crop_np + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)