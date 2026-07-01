import gradio as gr
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn.functional as F
from utils.inference import detect_parts, DETECTOR, CLASSIFIER, CLASSIFIER_TRANSFORM
from config import PART_TO_ID, ID_TO_DAMAGE, DAMAGE_CLASSES, DEVICE

def classify_detect_parts(
    image: Image.Image, conf_threshold: float
) -> tuple[Image.Image | None, list, gr.update]:
    """
    Detecta las partes del vehículo en la imagen y devuelve la imagen con las detecciones,
    una lista de diccionarios con la información de cada detección y un objeto gr.update
    """
    if image is None:
        return None, [], gr.update(choices=[], value=None)

    image_np = np.array(image)
    results = DETECTOR.predict(source=image_np, conf=conf_threshold, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return image, [], gr.update(choices=["No se detectaron partes"], value=None)

    result = results[0]
    plotted = result.plot()
    output_image = Image.fromarray(plotted)

    detections = []
    choices = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        part_name = result.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = f"{part_name} #{i + 1}"
        detections.append({"label": label, "part_name": part_name, "bbox": [x1, y1, x2, y2]})
        choices.append(label)

    return output_image, detections, gr.update(choices=choices, value=choices[0])


def make_gradcam_overlay(crop: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay GradCAM heatmap on the crop image."""
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
    """Compute GradCAM heatmaps on backbone.stage2 and backbone.stage3 in a single pass."""
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
    """Overlay GradCAM heatmap on the crop image."""
    w, h = crop.size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]  # RGB from colormap
    heatmap = (heatmap * 255).astype(np.uint8)

    crop_np = np.array(crop.convert("RGB")).astype(np.float32)
    overlay = (1 - alpha) * crop_np + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def classify_selected_part(
    image: Image.Image, selected_label: str, detections: list
) -> tuple[str, dict[str, float], Image.Image | None, Image.Image | None]:
    """Classify damage on the selected bounding box and produce dual GradCAM overlays."""
    if image is None:
        return "Subi una imagen primero.", {}, None, None

    if not detections or not selected_label:
        return "Primero detecta las partes.", {}, None, None

    detection = None
    for d in detections:
        if d["label"] == selected_label:
            detection = d
            break

    if detection is None:
        return "Parte no encontrada en las detecciones.", {}, None, None

    part_name = detection["part_name"]
    if part_name not in PART_TO_ID:
        return f"La parte '{part_name}' no es reconocida por el clasificador.", {}, None, None

    x1, y1, x2, y2 = detection["bbox"]
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

    crop_tensor = CLASSIFIER_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
    part_id_tensor = torch.tensor([PART_TO_ID[part_name]], dtype=torch.long, device=DEVICE)

    # GradCAM needs gradients, so no inference_mode
    with torch.enable_grad():
        logits = CLASSIFIER(crop_tensor, part_id_tensor)
        probabilities = F.softmax(logits, dim=-1).squeeze(0).detach().cpu()
        predicted_id = int(probabilities.argmax().item())
        predicted_label = ID_TO_DAMAGE[predicted_id]
        confidence = float(probabilities[predicted_id].item())

        cam_s2, cam_s3 = compute_gradcam_dual(crop_tensor, part_id_tensor, predicted_id)

    gradcam_s2 = make_gradcam_overlay(crop, cam_s2)
    gradcam_s3 = make_gradcam_overlay(crop, cam_s3)

    label_scores = {
        ID_TO_DAMAGE[index]: float(probabilities[index].item())
        for index in range(len(DAMAGE_CLASSES))
    }
    summary = (
        f"Parte evaluada: {part_name}\n"
        f"Daño estimado: {predicted_label}\n"
        f"Confianza: {confidence:.2%}"
    )
    return summary, label_scores, gradcam_s2, gradcam_s3

with gr.Blocks(title="Analisis de Partes y Daños") as demo:
    gr.Markdown("# Analisis de partes y daños")
    gr.Markdown(
        "La aplicacion permite correr deteccion de partes con YOLO y clasificacion de daño "
        "condicionada por parte con el modelo `models/classificator/best.pth`."
    )

    # with gr.Tab("Deteccion"):
    #     with gr.Row():
    #         detection_image_input = gr.Image(type="pil", label="Imagen de entrada")
    #         detection_image_output = gr.Image(type="pil", label="Imagen con detecciones")

    #     conf_slider = gr.Slider(
    #         minimum=0.05,
    #         maximum=0.95,
    #         value=0.25,
    #         step=0.05,
    #         label="Umbral de confianza",
    #     )
    #     detect_button = gr.Button("Detectar partes")
    #     detection_summary_output = gr.Markdown(value="", label="Resumen")

    #     detect_button.click(
    #         fn=detect_parts,
    #         inputs=[detection_image_input, conf_slider],
    #         outputs=[detection_image_output, detection_summary_output],
    #     )

    with gr.Tab("Clasificacion de daño"):
        cls_detections_state = gr.State([])

        # Interfaz
        # Fila de Deteccion
        gr.Markdown("#### Deteccion de partes")
        with gr.Row():
            
            with gr.Column(scale=1):
                with gr.Row():
                    classification_image_input = gr.Image(type="pil", label="Imagen del Vehiculo")
                # with gr.Row():
                #     cls_conf_slider = gr.Slider(minimum=0.00, maximum=1.0, value=0.1, step=0.05, label="Umbral de confianza (deteccion)")
                # with gr.Row():
                #     cls_detect_button = gr.Button("Detectar partes")
            with gr.Column(scale=1):
                classification_image_detected = gr.Image(type="pil", label="Imagen con detecciones", interactive=True, scale=1)

        with gr.Row():
            cls_conf_slider = gr.Slider(minimum=0.00, maximum=1.0, value=0.1, step=0.05, label="Umbral de confianza (deteccion)")
            cls_detect_button = gr.Button("Detectar partes")

        # Fila de Clasificacion
        gr.Markdown("#### Clasificacion de daño")
        with gr.Row():
    
            with gr.Column():
                part_input = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Parte detectada",
                    interactive=True,
                )
                classify_button = gr.Button("Clasificar daño")
                
            with gr.Column():
                classification_summary_output = gr.Textbox(
                    label="Resultado",
                    interactive=False,
                    lines=3,
                )

        with gr.Row():
            classification_scores_output = gr.Label(label="Probabilidades")

        with gr.Row():
            with gr.Column():
                gradcam_s2_output = gr.Image(
                    type="pil",
                    label="Stage 2 (penultima capa)",
                    interactive=True,
                )
            with gr.Column():
                gradcam_s3_output = gr.Image(
                    type="pil",
                    label="Stage 3 (ultima capa)",
                    interactive=True,
                )

        cls_detect_button.click(
            fn=classify_detect_parts,
            inputs=[classification_image_input, cls_conf_slider],
            outputs=[classification_image_detected, cls_detections_state, part_input],
        )

        classify_button.click(
            fn=classify_selected_part,
            inputs=[classification_image_input, part_input, cls_detections_state],
            outputs=[classification_summary_output, classification_scores_output, gradcam_s2_output, gradcam_s3_output],
        )


if __name__ == "__main__":
    demo.launch(debug=True)