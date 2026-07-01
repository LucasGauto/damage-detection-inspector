
from PIL import Image
import torch
import torch.nn.functional as F
from config import PART_TO_ID, ID_TO_DAMAGE, DAMAGE_CLASSES, DEVICE
from utils.inference import CLASSIFIER, CLASSIFIER_TRANSFORM
from callbacks.visualization import make_gradcam_overlay, compute_gradcam_dual

def classify_selected_part(
    image: Image.Image, 
    selected_label: str, 
    detections: list
) -> tuple[str, dict[str, float], Image.Image | None, Image.Image | None]:
    
    """Clasificar daño en el bouynding box seleccionado y producir overlays GradCAM duales."""
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

    # GradCAM necesita gradientes, por lo que no se usa inference_mode
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