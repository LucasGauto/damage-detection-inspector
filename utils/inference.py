
import torch
from PIL import Image
import numpy as np
from torch.nn import functional as F
from utils.load_models import load_detector, load_classifier
from utils.transform import build_transform
from config import PART_TO_ID, ID_TO_DAMAGE, DEVICE

DETECTOR = load_detector()
CLASSIFIER = load_classifier()
CLASSIFIER_TRANSFORM = build_transform()

@torch.inference_mode() # Desactiva el cálculo de gradientes para mejorar el rendimiento durante la inferencia
def detect_parts(image: Image.Image, conf_threshold: float) -> tuple[Image.Image | None, str]:
    """
    Detecta las partes del vehículo en la imagen y clasifica su estado.
    """
    if image is None:
        return None, "Subi una imagen para ejecutar la deteccion."

    image_np = np.array(image)
    results = DETECTOR.predict(source=image_np, conf=conf_threshold, verbose=False)

    if not results:
        return image, "No se obtuvo salida del detector."

    result = results[0]
    plotted = result.plot()
    output_image = Image.fromarray(plotted)

    total = 0 if result.boxes is None else len(result.boxes)
    if total == 0:
        return output_image, "No se detectaron partes."

    damaged: list[tuple[str, str, float]] = []
    undamaged: list[tuple[str, str, float]] = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        part_name = result.names[cls_id]

        if part_name not in PART_TO_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

        crop_tensor = CLASSIFIER_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
        part_id_tensor = torch.tensor([PART_TO_ID[part_name]], dtype=torch.long, device=DEVICE)
        logits = CLASSIFIER(crop_tensor, part_id_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
        predicted_id = int(probs.argmax().item())
        predicted_label = ID_TO_DAMAGE[predicted_id]
        confidence = float(probs[predicted_id].item())

        if predicted_label == "sin daño":
            undamaged.append((part_name, predicted_label, confidence))
        else:
            damaged.append((part_name, predicted_label, confidence))

    lines = [f"**Detecciones encontradas: {total}**"]

    lines.append("\n### Con daño")
    if damaged:
        lines.append("| Parte | Nivel de daño | Confianza |")
        lines.append("|---|---|---|")
        for name, label, conf in damaged:
            lines.append(f"| {name} | {label} | {conf:.2%} |")
    else:
        lines.append("_No se detectaron partes con daño._")

    lines.append("\n### Sin daño")
    if undamaged:
        lines.append("| Parte | Nivel de daño | Confianza |")
        lines.append("|---|---|---|")
        for name, label, conf in undamaged:
            lines.append(f"| {name} | {label} | {conf:.2%} |")
    else:
        lines.append("_No se detectaron partes sin daño._")

    summary = "\n".join(lines)
    return output_image, summary
