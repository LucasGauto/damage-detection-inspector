from PIL import Image
from utils.inference import DETECTOR
import gradio as gr
import numpy as np

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
