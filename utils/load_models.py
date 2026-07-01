from functools import lru_cache
import torch
from pathlib import Path
from ultralytics import YOLO
from src.models import DamageClassifierFiLM
from config import DEVICE, DETECTOR_PATH, CLASSIFIER_PATH

@lru_cache(maxsize=1) # Cachear el modelo para no cargarlo cada vez que se llama a la funcion
def load_detector() -> YOLO:
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo detector en: {DETECTOR_PATH}")
    return YOLO(str(DETECTOR_PATH))

@lru_cache(maxsize=1) # Cachear el modelo para no cargarlo cada vez que se llama a la funcion
def load_classifier() -> DamageClassifierFiLM:
    """
    Carga el modelo de clasificacion de daños y partes del vehiculo.
    """
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo clasificador en: {CLASSIFIER_PATH}"
        )

    state_dict = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
    model = DamageClassifierFiLM(pretrained=False, late_fusion=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model