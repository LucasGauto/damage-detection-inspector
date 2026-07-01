from dotenv import load_dotenv
import torch
import os
from pathlib import Path

load_dotenv()

PART_TO_ID = {
    "Baul/Portón": 0,
    "Capot": 1,
    "Compuerta": 2,
    "Espejo": 3,
    "Faro": 4,
    "Guard.Delantero": 5,
    "Guard.Trasero": 6,
    "Lateral Caja": 7,
    "Optica": 8,
    "Parabrisas": 9,
    "Paragolpe Delantero": 10,
    "Paragolpe Trasero": 11,
    "Puerta Delantera": 12,
    "Puerta Trasera": 13,
    "Zócalo": 14,
}

DAMAGE_TO_ID = {
    "daño fuerte": 0,
    "daño leve": 1,
    "daño medio": 2,
    "sin daño": 3,
}

ID_TO_PART = {value: key for key, value in PART_TO_ID.items()}
ID_TO_DAMAGE = {value: key for key, value in DAMAGE_TO_ID.items()}
PART_CLASSES = [ID_TO_PART[index] for index in range(len(ID_TO_PART))]
DAMAGE_CLASSES = [ID_TO_DAMAGE[index] for index in range(len(ID_TO_DAMAGE))]

# Variables de entorno
TARGET_SIZE_CLASSIFIER = int(os.getenv("TARGET_SIZE_CLASSIFIER", 224))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTOR_PATH = Path(os.getenv("DETECTOR_PATH"))
CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_PATH"))