"""
Zentrale Konfiguration für verschiedene Modelle (car, butterfly).
Alle modellabhängigen Parameter werden hier definiert.
"""

from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Modell-Definitionen
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "car": {
        "classes": [
            'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
            'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
            'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
            'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
            'tailgate', 'trunk', 'wheel'
        ],
        "num_classes": 23,
        "dataset_name": "car_parts_train",
        "dataset_root": "/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets",
        "annotations_train": "annotations/instances_train2017.json",
        "annotations_val": "annotations/instances_val2017.json",
        "images_subdir": "images",
    },
    "butterfly": {
        "classes": [
            "Danaus plexippus",       # 001 - Monarch
            "Heliconius charitonius", # 002 - Zebra Longwing
            "Heliconius erato",       # 003 - Red Postman
            "Junonia coenia",         # 004 - Common Buckeye
            "Lycaena phlaeas",        # 005 - Small Copper
            "Nymphalis antiopa",      # 006 - Mourning Cloak
            "Papilio cresphontes",    # 007 - Giant Swallowtail
            "Pieris rapae",           # 008 - Cabbage White
            "Vanessa atalanta",       # 009 - Red Admiral
            "Vanessa cardui",         # 010 - Painted Lady
        ],
        "num_classes": 10,
        "dataset_name": "butterfly_train",
        "dataset_root": "/Users/nicklehmacher/Alles/MasterArbeit/leedsbutterfly/coco",
        "annotations_train": "annotations/instances_train2017.json",
        "annotations_val": "annotations/instances_val2017.json",
        "images_subdir": "train2017",
    },
}


def get_model_config(model: str) -> Dict[str, Any]:
    """
    Gibt die Konfiguration für ein bestimmtes Modell zurück.
    
    Args:
        model: Name des Modells ("car" oder "butterfly")
        
    Returns:
        Dictionary mit allen modellspezifischen Parametern
        
    Raises:
        ValueError: Wenn das Modell nicht bekannt ist
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unbekanntes Modell: '{model}'. Verfügbare Modelle: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model]


def get_classes(model: str) -> List[str]:
    """Gibt die Klassenliste für ein Modell zurück."""
    return get_model_config(model)["classes"]


def get_num_classes(model: str) -> int:
    """Gibt die Anzahl der Klassen für ein Modell zurück."""
    return get_model_config(model)["num_classes"]


def get_dataset_name(model: str) -> str:
    """Gibt den Dataset-Namen für ein Modell zurück."""
    return get_model_config(model)["dataset_name"]


def get_dataset_root(model: str) -> str:
    """Gibt den Dataset-Root-Pfad für ein Modell zurück."""
    return get_model_config(model)["dataset_root"]
