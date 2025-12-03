"""
LRP Analyse - High-Level Tools für LRP-Analyse.

Dieses Modul enthält Batch-Verarbeitung und Context Manager für
die einfache Verwendung der LRP-Analyse.

Verwendung:
    >>> from lrp_analysis import run_lrp_analysis, LRPAnalysisContext
    >>> with LRPAnalysisContext(model) as controller:
    ...     result = controller.run(inputs)
"""
from __future__ import annotations

import gc
import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .io_utils import collect_images
from .lrp_structs import LRPResult
from .tensor_ops import aggregate_channel_relevance


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger("lrp.analysis")


# =============================================================================
# Batch-Analyse
# =============================================================================


def run_lrp_analysis(
    model: nn.Module,
    images_dir: str,
    output_csv: str,
    target_class: Optional[int] = None,
    target_query: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    """Führt LRP-Analyse über einen Ordner mit Bildern durch.
    
    Args:
        model: Das vorbereitete Modell
        images_dir: Pfad zum Bildordner
        output_csv: Ausgabepfad für CSV-Ergebnisse
        target_class: Zielklasse für Attribution
        target_query: Ziel-Query-Index
        device: Berechnungsgerät
        verbose: Ausführliches Logging
    """
    import pandas as pd
    from PIL import Image
    from detectron2.data import transforms as T
    
    # Importiere LRPController hier, um zirkuläre Imports zu vermeiden
    from .lrp_controller import LRPController
    
    controller = LRPController(model, device=device, verbose=verbose)
    controller.prepare()
    
    # Sammle Bilder
    img_files = collect_images(images_dir)
    if not img_files:
        raise FileNotFoundError(f"Keine Bilder in {images_dir}")
    
    # Bildvorverarbeitung
    resize_aug = T.ResizeShortestEdge(
        short_edge_length=800,
        max_size=1333,
    )
    
    results = []
    
    for img_path in img_files:
        try:
            # Bild laden
            pil_im = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_im)
            
            # Resize
            tfm = resize_aug.get_transform(img_np)
            img_np = tfm.apply_image(img_np)
            
            # Zu Tensor
            img_tensor = torch.as_tensor(img_np.astype("float32").transpose(2, 0, 1))
            img_tensor = img_tensor.to(device)
            
            # Batch erstellen
            batched_inputs = [{
                "image": img_tensor,
                "height": pil_im.height,
                "width": pil_im.width,
            }]
            
            # LRP ausführen
            result = controller.run(
                batched_inputs,
                target_class=target_class,
                target_query=target_query,
            )
            
            # Aggregiere Relevanz
            if result.R_input is not None:
                relevance = aggregate_channel_relevance(result.R_input)
                results.append({
                    "image": os.path.basename(img_path),
                    "total_relevance": relevance.sum().item(),
                    "conservation_error": result.metadata.get("total_conservation_error", 0),
                    "num_layers": result.metadata.get("num_layers_processed", 0),
                })
                
        except Exception as e:
            logger.error(f"Fehler bei {img_path}: {e}")
            continue
    
    # Speichere Ergebnisse
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Ergebnisse gespeichert: {output_csv}")


def run_lrp_batch(
    controller,
    inputs_list: List[Union[Tensor, List[dict]]],
    target_class: Optional[int] = None,
    target_query: int = 0,
    normalize: str = "sum1",
) -> List[LRPResult]:
    """Führt LRP-Analyse für mehrere Eingaben durch.
    
    Effizienter als wiederholte Aufrufe von controller.run(), da
    Speicher zwischen Analysen bereinigt wird.
    
    Args:
        controller: Ein vorbereiteter LRPController
        inputs_list: Liste von Eingaben (Tensor oder Detectron2-Batch)
        target_class: Zielklasse für Attribution
        target_query: Ziel-Query-Index
        normalize: Normalisierungsmethode
        
    Returns:
        Liste von LRPResult Objekten
    """
    results = []
    
    for i, inputs in enumerate(inputs_list):
        try:
            result = controller.run(
                inputs,
                target_class=target_class,
                target_query=target_query,
                normalize=normalize,
            )
            results.append(result)
            
            # Speicher freigeben nach jeder Analyse
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Fehler bei Eingabe {i}: {e}")
            results.append(LRPResult())  # Leeres Ergebnis
    
    return results


# =============================================================================
# Context Manager
# =============================================================================


class LRPAnalysisContext:
    """Context Manager für sichere LRP-Analyse.
    
    Stellt sicher, dass Aktivierungen bereinigt und der LRP-Modus
    zurückgesetzt wird, auch bei Exceptions.
    
    Example:
        >>> with LRPAnalysisContext(model) as controller:
        ...     result = controller.run(inputs)
        
    Attributes:
        controller: Der interne LRPController
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        """Initialisiert den Context Manager.
        
        Args:
            model: Das zu analysierende Modell
            **kwargs: Zusätzliche Argumente für LRPController
        """
        # Importiere LRPController hier, um zirkuläre Imports zu vermeiden
        from .lrp_controller import LRPController
        self.controller = LRPController(model, **kwargs)
    
    def __enter__(self):
        """Bereitet den Controller vor und gibt ihn zurück."""
        self.controller.prepare()
        return self.controller
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Bereinigt Aktivierungen und Speicher."""
        self.controller.cleanup()
        return False


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "run_lrp_analysis",
    "run_lrp_batch",
    "LRPAnalysisContext",
]
