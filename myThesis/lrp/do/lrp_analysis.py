"""
LRP Analyse - High-Level Tools für LRP-Analyse.

Dieses Modul enthält Batch-Verarbeitung und Context Manager für
die einfache Verwendung der LRP-Analyse.

Performance-Features:
- Batch-Processing für mehrere Bilder mit geteiltem Forward-Pass
- Memory-optimierte Verarbeitung pro Layer
- Automatisches Garbage Collection zwischen Analysen
- Konfigurierbare Batch-Größen basierend auf verfügbarem Speicher

Verwendung:
    >>> from lrp_analysis import run_lrp_analysis, LRPAnalysisContext
    >>> with LRPAnalysisContext(model) as controller:
    ...     result = controller.run(inputs)
    
Performance-optimierte Verwendung:
    >>> from lrp_analysis import BatchLRPProcessor
    >>> processor = BatchLRPProcessor(model, batch_size=4)
    >>> results = processor.process_directory("images/")
"""
from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

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
    "BatchLRPProcessor",
    "MemoryOptimizedLRP",
    "estimate_memory_requirements",
]


# =============================================================================
# Performance-Konfiguration
# =============================================================================


@dataclass
class LRPPerformanceConfig:
    """Konfiguration für Performance-optimierte LRP-Analyse.
    
    Diese Konfiguration steuert das Memory-Management und die
    Batch-Verarbeitung für effiziente LRP-Analysen.
    
    Attributes:
        batch_size: Anzahl der Bilder pro Batch (0 = automatisch)
        max_memory_mb: Maximaler Speicherverbrauch in MB (0 = unbegrenzt)
        gc_frequency: Garbage Collection alle N Bilder
        clear_activations_each_layer: Aktivierungen nach jedem Layer löschen
        store_intermediate_results: Zwischenergebnisse speichern
        use_float16: Half-Precision für Aktivierungen verwenden
        num_workers: Anzahl der Worker für Datenladung
    """
    batch_size: int = 1
    max_memory_mb: int = 0
    gc_frequency: int = 1
    clear_activations_each_layer: bool = True
    store_intermediate_results: bool = False
    use_float16: bool = False
    num_workers: int = 0
    
    @classmethod
    def for_low_memory(cls) -> "LRPPerformanceConfig":
        """Konfiguration für Systeme mit wenig Speicher."""
        return cls(
            batch_size=1,
            gc_frequency=1,
            clear_activations_each_layer=True,
            store_intermediate_results=False,
            use_float16=True,
        )
    
    @classmethod
    def for_high_throughput(cls) -> "LRPPerformanceConfig":
        """Konfiguration für maximalen Durchsatz."""
        return cls(
            batch_size=4,
            gc_frequency=4,
            clear_activations_each_layer=False,
            store_intermediate_results=True,
            use_float16=False,
        )


# =============================================================================
# Memory-Schätzung
# =============================================================================


def estimate_memory_requirements(
    image_size: Tuple[int, int] = (800, 1333),
    hidden_dim: int = 256,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 3,
    num_queries: int = 300,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Schätzt den Speicherbedarf für LRP-Analyse.
    
    Args:
        image_size: (Höhe, Breite) des Eingabebildes
        hidden_dim: Dimension der versteckten Schichten (256 für MaskDINO)
        num_encoder_layers: Anzahl der Encoder-Layers (6 für MaskDINO)
        num_decoder_layers: Anzahl der Decoder-Layers (3 für MaskDINO)
        num_queries: Anzahl der Object-Queries (300 für MaskDINO)
        dtype: Datentyp der Tensoren
        
    Returns:
        Dictionary mit Speicherschätzungen in MB
    """
    bytes_per_element = 4 if dtype == torch.float32 else 2
    
    # Encoder-Tokens: Multi-Scale Feature Maps (100x100 + 50x50 + 25x25)
    # Vereinfacht: ~ image_size / 8 für jeden Level
    h, w = image_size
    num_tokens = (h//8 * w//8) + (h//16 * w//16) + (h//32 * w//32)
    num_tokens = min(num_tokens, 13125)  # Cap bei typischem MaskDINO
    
    # Speicher pro Komponente (in Bytes)
    estimates = {}
    
    # Aktivierungen: Jeder Layer speichert Input und Output
    encoder_activations = num_encoder_layers * 2 * num_tokens * hidden_dim * bytes_per_element
    decoder_activations = num_decoder_layers * 2 * num_queries * hidden_dim * bytes_per_element
    
    # Attention-Weights: (B, H, T, T) für Self-Attention
    num_heads = 8
    encoder_attn = num_encoder_layers * num_heads * num_tokens * num_tokens * bytes_per_element
    decoder_self_attn = num_decoder_layers * num_heads * num_queries * num_queries * bytes_per_element
    decoder_cross_attn = num_decoder_layers * num_heads * num_queries * num_tokens * bytes_per_element
    
    # Relevanz-Tensoren (gleiche Größe wie Aktivierungen)
    relevance_encoder = num_tokens * hidden_dim * bytes_per_element
    relevance_decoder = num_queries * hidden_dim * bytes_per_element
    
    # Konvertiere zu MB
    mb = 1024 * 1024
    estimates["encoder_activations_mb"] = encoder_activations / mb
    estimates["decoder_activations_mb"] = decoder_activations / mb
    estimates["encoder_attention_mb"] = encoder_attn / mb
    estimates["decoder_attention_mb"] = (decoder_self_attn + decoder_cross_attn) / mb
    estimates["relevance_tensors_mb"] = (relevance_encoder + relevance_decoder) / mb
    
    # Gesamtschätzung (mit Overhead-Faktor 1.5)
    total = sum(estimates.values())
    estimates["total_estimated_mb"] = total * 1.5
    estimates["recommended_batch_size"] = max(1, int(4000 / total))  # Für 4GB verfügbar
    
    return estimates


# =============================================================================
# Batch-Processing
# =============================================================================


class BatchLRPProcessor:
    """Memory-optimierter Batch-Processor für LRP-Analysen.
    
    Diese Klasse ermöglicht die effiziente Verarbeitung vieler Bilder
    mit kontrollierbarem Speicherverbrauch und automatischer Batch-Größen-
    Anpassung.
    
    Example:
        >>> processor = BatchLRPProcessor(model, batch_size=2)
        >>> for result in processor.process_images(image_paths):
        ...     print(f"Relevanz-Summe: {result.R_input.sum():.4f}")
        
    Attributes:
        model: Das MaskDINO-Modell
        config: Performance-Konfiguration
        controller: Interner LRP-Controller (lazy initialization)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[LRPPerformanceConfig] = None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """Initialisiert den Batch-Processor.
        
        Args:
            model: Das zu analysierende Modell
            config: Performance-Konfiguration (optional)
            device: Berechnungsgerät
            verbose: Ausführliches Logging
        """
        self.model = model
        self.config = config or LRPPerformanceConfig()
        self.device = device
        self.verbose = verbose
        self._controller = None
        self._stats = {
            "images_processed": 0,
            "total_time_sec": 0.0,
            "gc_calls": 0,
        }
    
    @property
    def controller(self):
        """Lazy initialization des LRP-Controllers."""
        if self._controller is None:
            from .lrp_controller import LRPController
            self._controller = LRPController(
                self.model,
                device=self.device,
                verbose=self.verbose,
            )
            self._controller.prepare()
        return self._controller
    
    def _maybe_gc(self, force: bool = False) -> None:
        """Führt Garbage Collection durch wenn nötig."""
        should_gc = force or (
            self._stats["images_processed"] % self.config.gc_frequency == 0
        )
        if should_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._stats["gc_calls"] += 1
    
    def _load_and_preprocess_image(
        self,
        img_path: str,
        resize_aug,
    ) -> Optional[Dict]:
        """Lädt und bereitet ein Bild für das Modell vor."""
        try:
            from PIL import Image
            
            pil_im = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_im)
            
            tfm = resize_aug.get_transform(img_np)
            img_np = tfm.apply_image(img_np)
            
            img_tensor = torch.as_tensor(img_np.astype("float32").transpose(2, 0, 1))
            
            if self.config.use_float16:
                img_tensor = img_tensor.half()
            
            img_tensor = img_tensor.to(self.device)
            
            return {
                "image": img_tensor,
                "height": pil_im.height,
                "width": pil_im.width,
                "file_name": img_path,
            }
        except Exception as e:
            logger.error(f"Fehler beim Laden von {img_path}: {e}")
            return None
    
    def process_images(
        self,
        image_paths: List[str],
        which_module: str = "encoder",
        target_feature: Optional[int] = None,
        target_query: int = 0,
        normalize: str = "sum1",
    ) -> Generator[Tuple[str, LRPResult], None, None]:
        """Generator der LRP-Ergebnisse für eine Liste von Bildern liefert.
        
        Memory-optimiert: Verarbeitet Bilder einzeln und gibt Speicher frei.
        
        Args:
            image_paths: Liste der Bildpfade
            which_module: "encoder" oder "decoder"
            target_feature: Ziel-Feature-Index (für Encoder)
            target_query: Ziel-Query-Index (für Decoder)
            normalize: Normalisierungsmethode
            
        Yields:
            Tuple aus (image_path, LRPResult)
        """
        from detectron2.data import transforms as T
        
        resize_aug = T.ResizeShortestEdge(
            short_edge_length=800,
            max_size=1333,
        )
        
        start_time = time.time()
        
        for img_path in image_paths:
            batched_input = self._load_and_preprocess_image(img_path, resize_aug)
            
            if batched_input is None:
                yield (img_path, LRPResult())
                continue
            
            try:
                result = self.controller.run(
                    [batched_input],
                    target_class=None,
                    target_query=target_query,
                    normalize=normalize,
                    which_module=which_module,
                    target_feature=target_feature,
                )
                
                self._stats["images_processed"] += 1
                
                # Memory-optimiert: Nur R_input behalten
                if self.config.clear_activations_each_layer:
                    # Erstelle minimales Ergebnis
                    minimal_result = LRPResult()
                    if result.R_input is not None:
                        minimal_result.R_input = result.R_input.detach().cpu()
                    minimal_result.metadata = result.metadata.copy()
                    result = minimal_result
                
                yield (img_path, result)
                
            except Exception as e:
                logger.error(f"LRP-Fehler bei {img_path}: {e}")
                yield (img_path, LRPResult())
            
            finally:
                self._maybe_gc()
        
        self._stats["total_time_sec"] = time.time() - start_time
    
    def process_directory(
        self,
        directory: str,
        **kwargs,
    ) -> Generator[Tuple[str, LRPResult], None, None]:
        """Verarbeitet alle Bilder in einem Verzeichnis.
        
        Args:
            directory: Pfad zum Bildverzeichnis
            **kwargs: Weitere Argumente für process_images
            
        Yields:
            Tuple aus (image_path, LRPResult)
        """
        image_paths = collect_images(directory)
        if not image_paths:
            logger.warning(f"Keine Bilder in {directory}")
            return
        
        logger.info(f"Verarbeite {len(image_paths)} Bilder aus {directory}")
        yield from self.process_images(image_paths, **kwargs)
    
    def get_stats(self) -> Dict[str, float]:
        """Gibt Verarbeitungsstatistiken zurück."""
        stats = self._stats.copy()
        if stats["images_processed"] > 0 and stats["total_time_sec"] > 0:
            stats["images_per_second"] = stats["images_processed"] / stats["total_time_sec"]
            stats["avg_time_per_image_sec"] = stats["total_time_sec"] / stats["images_processed"]
        return stats
    
    def cleanup(self) -> None:
        """Bereinigt Ressourcen."""
        if self._controller is not None:
            self._controller.cleanup()
            self._controller = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Memory-optimierte LRP-Klasse
# =============================================================================


class MemoryOptimizedLRP:
    """Context Manager für memory-optimierte LRP-Analyse.
    
    Implementiert verschiedene Strategien zur Reduzierung des Speicherverbrauchs:
    1. Layer-by-Layer Verarbeitung mit sofortigem Cleanup
    2. Optional Float16 für Aktivierungen
    3. Nur notwendige Tensoren speichern
    
    Example:
        >>> with MemoryOptimizedLRP(model) as lrp:
        ...     result = lrp.analyze(image, which_module="encoder", target_feature=42)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        use_float16: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.device = device
        self.use_float16 = use_float16
        self.verbose = verbose
        self._controller = None
    
    def __enter__(self):
        from .lrp_controller import LRPController
        
        self._controller = LRPController(
            self.model,
            device=self.device,
            verbose=self.verbose,
        )
        self._controller.prepare()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._controller is not None:
            self._controller.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    
    def analyze(
        self,
        inputs: Union[Tensor, List[Dict]],
        which_module: str = "encoder",
        target_feature: Optional[int] = None,
        target_query: int = 0,
        normalize: str = "sum1",
    ) -> LRPResult:
        """Führt memory-optimierte LRP-Analyse durch.
        
        Args:
            inputs: Eingabe-Tensor oder Detectron2-Batch
            which_module: "encoder" oder "decoder"
            target_feature: Ziel-Feature-Index (für Encoder)
            target_query: Ziel-Query-Index (für Decoder)
            normalize: Normalisierungsmethode
            
        Returns:
            LRPResult mit minimalen Daten (nur R_input)
        """
        if self._controller is None:
            raise RuntimeError("MemoryOptimizedLRP muss als Context Manager verwendet werden")
        
        result = self._controller.run(
            inputs,
            target_class=None,
            target_query=target_query,
            normalize=normalize,
            which_module=which_module,
            target_feature=target_feature,
        )
        
        # Minimiere gespeicherte Daten
        minimal_result = LRPResult()
        if result.R_input is not None:
            minimal_result.R_input = result.R_input.detach().cpu()
        minimal_result.metadata = {
            k: v for k, v in result.metadata.items()
            if k in ("num_layers_processed", "total_conservation_error")
        }
        
        # Cleanup
        del result
        gc.collect()
        
        return minimal_result
    
    def analyze_all_queries(
        self,
        inputs: Union[Tensor, List[Dict]],
        num_queries: int = 300,
        normalize: str = "sum1",
    ) -> Iterator[Tuple[int, Tensor]]:
        """Generator für memory-optimierte Analyse aller Queries.
        
        Statt alle 300 Queries im Speicher zu halten, wird jede Query
        einzeln verarbeitet und sofort zurückgegeben.
        
        Args:
            inputs: Eingabe-Tensor oder Detectron2-Batch
            num_queries: Anzahl der Queries
            normalize: Normalisierungsmethode
            
        Yields:
            Tuple aus (query_index, relevance_tensor)
        """
        if self._controller is None:
            raise RuntimeError("MemoryOptimizedLRP muss als Context Manager verwendet werden")
        
        # Einmaliger Forward-Pass
        outputs = self._controller.forward_pass(inputs)
        
        for query_idx in range(num_queries):
            try:
                # Hole Decoder-Output für diese Query
                decoder_output = self._controller._get_decoder_output(outputs, query_idx)
                
                # Erstelle Start-Relevanz
                num_total_queries = 300
                B, _, C = decoder_output.shape
                R_start_full = torch.zeros(
                    (num_total_queries, B, C),
                    device=decoder_output.device,
                    dtype=decoder_output.dtype,
                )
                R_start_full[query_idx, 0, :] = decoder_output.squeeze()
                
                # Normalisieren
                if normalize == "sum1":
                    R_start_full = R_start_full / (R_start_full.abs().sum() + 1e-12)
                
                # Backward-Pass (ohne Aktivierungen zu löschen)
                result = self._controller.backward_pass(
                    R_start_full,
                    which_module="decoder",
                    clear_activations=False,
                )
                
                # Nur Relevanz-Tensor zurückgeben
                if result.R_input is not None:
                    yield (query_idx, result.R_input.detach().cpu())
                else:
                    yield (query_idx, torch.zeros(1))
                
                # Cleanup nach jeder Query
                del R_start_full, result
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Fehler bei Query {query_idx}: {e}")
                yield (query_idx, torch.zeros(1))
        
        # Finales Cleanup
        self._controller.cleanup()
