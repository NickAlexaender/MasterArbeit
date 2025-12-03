"""
LRP Controller - Der Manager für LRP-Analyse (Flow Control).

Dieses Modul enthält die LRPController-Klasse, die für den Workflow
der LRP-Analyse verantwortlich ist: prepare, forward, backward loop
und State-Management.

Die mathematische Propagationslogik liegt in lrp_propagators.py.
Datenstrukturen sind in lrp_structs.py definiert.
High-Level Tools (Batch-Verarbeitung) sind in lrp_analysis.py.

Verwendung:
    >>> from lrp_controller import LRPController
    >>> controller = LRPController(model, verbose=True)
    >>> controller.prepare()  # Swapped Module für LRP-Capture
    >>> relevance_map = controller.run(image_tensor, target_class=0)
"""
from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

# Lokale Module - Konfiguration
from .config import (
    ATTN_QK_SHARE,
    LN_RULE,
    RESIDUAL_SPLIT,
    SIGN_PRESERVING,
)

# Lokale Module - Modell-Graph und Parameter
from .model_graph_wrapper import LayerNode, ModelGraph
from .param_patcher import (
    LRPModuleMixin,
    clear_all_activations,
    prepare_model_for_lrp,
    set_lrp_mode,
)

# Lokale Module - Tensor-Operationen
from .tensor_ops import build_target_relevance

# LRP Datenstrukturen (aus lrp_structs.py)
from .lrp_structs import LRPResult, LayerRelevance

# LRP Propagatoren (aus lrp_propagators.py)
from .lrp_propagators import propagate_layer, propagate_residual


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger("lrp.controller")


# =============================================================================
# LRP Controller - Hauptklasse
# =============================================================================


class LRPController:
    """Zentraler Controller für Layer-wise Relevance Propagation.
    
    Diese Klasse verwaltet den vollständigen LRP-Workflow:
    1. Modell-Vorbereitung (Modul-Swapping für Aktivierungsspeicherung)
    2. Forward Pass mit Aktivierungserfassung
    3. Backward Pass mit Relevanz-Propagation
    4. Aggregation und Konservierungsprüfung
    
    Attributes:
        model: Das zu analysierende PyTorch-Modell
        graph: Der linearisierte Modell-Graph
        device: Zielgerät (cpu/cuda)
        eps: Epsilon für numerische Stabilität
        verbose: Ausführliche Logging-Ausgabe
        
    Example:
        >>> controller = LRPController(maskdino_model)
        >>> controller.prepare()
        >>> result = controller.run(image, target_query=0)
        >>> heatmap = result.R_input.sum(dim=1)  # Aggregiere über Kanäle
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        eps: float = 1e-6,
        verbose: bool = False,
    ):
        """Initialisiert den LRP Controller.
        
        Args:
            model: Das zu analysierende Modell (z.B. MaskDINO)
            device: Zielgerät für Berechnungen
            eps: Epsilon für numerische Stabilität
            verbose: Aktiviert ausführliches Logging
        """
        self.model = model
        self.device = device
        self.eps = eps
        self.verbose = verbose
        self._prepared = False
        self._graph: Optional[ModelGraph] = None
        
        # Speicher für Raw-Decoder-Outputs (vor Post-Processing)
        self._decoder_raw_outputs: Optional[Dict[str, Tensor]] = None
        self._decoder_hook_handle = None
        
        # Konfiguration
        self.residual_split_mode = RESIDUAL_SPLIT
        self.ln_rule = LN_RULE
        self.attn_qk_share = ATTN_QK_SHARE
        self.sign_preserving = SIGN_PRESERVING
        
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def _register_decoder_hook(self) -> None:
        """Registriert einen Hook auf dem Decoder, um Raw-Outputs abzufangen.
        
        MaskDINO löscht die pred_logits im Inferenz-Modus bevor sie zurückgegeben
        werden. Dieser Hook fängt sie vorher ab.
        """
        # Entferne alten Hook falls vorhanden
        if self._decoder_hook_handle is not None:
            self._decoder_hook_handle.remove()
            self._decoder_hook_handle = None
        
        # Finde den sem_seg_head (MaskDINO-spezifisch)
        decoder_module = None
        for name, module in self.model.named_modules():
            if name == "sem_seg_head" or name.endswith(".sem_seg_head"):
                decoder_module = module
                break
        
        if decoder_module is None:
            if self.verbose:
                logger.warning("sem_seg_head nicht gefunden - kein Decoder-Hook registriert")
            return
        
        def hook_fn(module, args, output):
            """Speichert die Raw-Decoder-Outputs."""
            # MaskDINO gibt (outputs_dict, mask_dict) zurück
            if isinstance(output, tuple) and len(output) >= 1:
                raw_output = output[0]
                if isinstance(raw_output, dict) and "pred_logits" in raw_output:
                    self._decoder_raw_outputs = {
                        k: v.clone() if isinstance(v, Tensor) else v
                        for k, v in raw_output.items()
                    }
                    if self.verbose:
                        logger.debug(f"Decoder-Raw-Outputs gespeichert: {list(self._decoder_raw_outputs.keys())}")
        
        self._decoder_hook_handle = decoder_module.register_forward_hook(hook_fn)
        if self.verbose:
            logger.info("Decoder-Hook registriert auf sem_seg_head")
        
    # =========================================================================
    # Vorbereitung
    # =========================================================================
    
    def prepare(
        self,
        swap_linear: bool = False,
        swap_layernorm: bool = True,
        swap_mha: bool = True,
        swap_msdeform: bool = True,
    ) -> Dict[str, int]:
        """Bereitet das Modell für LRP-Analyse vor.
        
        Ersetzt PyTorch-Standard-Module durch LRP-fähige Versionen, die
        Aktivierungen intern speichern können.
        
        Args:
            swap_linear: Ersetze nn.Linear (kann sehr viele sein)
            swap_layernorm: Ersetze nn.LayerNorm
            swap_mha: Ersetze nn.MultiheadAttention
            swap_msdeform: Ersetze MSDeformAttn
            
        Returns:
            Dictionary mit Anzahl der ersetzten Module pro Typ
        """
        if self._prepared:
            logger.warning("Modell wurde bereits vorbereitet. Überspringe.")
            return {}
        
        # Module austauschen
        stats = prepare_model_for_lrp(
            self.model,
            swap_linear=swap_linear,
            swap_layernorm=swap_layernorm,
            swap_mha=swap_mha,
            swap_msdeform=swap_msdeform,
            verbose=self.verbose,
        )
        
        # Decoder-Hook für Raw-Outputs registrieren
        self._register_decoder_hook()
        
        # Modell-Graph erstellen
        self._graph = ModelGraph(
            self.model,
            include_leaf_modules=True,
            verbose=self.verbose,
        )
        
        self._prepared = True
        
        if self.verbose:
            logger.info(f"LRP-Vorbereitung abgeschlossen: {stats}")
            logger.info(f"Graph enthält {len(self._graph)} Layer")
        
        return stats
    
    @property
    def graph(self) -> ModelGraph:
        """Gibt den Modell-Graphen zurück (lazy initialization)."""
        if self._graph is None:
            self._graph = ModelGraph(
                self.model,
                include_leaf_modules=True,
                verbose=self.verbose,
            )
        return self._graph
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward_pass(
        self,
        inputs: Union[Tensor, List[Dict]],
    ) -> Dict[str, any]:
        """Führt den Forward Pass mit Aktivierungserfassung durch.
        
        Args:
            inputs: Eingabe-Tensor oder Detectron2-kompatible Batch-Liste
            
        Returns:
            Model-Ausgabe (predictions)
        """
        # LRP-Modus aktivieren
        set_lrp_mode(self.model, enabled=True)
        
        # Raw-Outputs zurücksetzen
        self._decoder_raw_outputs = None
        
        try:
            with torch.inference_mode():
                outputs = self.model(inputs)
        finally:
            # LRP-Modus bleibt aktiv für Backward Pass
            pass
        
        return outputs
    
    # =========================================================================
    # Backward Pass - Hauptloop
    # =========================================================================
    
    def backward_pass(
        self,
        R_start: Tensor,
        target_layer: Optional[str] = None,
    ) -> LRPResult:
        """Führt den LRP Backward Pass durch.
        
        Iteriert durch den Modell-Graphen in umgekehrter Reihenfolge und
        propagiert Relevanz gemäß den Layer-spezifischen LRP-Regeln.
        
        Args:
            R_start: Start-Relevanz (typisch auf Decoder-Output)
            target_layer: Optional - stoppe bei diesem Layer
            
        Returns:
            LRPResult mit Relevanz-Maps und Metadaten
        """
        result = LRPResult()
        R_current = R_start.clone()
        
        # Hole die Layer-Liste in LRP-Propagations-Reihenfolge
        # (Decoder -> Encoder -> Pixel Decoder -> Backbone)
        propagation_order = self.graph.get_lrp_propagation_order()
        
        if self.verbose:
            logger.info(f"Starte Backward Pass mit {len(propagation_order)} Layern")
        
        for layer_name, module in propagation_order:
            # Stoppe bei Ziel-Layer falls angegeben
            if target_layer and layer_name == target_layer:
                if self.verbose:
                    logger.info(f"Erreiche Ziel-Layer: {layer_name}")
                break
            
            # Hole Layer-Node für Metadaten
            node = self.graph.nodes.get(layer_name)
            if node is None:
                continue
            
            # Relevanz propagieren basierend auf Layer-Typ
            try:
                R_prev = self._propagate_layer(
                    module=module,
                    node=node,
                    R_out=R_current,
                )
                
                # Konservierungsprüfung
                conservation_error = self._check_conservation(R_current, R_prev)
                result.conservation_errors.append(conservation_error)
                
                if abs(conservation_error) > 0.01 and self.verbose:
                    logger.warning(
                        f"Konservierungsfehler bei {layer_name}: {conservation_error:.4f}"
                    )
                
                # Speichere Relevanz für diesen Layer
                result.R_per_layer[layer_name] = R_prev.clone()
                
                # Update für nächste Iteration
                R_current = R_prev
                
            except Exception as e:
                logger.error(f"Fehler bei Layer {layer_name}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        result.R_input = R_current
        result.metadata["num_layers_processed"] = len(result.R_per_layer)
        result.metadata["total_conservation_error"] = sum(result.conservation_errors)
        
        return result
    
    def _propagate_layer(
        self,
        module: nn.Module,
        node: LayerNode,
        R_out: Tensor,
    ) -> Tensor:
        """Propagiert Relevanz durch einen einzelnen Layer.
        
        Delegiert die eigentliche Berechnung an die Propagator-Funktionen
        aus lrp_propagators.py.
        
        Args:
            module: Das PyTorch-Modul
            node: Der zugehörige LayerNode
            R_out: Ausgabe-Relevanz
            
        Returns:
            R_in: Eingabe-Relevanz
        """
        # Prüfe ob es ein LRP-fähiges Modul ist
        if not isinstance(module, LRPModuleMixin):
            # Für nicht-LRP-Module: Identitäts-Propagation
            return R_out
        
        # Hole gespeicherte Aktivierungen
        activations = module.activations
        if activations is None or activations.input is None:
            logger.warning(f"Keine Aktivierungen für {node.name} - Identität")
            return R_out
        
        # Delegiere an generische Propagator-Funktion
        return propagate_layer(
            module=module,
            activations=activations,
            R_out=R_out,
            eps=self.eps,
            ln_rule=self.ln_rule,
            attn_qk_share=self.attn_qk_share,
        )
    
    # =========================================================================
    # Residual-Handling
    # =========================================================================
    
    def propagate_with_residual(
        self,
        x: Tensor,
        Fx: Tensor,
        R_y: Tensor,
    ) -> tuple:
        """Propagiert Relevanz durch eine Residual-Verbindung y = x + F(x).
        
        Args:
            x: Skip-Pfad Aktivierungen
            Fx: Transform-Pfad Aktivierungen
            R_y: Relevanz am Ausgang
            
        Returns:
            (R_x, R_Fx): Relevanz für Skip- und Transform-Pfad
        """
        return propagate_residual(
            x=x,
            Fx=Fx,
            R_y=R_y,
            mode=self.residual_split_mode,
            eps=self.eps,
        )
    
    # =========================================================================
    # Hilfsmethoden
    # =========================================================================
    
    def _check_conservation(self, R_out: Tensor, R_in: Tensor) -> float:
        """Prüft die Relevanz-Konservierung.
        
        LRP sollte idealerweise konservativ sein: sum(R_in) ≈ sum(R_out)
        
        Returns:
            Relativer Konservierungsfehler
        """
        sum_out = R_out.sum().item()
        sum_in = R_in.sum().item()
        
        if abs(sum_out) < 1e-12:
            return 0.0
        
        return (sum_in - sum_out) / abs(sum_out)
    
    def cleanup(self):
        """Bereinigt Aktivierungen und setzt LRP-Modus zurück."""
        set_lrp_mode(self.model, enabled=False)
        clear_all_activations(self.model)
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # =========================================================================
    # High-Level API
    # =========================================================================
    
    def run(
        self,
        inputs: Union[Tensor, List[Dict]],
        target_class: Optional[int] = None,
        target_query: int = 0,
        normalize: str = "sum1",
    ) -> LRPResult:
        """Führt vollständige LRP-Analyse durch.
        
        Args:
            inputs: Eingabe-Tensor oder Detectron2-Batch
            target_class: Zielklasse für Attribution (optional)
            target_query: Ziel-Query-Index (bei Object Detection)
            normalize: Normalisierungsmethode ("sum1", "sumAbs1", "none")
            
        Returns:
            LRPResult mit Relevanz-Maps
        """
        if not self._prepared:
            self.prepare()
        
        try:
            # Forward Pass
            outputs = self.forward_pass(inputs)
            
            # Start-Relevanz erstellen
            # Für MaskDINO: Decoder-Ausgabe für target_query
            # TODO: Anpassen je nach Modellarchitektur
            decoder_output = self._get_decoder_output(outputs, target_query)
            
            R_start = build_target_relevance(
                layer_output=decoder_output,
                feature_index=target_class if target_class is not None else 0,
                token_reduce="mean",
                target_norm=normalize,
                index_axis="channel" if target_class is not None else "token",
            )
            
            # Backward Pass
            result = self.backward_pass(R_start)
            
            # Normalisierung
            if result.R_input is not None and normalize != "none":
                if normalize == "sum1":
                    result.R_input = result.R_input / (result.R_input.sum() + 1e-12)
                elif normalize == "sumAbs1":
                    result.R_input = result.R_input / (result.R_input.abs().sum() + 1e-12)
            
            return result
            
        finally:
            self.cleanup()
    
    def run_all_queries(
        self,
        inputs: Union[Tensor, List[Dict]],
        num_queries: int = 300,
        target_class: Optional[int] = None,
        normalize: str = "sum1",
    ) -> List[LRPResult]:
        """Führt LRP-Analyse für alle Queries auf einmal durch.
        
        Diese Methode ist effizienter als run() mehrfach aufzurufen,
        da der Forward Pass nur einmal ausgeführt wird.
        
        WICHTIG: Jede Query bekommt ihre eigene Start-Relevanz basierend auf
        ihren tatsächlichen Ausgabewerten. Die Relevanz ist proportional zu den
        absoluten Werten der Decoder-Outputs für diese Query.
        
        Args:
            inputs: Eingabe-Tensor oder Detectron2-Batch
            num_queries: Anzahl der Queries (Standard: 300 für MaskDINO)
            target_class: Zielklasse für Attribution (optional)
            normalize: Normalisierungsmethode ("sum1", "sumAbs1", "none")
            
        Returns:
            Liste von LRPResult, eine pro Query
        """
        import gc
        
        if not self._prepared:
            self.prepare()
        
        results = []
        
        try:
            # Forward Pass nur einmal
            outputs = self.forward_pass(inputs)
            
            # Für jede Query den Backward Pass ausführen
            for query_idx in range(num_queries):
                try:
                    # Start-Relevanz für diese Query erstellen
                    decoder_output = self._get_decoder_output(outputs, query_idx)
                    # decoder_output hat Shape (B, 1, hidden_dim) = (1, 1, 256)
                    
                    # WICHTIG: Erstelle R_start direkt aus den Decoder-Outputs
                    # Die Relevanz soll proportional zu den tatsächlichen Werten sein,
                    # nicht eine gleichmäßige Verteilung!
                    R_start = decoder_output.clone()
                    
                    # Normalisiere R_start auf Summe 1 (wichtig für LRP-Konservierung)
                    if normalize == "sum1":
                        R_start = R_start / (R_start.abs().sum() + 1e-12)
                    elif normalize == "sumAbs1":
                        R_start = R_start / (R_start.abs().sum() + 1e-12)
                    # Bei "none" keine Normalisierung
                    
                    if self.verbose and query_idx < 3:
                        logger.debug(f"Query {query_idx}: R_start sum={R_start.sum().item():.6f}, shape={R_start.shape}")
                    
                    # Backward Pass
                    result = self.backward_pass(R_start)
                    
                    # Nachträgliche Normalisierung des Ergebnisses
                    if result.R_input is not None and normalize != "none":
                        if normalize == "sum1":
                            result.R_input = result.R_input / (result.R_input.sum() + 1e-12)
                        elif normalize == "sumAbs1":
                            result.R_input = result.R_input / (result.R_input.abs().sum() + 1e-12)
                    
                    # Speichere nur R_input, nicht das ganze result (spart Speicher)
                    # Erstelle ein minimales Result mit nur den nötigen Daten
                    minimal_result = LRPResult()
                    if result.R_input is not None:
                        minimal_result.R_input = result.R_input.detach().cpu().clone()
                    results.append(minimal_result)
                    
                    # Speicher freigeben nach jeder Query
                    del result, R_start, decoder_output
                    gc.collect()  # Aggressives GC nach jeder Query
                    
                except Exception as e:
                    logger.warning(f"Fehler bei Query {query_idx}: {e}")
                    results.append(LRPResult())  # Leeres Ergebnis
                    gc.collect()
            
            return results
            
        finally:
            self.cleanup()
            gc.collect()
    
    def _get_decoder_output(
        self,
        outputs: Dict,
        target_query: int,
    ) -> Tensor:
        """Extrahiert Decoder-Ausgabe für LRP-Start.
        
        Für MaskDINO: Hole die Objekt-Query-Features.
        Nutzt zuerst die via Hook gespeicherten Raw-Outputs (vor Post-Processing).
        
        WICHTIG: Gibt die Logits zurück und propagiert durch den Klassifikationskopf,
        um die Decoder-Features (hidden_dim=256) zu erhalten.
        """
        # Priorität 1: Gespeicherte Raw-Outputs vom Decoder-Hook
        logits = None
        if self._decoder_raw_outputs is not None:
            if "pred_logits" in self._decoder_raw_outputs:
                logits = self._decoder_raw_outputs["pred_logits"]
                if self.verbose:
                    logger.debug(f"Verwende Raw-Decoder-Outputs: pred_logits {tuple(logits.shape)}")
        
        # Priorität 2: Versuche verschiedene Output-Strukturen
        if logits is None and isinstance(outputs, dict):
            if "pred_logits" in outputs:
                logits = outputs["pred_logits"]
        
        if logits is None:
            # Fallback: Verwende ersten Tensor
            if isinstance(outputs, Tensor):
                return outputs
            for key, value in outputs.items():
                if isinstance(value, Tensor):
                    return value[:, target_query:target_query+1, :] if value.dim() == 3 else value
            raise ValueError("Konnte keinen Tensor in outputs dict finden.")
        
        # Logits haben Shape (B, Q, num_classes) - wir brauchen Features (B, Q, hidden_dim)
        # Propagiere durch den Klassifikationskopf zurück
        logits_selected = logits[:, target_query:target_query+1, :]  # (1, 1, num_classes)
        
        # Finde den class_embed Layer
        class_embed = None
        for name, module in self.model.named_modules():
            if name.endswith("class_embed") or name == "sem_seg_head.predictor.class_embed":
                class_embed = module
                break
        
        if class_embed is not None and hasattr(class_embed, 'weight'):
            # class_embed ist ein Linear Layer: (hidden_dim -> num_classes)
            # Für LRP zurückpropagieren: (1, 1, num_classes) -> (1, 1, hidden_dim)
            weight = class_embed.weight.detach().clone()  # (num_classes, hidden_dim)
            logits_for_proj = logits_selected.detach().clone()
            
            # Einfache Rückprojektion: R_in = R_out @ W (transponiert)
            # Dies ist eine vereinfachte Version - für echtes LRP müssten wir die
            # Aktivierungen des class_embed Layers während des Forward Pass speichern
            with torch.no_grad():
                R_features = torch.einsum("btc,ch->bth", logits_for_proj, weight)  # (1, 1, hidden_dim)
            
            if self.verbose:
                logger.debug(f"Propagiere durch class_embed: {tuple(logits_selected.shape)} -> {tuple(R_features.shape)}")
            
            return R_features
        else:
            logger.warning("class_embed nicht gefunden - verwende Logits direkt")
            return logits_selected
        
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            if isinstance(outputs[0], dict) and "pred_logits" in outputs[0]:
                return outputs[0]["pred_logits"][:, target_query:target_query+1, :]
            if isinstance(outputs[0], Tensor):
                return outputs[0]
        
        raise ValueError(
            f"Konnte Decoder-Output nicht extrahieren. "
            f"Typ: {type(outputs)}, Inhalt: {outputs if not isinstance(outputs, dict) else list(outputs.keys())}. "
            f"Raw-Decoder-Outputs verfügbar: {self._decoder_raw_outputs is not None}"
        )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Hauptklasse
    "LRPController",
    
    # Re-exports für Abwärtskompatibilität (aus lrp_structs.py)
    "LRPResult",
    "LayerRelevance",
]
