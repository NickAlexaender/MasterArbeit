"""
MaskDINO Model Graph Wrapper für LRP-Analyse.

Dieses Modul linearisiert die komplexe verschachtelte Struktur von Detectron2/MaskDINO
Modellen in einen navigierbaren Graphen. Es erstellt ein gemapptes Dictionary von
Layern und eine geordnete Liste für die Rückwärts-Iteration bei LRP.

Komponenten:
    - LayerNode: Datenklasse für einzelne Layer mit Metadaten und Abhängigkeiten
    - LayerType: Enum für die Kategorisierung von Layern (Backbone, Encoder, Decoder)
    - ModelGraph: Hauptklasse, die das Modell einmalig scannt und cached

Der Graph ermöglicht:
    - Effiziente Iteration durch die Layer in Forward- oder Rückwärts-Reihenfolge
    - Korrektes Routing von Relevanz vom Transformer Decoder zu Pixel Decoder Features
    - Zugriff auf LRP-fähige Module für die Aktivierungsspeicherung

Example:
    >>> from model_graph_wrapper import ModelGraph
    >>> graph = ModelGraph(model, verbose=True)
    >>> # Rückwärts-Iteration für LRP
    >>> for node in graph.backward_order():
    ...     if node.has_lrp_module:
    ...         relevance = propagate_relevance(node.module, relevance)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable, Dict, Iterator, List, Optional, 
    Set, Tuple, Type, Union
)
import re
import warnings

import torch.nn as nn

from .param_patcher import (
    LRPModuleMixin,
    LRP_Linear,
    LRP_LayerNorm,
    LRP_MultiheadAttention,
    LRP_MSDeformAttn,
    get_lrp_modules,
)


# =============================================================================
# Layer-Typen und Kategorisierung
# =============================================================================


class LayerType(Enum):
    """Kategorisierung von Layern in der MaskDINO-Architektur."""
    
    BACKBONE = auto()           # ResNet, Swin, etc.
    PIXEL_DECODER = auto()      # FPN, DeformableDetrEncoder im Pixel-Decoder
    ENCODER = auto()            # Transformer Encoder Layers
    DECODER = auto()            # Transformer Decoder Layers (Objekt-Queries)
    HEAD = auto()               # Klassifikations-/Mask-Heads
    OTHER = auto()              # Sonstige Module
    
    @classmethod
    def from_path(cls, path: str) -> 'LayerType':
        """Bestimmt den LayerType anhand des Modul-Pfads."""
        lpath = path.lower()
        
        # HEAD zuerst prüfen - diese Module sind innerhalb des Decoders,
        # aber gehören zur Ausgabeschicht (nicht zum Decoder-Transformer-Block)
        if any(k in lpath for k in ('class_embed', 'mask_embed', 'bbox_embed', 'ref_point_head')):
            return cls.HEAD
        
        # Decoder vor Encoder prüfen (wegen pixel_decoder.transformer.encoder)
        # Aber nur wenn es NICHT ein HEAD-Modul ist (bereits oben geprüft)
        if '.decoder.' in lpath or 'transformer_decoder' in lpath:
            # Zusätzlich prüfen: Ist es ein Decoder-Layer (layers.X) oder ein anderes Decoder-Modul?
            if '.layers.' in lpath:
                if '.encoder.' not in lpath or lpath.index('.decoder.') > lpath.index('.encoder.'):
                    return cls.DECODER
            elif '.encoder.' not in lpath:
                # Andere Decoder-Module (nicht layers, nicht HEAD) -> auch DECODER
                return cls.DECODER
        
        if any(k in lpath for k in ('backbone', 'res', 'swin', 'convnext')):
            return cls.BACKBONE
        
        if 'pixel_decoder' in lpath:
            if '.encoder.' in lpath:
                return cls.ENCODER
            return cls.PIXEL_DECODER
        
        if any(k in lpath for k in ('encoder', 'transformer.encoder')):
            return cls.ENCODER
        
        if any(k in lpath for k in ('class_embed', 'mask_embed', 'bbox_embed', 'head')):
            return cls.HEAD
        
        return cls.OTHER


# =============================================================================
# Layer-Node Datenstruktur
# =============================================================================


@dataclass
class LayerNode:
    """Repräsentiert einen einzelnen Layer im linearisierten Modell-Graphen.
    
    Attributes:
        name: Vollständiger Pfadname des Moduls (z.B. 'sem_seg_head.pixel_decoder.transformer.encoder.layers.0')
        module: Das PyTorch-Modul selbst
        layer_type: Kategorisierung (BACKBONE, ENCODER, DECODER, etc.)
        order_index: Position in der Vorwärts-Reihenfolge (0-basiert)
        parent_name: Name des übergeordneten Moduls
        is_attention: True wenn Attention-Layer (MHA, MSDeformAttn)
        is_lrp_module: True wenn LRP-fähiges Modul (LRPModuleMixin)
        
        # Für LRP-Routing:
        input_from: Liste der Layer-Namen, von denen Input kommt
        output_to: Liste der Layer-Namen, zu denen Output geht
    """
    name: str
    module: nn.Module
    layer_type: LayerType
    order_index: int
    parent_name: str = ""
    is_attention: bool = False
    is_lrp_module: bool = False
    input_from: List[str] = field(default_factory=list)
    output_to: List[str] = field(default_factory=list)
    
    @property
    def module_class_name(self) -> str:
        """Gibt den Klassennamen des Moduls zurück."""
        return type(self.module).__name__
    
    @property
    def has_activations(self) -> bool:
        """Prüft ob das Modul Aktivierungen gespeichert hat (nur für LRP-Module)."""
        if not self.is_lrp_module:
            return False
        if not isinstance(self.module, LRPModuleMixin):
            return False
        acts = self.module.activations
        return acts is not None and acts.output is not None
    
    def __repr__(self) -> str:
        return (
            f"LayerNode(name='{self.name}', "
            f"type={self.layer_type.name}, "
            f"idx={self.order_index}, "
            f"class={self.module_class_name}, "
            f"lrp={self.is_lrp_module})"
        )


# =============================================================================
# Attention-Typ-Erkennung
# =============================================================================


_ATTENTION_CLASS_NAMES = frozenset({
    'multiheadattention',
    'msdeformattn',
    'selfattention',
    'selfattn',
    'crossattention',
    'lrp_multiheadattention',
    'lrp_msdeformattn',
})


def _is_attention_module(module: nn.Module) -> bool:
    """Prüft ob ein Modul ein Attention-Layer ist."""
    cls_name = type(module).__name__.lower()
    return cls_name in _ATTENTION_CLASS_NAMES


def _is_transformer_block(module: nn.Module) -> bool:
    """Prüft ob ein Modul ein Transformer-Block ist (enthält Attention-Layer als Child)."""
    for child in module.children():
        if _is_attention_module(child):
            return True
        # Auch verschachtelte Module prüfen (z.B. self_attn in TransformerDecoderLayer)
        for subchild in child.children():
            if _is_attention_module(subchild):
                return True
    return False


# =============================================================================
# Model Graph Hauptklasse
# =============================================================================


class ModelGraph:
    """Linearisiert ein MaskDINO-Modell in einen navigierbaren Graphen.
    
    Diese Klasse scannt das Modell einmalig und cached die Struktur.
    Sie bietet effiziente Methoden für:
    - Vorwärts-/Rückwärts-Iteration durch die Layer
    - Filterung nach LayerType
    - Zugriff auf LRP-fähige Module
    
    Attributes:
        model: Das zugrundeliegende PyTorch-Modell
        nodes: Dictionary aller LayerNodes, indexiert nach Namen
        ordered_list: Liste aller LayerNodes in Forward-Reihenfolge
        
    Example:
        >>> graph = ModelGraph(maskdino_model)
        >>> 
        >>> # Alle Decoder-Layer in Rückwärts-Reihenfolge
        >>> for node in graph.backward_order(LayerType.DECODER):
        ...     print(f"Processing {node.name}")
        >>> 
        >>> # Flattened Liste für LRP-Rückpropagation
        >>> lrp_layers = graph.get_lrp_propagation_order()
        >>> for name, module in reversed(lrp_layers):
        ...     # Propagiere Relevanz rückwärts
        ...     pass
    """
    
    def __init__(
        self,
        model: nn.Module,
        include_leaf_modules: bool = False,
        verbose: bool = False,
    ):
        """Initialisiert den ModelGraph durch Scannen des Modells.
        
        Args:
            model: Das zu analysierende PyTorch-Modell
            include_leaf_modules: Wenn True, werden auch Blatt-Module (ohne Children)
                                  in die Layer-Liste aufgenommen
            verbose: Wenn True, werden Informationen während des Scannens ausgegeben
        """
        self.model = model
        self._verbose = verbose
        self._include_leaf = include_leaf_modules
        
        # Gecachte Strukturen
        self.nodes: Dict[str, LayerNode] = {}
        self.ordered_list: List[LayerNode] = []
        
        # Layer-Typ-Gruppen für schnellen Zugriff
        self._by_type: Dict[LayerType, List[LayerNode]] = {t: [] for t in LayerType}
        
        # LRP-Module separat cached
        self._lrp_modules: Dict[str, LayerNode] = {}
        
        # Scannen und cachen
        self._build_graph()
        
        if verbose:
            self._print_summary()
    
    def _build_graph(self) -> None:
        """Scannt das Modell und baut den Graphen auf."""
        order_idx = 0
        seen_ids: Set[int] = set()
        
        # Atomare LRP-Modul-Klassen, die immer aufgenommen werden sollen
        # (auch wenn sie Children haben)
        ATOMIC_LRP_CLASS_NAMES = frozenset({
            'LRP_MSDeformAttn',
            'LRP_MultiheadAttention',
        })
        
        for name, module in self.model.named_modules():
            # Überspringe Root-Modul (leerer Name)
            if not name:
                continue
            
            # Duplikate vermeiden (shared modules)
            mod_id = id(module)
            if mod_id in seen_ids:
                continue
            seen_ids.add(mod_id)
            
            # Bestimme ob es ein relevanter Layer ist
            is_block = _is_transformer_block(module)
            has_children = bool(list(module.children()))
            cls_name = type(module).__name__
            
            # WICHTIG: Atomare LRP-Module immer aufnehmen, unabhängig von Children
            is_atomic_lrp = cls_name in ATOMIC_LRP_CLASS_NAMES
            
            # Filterlogik: Wir wollen Transformer-Blöcke und bestimmte wichtige Layer
            if not is_block and has_children and not self._include_leaf and not is_atomic_lrp:
                # Container-Modul ohne Attention -> überspringe (wird durch Children abgedeckt)
                # AUSNAHME: Backbone-Stufen, Pixel-Decoder-Stufen
                lname = name.lower()
                if not any(k in lname for k in ('layer1', 'layer2', 'layer3', 'layer4', 'stages')):
                    continue
            
            if not has_children and not self._include_leaf:
                # Einzelnes Modul (z.B. Linear, Conv) -> nur wenn LRP-fähig
                if not isinstance(module, LRPModuleMixin):
                    continue
            
            # LayerNode erstellen
            layer_type = LayerType.from_path(name)
            parent_name = '.'.join(name.split('.')[:-1])
            is_attn = _is_attention_module(module)
            is_lrp = isinstance(module, LRPModuleMixin)
            
            node = LayerNode(
                name=name,
                module=module,
                layer_type=layer_type,
                order_index=order_idx,
                parent_name=parent_name,
                is_attention=is_attn,
                is_lrp_module=is_lrp,
            )
            
            self.nodes[name] = node
            self.ordered_list.append(node)
            self._by_type[layer_type].append(node)
            
            if is_lrp:
                self._lrp_modules[name] = node
            
            order_idx += 1
        
        # Abhängigkeiten inferieren (vereinfacht: sequentiell nach Typ)
        self._infer_dependencies()
    
    def _infer_dependencies(self) -> None:
        """Inferiert Input/Output-Abhängigkeiten zwischen Layern."""
        # Innerhalb jedes Typs: sequentielle Abhängigkeit
        for layer_type in LayerType:
            nodes = self._by_type[layer_type]
            for i, node in enumerate(nodes):
                if i > 0:
                    node.input_from.append(nodes[i - 1].name)
                if i < len(nodes) - 1:
                    node.output_to.append(nodes[i + 1].name)
        
        # Typ-übergreifende Abhängigkeiten (MaskDINO-spezifisch)
        # Backbone -> Pixel Decoder
        backbone_nodes = self._by_type[LayerType.BACKBONE]
        pixel_decoder_nodes = self._by_type[LayerType.PIXEL_DECODER]
        if backbone_nodes and pixel_decoder_nodes:
            # Letzte Backbone-Layer speisen ersten Pixel-Decoder
            pixel_decoder_nodes[0].input_from.append(backbone_nodes[-1].name)
        
        # Pixel Decoder -> Encoder
        encoder_nodes = self._by_type[LayerType.ENCODER]
        if pixel_decoder_nodes and encoder_nodes:
            encoder_nodes[0].input_from.append(pixel_decoder_nodes[-1].name)
        
        # Encoder -> Decoder (Memory Connection)
        decoder_nodes = self._by_type[LayerType.DECODER]
        if encoder_nodes and decoder_nodes:
            # Alle Encoder-Layer gehen als Memory zum Decoder
            for dec_node in decoder_nodes:
                if encoder_nodes:
                    dec_node.input_from.append(encoder_nodes[-1].name)
        
        # Pixel Decoder Features -> Decoder (für Mask-Prediction)
        if pixel_decoder_nodes and decoder_nodes:
            for dec_node in decoder_nodes:
                dec_node.input_from.append(pixel_decoder_nodes[-1].name)
    
    def _print_summary(self) -> None:
        """Gibt eine Zusammenfassung des Graphen aus."""
        print(f"\n{'='*60}")
        print("ModelGraph Summary")
        print(f"{'='*60}")
        print(f"Total Layers: {len(self.ordered_list)}")
        print(f"LRP-fähige Module: {len(self._lrp_modules)}")
        print()
        for layer_type in LayerType:
            nodes = self._by_type[layer_type]
            if nodes:
                lrp_count = sum(1 for n in nodes if n.is_lrp_module)
                print(f"{layer_type.name:15} : {len(nodes):3} layers ({lrp_count} LRP-fähig)")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # Iteration Methods
    # =========================================================================
    
    def forward_order(
        self,
        layer_type: Optional[LayerType] = None
    ) -> Iterator[LayerNode]:
        """Iteriert durch Layer in Forward-Reihenfolge.
        
        Args:
            layer_type: Optional - nur Layer dieses Typs zurückgeben
        
        Yields:
            LayerNode-Objekte in Forward-Reihenfolge
        """
        if layer_type is None:
            yield from self.ordered_list
        else:
            yield from self._by_type[layer_type]
    
    def backward_order(
        self,
        layer_type: Optional[LayerType] = None
    ) -> Iterator[LayerNode]:
        """Iteriert durch Layer in Rückwärts-Reihenfolge (für LRP).
        
        Args:
            layer_type: Optional - nur Layer dieses Typs zurückgeben
        
        Yields:
            LayerNode-Objekte in Rückwärts-Reihenfolge
        """
        if layer_type is None:
            yield from reversed(self.ordered_list)
        else:
            yield from reversed(self._by_type[layer_type])
    
    def get_lrp_propagation_order(
        self,
        which_module: str = "all",
    ) -> List[Tuple[str, nn.Module]]:
        """Gibt die geordnete Liste von LRP-fähigen Modulen für Rückpropagation.
        
        Dies ist die Hauptmethode für LRP-Analyse: Sie liefert die Module
        in der korrekten Reihenfolge für die Relevanz-Rückpropagation.
        
        Reihenfolge (rückwärts):
        1. Decoder (von letztem zu erstem Layer) - nur wenn which_module="decoder" oder "all"
        2. Encoder (von letztem zu erstem Layer) - nur wenn which_module="encoder" oder "all"
        3. Pixel Decoder
        4. Backbone
        
        WICHTIG: Kinder von atomaren LRP-Modulen (LRP_MSDeformAttn, LRP_MultiheadAttention)
        werden ausgeschlossen, da das Eltern-Modul die vollständige Propagation übernimmt.
        
        Args:
            which_module: "all", "encoder" oder "decoder"
                - "all": Vollständige Propagation durch alle Module
                - "encoder": Nur Encoder + Pixel Decoder + Backbone (kein Decoder)
                - "decoder": Nur Decoder-Module
        
        Returns:
            Liste von (name, module) Tupeln in LRP-Rückpropagations-Reihenfolge
        """
        result: List[Tuple[str, nn.Module]] = []
        
        # Sammle Namen von atomaren LRP-Modulen (MSDeformAttn, MultiheadAttention)
        # deren Kinder wir überspringen müssen
        atomic_lrp_parents: Set[str] = set()
        for node in self.ordered_list:
            if node.is_lrp_module:
                cls_name = type(node.module).__name__
                if cls_name in ("LRP_MSDeformAttn", "LRP_MultiheadAttention"):
                    atomic_lrp_parents.add(node.name)
        
        def _is_child_of_atomic(name: str) -> bool:
            """Prüft ob ein Layer ein Kind eines atomaren LRP-Moduls ist."""
            for parent in atomic_lrp_parents:
                if name.startswith(parent + "."):
                    return True
            return False
        
        # Reihenfolge basierend auf which_module
        if which_module == "decoder":
            # Nur Decoder
            order = [LayerType.DECODER]
        elif which_module == "encoder":
            # Nur Encoder + Pixel Decoder + Backbone (KEIN Decoder)
            order = [
                LayerType.ENCODER,
                LayerType.PIXEL_DECODER,
                LayerType.BACKBONE,
            ]
        else:
            # Alle Module
            order = [
                LayerType.DECODER,
                LayerType.ENCODER,
                LayerType.PIXEL_DECODER,
                LayerType.BACKBONE,
            ]
        
        for layer_type in order:
            for node in reversed(self._by_type[layer_type]):
                if node.is_lrp_module:
                    # Überspringe Kinder von atomaren LRP-Modulen
                    if _is_child_of_atomic(node.name):
                        continue
                    result.append((node.name, node.module))
        
        return result
    
    def get_flattened_layer_list(self) -> List[Tuple[str, nn.Module]]:
        """Gibt eine flache Liste aller Layer in Forward-Reihenfolge.
        
        Format: [backbone.layer1, ..., pixel_decoder.layerX, transformer.decoder.layer0, ...]
        
        Returns:
            Liste von (name, module) Tupeln
        """
        return [(node.name, node.module) for node in self.ordered_list]
    
    # =========================================================================
    # Access Methods
    # =========================================================================
    
    def __getitem__(self, name: str) -> LayerNode:
        """Zugriff auf einen Layer nach Namen."""
        return self.nodes[name]
    
    def __contains__(self, name: str) -> bool:
        """Prüft ob ein Layer mit diesem Namen existiert."""
        return name in self.nodes
    
    def __len__(self) -> int:
        """Anzahl der Layer im Graphen."""
        return len(self.ordered_list)
    
    def __iter__(self) -> Iterator[LayerNode]:
        """Iteration in Forward-Reihenfolge."""
        return iter(self.ordered_list)
    
    def get_by_type(self, layer_type: LayerType) -> List[LayerNode]:
        """Gibt alle Layer eines bestimmten Typs zurück."""
        return self._by_type[layer_type].copy()
    
    def get_lrp_modules(self) -> Dict[str, LRPModuleMixin]:
        """Gibt alle LRP-fähigen Module zurück.
        
        Delegiert an param_patcher.get_lrp_modules für Konsistenz.
        """
        return get_lrp_modules(self.model)
    
    def get_attention_layers(self) -> List[LayerNode]:
        """Gibt alle Attention-Layer zurück (MHA, MSDeformAttn)."""
        return [node for node in self.ordered_list if node.is_attention]
    
    def find_by_pattern(self, pattern: str) -> List[LayerNode]:
        """Findet Layer deren Namen einem Regex-Muster entsprechen.
        
        Args:
            pattern: Regex-Muster für den Layer-Namen
        
        Returns:
            Liste der passenden LayerNodes
        """
        regex = re.compile(pattern, re.IGNORECASE)
        return [node for node in self.ordered_list if regex.search(node.name)]
    
    # =========================================================================
    # LRP-spezifische Hilfsmethoden
    # =========================================================================
    
    def get_decoder_to_encoder_mapping(self) -> Dict[str, List[str]]:
        """Gibt das Mapping von Decoder-Layern zu ihren Encoder-Inputs zurück.
        
        Dies ist wichtig für Cross-Attention, wo Decoder-Queries auf
        Encoder-Memory zugreifen.
        
        Returns:
            Dictionary: decoder_name -> [encoder_names]
        """
        mapping: Dict[str, List[str]] = {}
        encoder_names = [n.name for n in self._by_type[LayerType.ENCODER]]
        
        for node in self._by_type[LayerType.DECODER]:
            # Alle Decoder-Layer haben potenziell Zugriff auf alle Encoder-Outputs
            # (MaskDINO verwendet Memory von allen Encoder-Layern)
            mapping[node.name] = encoder_names.copy()
        
        return mapping
    
    def get_pixel_decoder_features(self) -> List[LayerNode]:
        """Gibt die Pixel-Decoder-Feature-Layer zurück.
        
        Diese werden sowohl vom Encoder verwendet als auch vom Decoder
        für die Mask-Prediction.
        """
        return self._by_type[LayerType.PIXEL_DECODER].copy()
    
    def trace_relevance_path(
        self,
        start_layer: str,
        target_type: LayerType = LayerType.BACKBONE
    ) -> List[str]:
        """Trackt den Relevanz-Pfad von einem Layer zurück zum Ziel-Typ.
        
        Args:
            start_layer: Name des Start-Layers
            target_type: Ziel-LayerType (Default: BACKBONE)
        
        Returns:
            Liste der Layer-Namen auf dem Pfad
        """
        if start_layer not in self.nodes:
            raise ValueError(f"Layer '{start_layer}' not found in graph")
        
        path = [start_layer]
        current = self.nodes[start_layer]
        
        while current.layer_type != target_type:
            # Nächsten Layer auf dem Rückwärts-Pfad finden
            if not current.input_from:
                break
            
            # Bevorzuge Layer des gleichen oder niedrigeren Typs
            next_name = None
            for inp in current.input_from:
                if inp in self.nodes:
                    inp_node = self.nodes[inp]
                    if inp_node.layer_type.value <= current.layer_type.value:
                        next_name = inp
                        break
            
            if next_name is None:
                next_name = current.input_from[0] if current.input_from else None
            
            if next_name is None or next_name in path:
                break
            
            path.append(next_name)
            current = self.nodes[next_name]
        
        return path


# =============================================================================
# Legacy-Kompatibilitätsfunktionen
# =============================================================================


def list_encoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Legacy-Funktion: Finde Encoder-Layer.
    
    DEPRECATED: Verwende stattdessen ModelGraph.get_by_type(LayerType.ENCODER)
    """
    warnings.warn(
        "list_encoder_like_layers ist deprecated. "
        "Verwende ModelGraph(model).get_by_type(LayerType.ENCODER)",
        DeprecationWarning,
        stacklevel=2
    )
    graph = ModelGraph(model, include_leaf_modules=True)
    return [(n.name, n.module) for n in graph.get_by_type(LayerType.ENCODER)]


def list_decoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Legacy-Funktion: Finde Decoder-Layer.
    
    DEPRECATED: Verwende stattdessen ModelGraph.get_by_type(LayerType.DECODER)
    """
    warnings.warn(
        "list_decoder_like_layers ist deprecated. "
        "Verwende ModelGraph(model).get_by_type(LayerType.DECODER)",
        DeprecationWarning,
        stacklevel=2
    )
    graph = ModelGraph(model, include_leaf_modules=True)
    return [(n.name, n.module) for n in graph.get_by_type(LayerType.DECODER)]


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Hauptklassen
    "ModelGraph",
    "LayerNode",
    "LayerType",
    
    # Legacy (deprecated)
    "list_encoder_like_layers",
    "list_decoder_like_layers",
]
