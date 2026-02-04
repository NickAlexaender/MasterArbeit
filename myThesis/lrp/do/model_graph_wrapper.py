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


# Kategorisierung von Layern in der MaskDINO-Architektur.

class LayerType(Enum):
    
    BACKBONE = auto()           # ResNet, Swin, etc.
    PIXEL_DECODER = auto()      # FPN, DeformableDetrEncoder im Pixel-Decoder
    ENCODER = auto()            # Transformer Encoder Layers
    DECODER = auto()            # Transformer Decoder Layers (Objekt-Queries)
    HEAD = auto()               # Klassifikations-/Mask-Heads
    OTHER = auto()              # Sonstige Module
    
    @classmethod
    def from_path(cls, path: str) -> 'LayerType':
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


# wir repräsentieren einen einzelnen Layer im linearisierten Modell-Graphen.

@dataclass
class LayerNode:
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
        return type(self.module).__name__
    
    @property
    def has_activations(self) -> bool:
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

_ATTENTION_CLASS_NAMES = frozenset({
    'multiheadattention',
    'msdeformattn',
    'selfattention',
    'selfattn',
    'crossattention',
    'lrp_multiheadattention',
    'lrp_msdeformattn',
})

# Prüfen ob Attention-Layer
def _is_attention_module(module: nn.Module) -> bool:
    cls_name = type(module).__name__.lower()
    return cls_name in _ATTENTION_CLASS_NAMES

# Prüfen ob Transformer-Block
def _is_transformer_block(module: nn.Module) -> bool:
    for child in module.children():
        if _is_attention_module(child):
            return True
        # Auch verschachtelte Module prüfen (z.B. self_attn in TransformerDecoderLayer)
        for subchild in child.children():
            if _is_attention_module(subchild):
                return True
    return False

# Wir scannen nun das MaskDINO-Modell, damit wir die Layer in der richtigen Reihenfolge für LRP haben.

class ModelGraph:
    
    def __init__(
        self,
        model: nn.Module,
        include_leaf_modules: bool = False,
        verbose: bool = False,
    ):
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
            
            # Atomare LRP-Module immer aufnehmen, unabhängig von Children
            is_atomic_lrp = cls_name in ATOMIC_LRP_CLASS_NAMES
            
            # Wir wollen Transformer-Blöcke und bestimmte wichtige Layer
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
        
        # Abhängigkeiten inferieren
        self._infer_dependencies()
    
    def _infer_dependencies(self) -> None:
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
    
# Iteriert durch Layer in Forward-Reihenfolge
    def forward_order(
        self,
        layer_type: Optional[LayerType] = None
    ) -> Iterator[LayerNode]:
        if layer_type is None:
            yield from self.ordered_list
        else:
            yield from self._by_type[layer_type]
    
# Iteriert durch Layer in Rückwärts-Reihenfolge
    def backward_order(
        self,
        layer_type: Optional[LayerType] = None
    ) -> Iterator[LayerNode]:
        if layer_type is None:
            yield from reversed(self.ordered_list)
        else:
            yield from reversed(self._by_type[layer_type])
    
    # Gibt die Reihenfolge der LRP-fähigen Module für die Rückpropagation zurück
    
    def get_lrp_propagation_order(
        self,
        which_module: str = "all",
    ) -> List[Tuple[str, nn.Module]]:
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
        return [(node.name, node.module) for node in self.ordered_list]

    def __getitem__(self, name: str) -> LayerNode:
        return self.nodes[name]
    
    def __contains__(self, name: str) -> bool:
        return name in self.nodes
    
    def __len__(self) -> int:
        return len(self.ordered_list)
    
    def __iter__(self) -> Iterator[LayerNode]:
        return iter(self.ordered_list)
    
    def get_by_type(self, layer_type: LayerType) -> List[LayerNode]:
        return self._by_type[layer_type].copy()
    
    def get_lrp_modules(self) -> Dict[str, LRPModuleMixin]:
        return get_lrp_modules(self.model)
    
    def get_attention_layers(self) -> List[LayerNode]:
        return [node for node in self.ordered_list if node.is_attention]
    
    def find_by_pattern(self, pattern: str) -> List[LayerNode]:
        regex = re.compile(pattern, re.IGNORECASE)
        return [node for node in self.ordered_list if regex.search(node.name)]
    
# Mappen von Decoder-Layern zu ihren Encoder-Inputs. Wichtifg für Cross-Attention
    
    def get_decoder_to_encoder_mapping(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        encoder_names = [n.name for n in self._by_type[LayerType.ENCODER]]
        
        for node in self._by_type[LayerType.DECODER]:
            # Alle Decoder-Layer haben potenziell Zugriff auf alle Encoder-Outputs
            # (MaskDINO verwendet Memory von allen Encoder-Layern)
            mapping[node.name] = encoder_names.copy()
        
        return mapping
    
# Geb die Pixel-Decoder-Feature-Layer zurück

    def get_pixel_decoder_features(self) -> List[LayerNode]:
        return self._by_type[LayerType.PIXEL_DECODER].copy()
    
# Tracke Relevanz-Pfad von einem Layer zurück
    def trace_relevance_path(
        self,
        start_layer: str,
        target_type: LayerType = LayerType.BACKBONE
    ) -> List[str]:
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




def list_encoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    warnings.warn(
        "list_encoder_like_layers ist deprecated. "
        "Verwende ModelGraph(model).get_by_type(LayerType.ENCODER)",
        DeprecationWarning,
        stacklevel=2
    )
    graph = ModelGraph(model, include_leaf_modules=True)
    return [(n.name, n.module) for n in graph.get_by_type(LayerType.ENCODER)]


def list_decoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    warnings.warn(
        "list_decoder_like_layers ist deprecated. "
        "Verwende ModelGraph(model).get_by_type(LayerType.DECODER)",
        DeprecationWarning,
        stacklevel=2
    )
    graph = ModelGraph(model, include_leaf_modules=True)
    return [(n.name, n.module) for n in graph.get_by_type(LayerType.DECODER)]


__all__ = [
    "ModelGraph",
    "LayerNode",
    "LayerType",
    "list_encoder_like_layers",
    "list_decoder_like_layers",
]
