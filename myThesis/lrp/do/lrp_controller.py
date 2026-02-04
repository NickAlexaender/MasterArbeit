from __future__ import annotations
import gc
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from .config import (
    ATTN_QK_SHARE,
    LN_RULE,
    RESIDUAL_SPLIT,
    SIGN_PRESERVING,
)
from .model_graph_wrapper import LayerNode, ModelGraph
from .param_patcher import (
    LRPModuleMixin,
    clear_all_activations,
    prepare_model_for_lrp,
    set_lrp_mode,
)
from .tensor_ops import build_target_relevance
from .lrp_structs import LRPResult, LayerRelevance
from .lrp_propagators import propagate_layer, propagate_residual

logger = logging.getLogger("lrp.controller")

# Hier entsteht der zentrale LRP-Controller - verwaltet den gesamten LRP-Workflow
# Modell-Vorbereitung
# Forward Pass mit Aktivierungserfassung
# Backward Pass mit Relevanz-Propagation
# Aggregation

class LRPController:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        eps: float = 1e-6,
        verbose: bool = False,
    ):
        self.model = model
        self.device = device
        self.eps = eps
        self.verbose = verbose
        self._prepared = False
        self._graph: Optional[ModelGraph] = None
        
        # Speicher für Raw-Decoder-Outputs
        self._decoder_raw_outputs: Optional[Dict[str, Tensor]] = None
        self._decoder_hook_handle = None
        
        # Konfiguration
        self.residual_split_mode = RESIDUAL_SPLIT
        self.ln_rule = LN_RULE
        self.attn_qk_share = ATTN_QK_SHARE
        self.sign_preserving = SIGN_PRESERVING
        if verbose:
            logger.setLevel(logging.INFO)
    
    # Hier Hook auf dem Decoder zum Anfngen der Raw-Outputs
    
    def _register_decoder_hook(self) -> None:
        
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
            # MaskDINO gibt (outputs_dict, mask_dict) zurück
            if isinstance(output, tuple) and len(output) >= 1:
                raw_output = output[0]
                if isinstance(raw_output, dict) and "pred_logits" in raw_output:
                    # OPTIMIZATION: Kein clone() - Tensoren werden nicht ver\u00e4ndert
                    self._decoder_raw_outputs = {
                        k: v if isinstance(v, Tensor) else v
                        for k, v in raw_output.items()
                    }
                    if self.verbose:
                        logger.debug(f"Decoder-Raw-Outputs gespeichert: {list(self._decoder_raw_outputs.keys())}")
        
        self._decoder_hook_handle = decoder_module.register_forward_hook(hook_fn)
        
# Wir bereiten das Modell auf LRP vor.
# Heißt wir ersetzen PyTorch-Standard_module durch LRP-fähige Versionen -> können dann aktivierungen intern speichern

    def prepare(
        self,
        swap_linear: bool = False,
        swap_layernorm: bool = True,
        swap_mha: bool = True,
        swap_msdeform: bool = True,
    ) -> Dict[str, int]:
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
        return stats
    
    
    @property
    def graph(self) -> ModelGraph:
        if self._graph is None:
            self._graph = ModelGraph(
                self.model,
                include_leaf_modules=True,
                verbose=self.verbose,
            )
        return self._graph
    
# Erst muss der Forward Pass durchgeführt werden, um die Aktivierungen zu speichern

    def forward_pass(
        self,
        inputs: Union[Tensor, List[Dict]],
    ) -> Dict[str, any]:
        # LRP-Modus aktivieren
        set_lrp_mode(self.model, enabled=True)
        
        # Raw-Outputs zurücksetzen
        self._decoder_raw_outputs = None
        
        try:
            with torch.no_grad():
                outputs = self.model(inputs)
        finally:
            pass
        
        return outputs
    
# Anschließend führen wir den Backward Pass durch, um die Relevanz zu propagieren - gemäß layer spezifischen regeln
# Relevanz wird auf dem OUTPUT des Start-Layers initialisiert
# Wir propagieren durch ALLE Layer

    def backward_pass(
        self,
        R_start: Tensor,
        target_layer: Optional[str] = None,
        which_module: str = "all",
        stop_after_cross_attn: bool = True,
        clear_activations: bool = True,
        start_layer_index: Optional[int] = None,
    ) -> LRPResult:
        result = LRPResult()
        R_current = R_start
        propagation_order = self.graph.get_lrp_propagation_order(which_module=which_module)
        if start_layer_index is not None and self.verbose:
            logger.debug(f"Layer-zu-Layer LRP: Start-Layer-Index {start_layer_index}, "
                        f"propagiere durch alle {len(propagation_order)} Layer")
        
        for layer_name, module in propagation_order:
            # Progress-Logging
            logger.debug(f"Propagiere Layer: {layer_name} (R_current.shape={R_current.shape})")
            
            if stop_after_cross_attn and which_module == "decoder":
                if "cross_attn" in layer_name and "decoder" in layer_name:
                    if self.verbose:
                        logger.debug(f"Überspringe cross_attn {layer_name} - bleibe im Query-Space (300)")
                    continue
            
            # Stoppe bei Ziel-Layer falls angegeben
            if target_layer and layer_name == target_layer:
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
                
                # Bei run_all_queries brauchen wir die Aktivierungen für alle Queries
                if clear_activations and hasattr(module, 'activations') and module.activations is not None:
                    module.activations.clear()
                
                # Konservierungsprüfung
                conservation_error = self._check_conservation(R_current, R_prev)
                result.conservation_errors.append(conservation_error)
                
                R_current = R_prev
                del R_prev
                
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
    
    # Propagiert Relevanz durch einen einzelnen Layer
    
    def _propagate_layer(
        self,
        module: nn.Module,
        node: LayerNode,
        R_out: Tensor,
    ) -> Tensor:
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
    
# Propagiert Relevanz durch eine Residual-Verbindung y = x + F(x).

    def propagate_with_residual(
        self,
        x: Tensor,
        Fx: Tensor,
        R_y: Tensor,
    ) -> tuple:
        return propagate_residual(
            x=x,
            Fx=Fx,
            R_y=R_y,
            mode=self.residual_split_mode,
            eps=self.eps,
        )
    
    #Konservierungsprüfung
    
    def _check_conservation(self, R_out: Tensor, R_in: Optional[Tensor]) -> float:
        # Robuster Check für None
        if R_in is None:
            logger.warning("R_in ist None - überspringe Konservierungsprüfung")
            return 0.0
        
        if R_out is None:
            logger.warning("R_out ist None - überspringe Konservierungsprüfung")
            return 0.0
        
        sum_out = R_out.sum().item()
        sum_in = R_in.sum().item()
        
        if abs(sum_out) < 1e-12:
            return 0.0
        
        return (sum_in - sum_out) / abs(sum_out)
    
    # Bereinigen des Modells nach LRP-Durchlauf
    
    def cleanup(self, run_gc: bool = False):
        set_lrp_mode(self.model, enabled=False)
        clear_all_activations(self.model)
        
        if run_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Jetzt können wir die gesamte LRP-Analyse in einer Methode zusammenfassen

    def run(
        self,
        inputs: Union[Tensor, List[Dict]],
        target_class: Optional[int] = None,
        target_query: int = 0,
        normalize: str = "sum1",
        which_module: str = "all",
        target_feature: Optional[int] = None,
        target_layer_index: Optional[int] = None,
        target_layer_name: Optional[str] = None,
    ) -> LRPResult:
        if not self._prepared:
            self.prepare()
        
        try:
            # Forward Pass
            outputs = self.forward_pass(inputs)
            
            # Start-Relevanz erstellen basierend auf which_module
            if which_module == "encoder":
                # Für Encoder-LRP: Verwende Output des spezifizierten Encoder-Layers
                R_start, prop_start_index = self._get_encoder_start_relevance(
                    feature_index=target_feature if target_feature is not None else 0,
                    normalize=normalize,
                    target_layer_index=target_layer_index,
                    target_layer_name=target_layer_name,
                )
            elif which_module == "decoder" and target_layer_name is not None:
                # Für Decoder Layer-zu-Layer LRP: Verwende Output des spezifizierten Decoder-Layers
                R_start, prop_start_index = self._get_decoder_start_relevance(
                    query_index=target_query,
                    normalize=normalize,
                    target_layer_name=target_layer_name,
                )
            else:
                # Für Decoder-LRP oder "all": Verwende Decoder-Output (alte Logik)
                decoder_output = self._get_decoder_output(outputs, target_query)
                prop_start_index = None  # Nicht verwendet für Decoder
                
                R_start = build_target_relevance(
                    layer_output=decoder_output,
                    feature_index=target_class if target_class is not None else 0,
                    token_reduce="mean",
                    target_norm=normalize,
                    index_axis="channel" if target_class is not None else "token",
                )
            
            # Backward Pass - mit which_module Parameter
            # Bei Layer-zu-Layer LRP starten wir am prop_start_index
            # Die Propagation läuft durch alle Layer darunter
            result = self.backward_pass(
                R_start, 
                which_module=which_module,
                start_layer_index=prop_start_index if (which_module == "encoder" or (which_module == "decoder" and target_layer_name)) else None,
            )
            
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
        which_module: str = "decoder",
        gc_interval: int = 300,
    ) -> List[LRPResult]:
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
                    
                    # WICHTIG: Für Decoder-LRP müssen wir R_start auf alle 300 Queries expandieren,
                    # da die Self-Attention alle Queries gleichzeitig verarbeitet.
                    # Nur die Ziel-Query erhält Relevanz, alle anderen = 0.
                    num_total_queries = 300  # MaskDINO verwendet immer 300 Queries
                    B, _, C = decoder_output.shape
                    
                    # Die Decoder-Aktivierungen sind im Format (T, B, C) = (300, 1, 256)
                    # (siehe debug_shapes.py Analyse: norm1, norm2, norm3 haben diese Shape)
                    # Daher muss R_start auch dieses Format haben!
                    R_start_full = torch.zeros((num_total_queries, B, C), 
                                               device=decoder_output.device, 
                                               dtype=decoder_output.dtype)
                    
                    # Setze Relevanz nur für die Ziel-Query an Position [query_idx, 0, :]
                    # decoder_output hat Shape (B, 1, C) = (1, 1, 256) -> squeeze und setze
                    R_start_full[query_idx, 0, :] = decoder_output.squeeze()
                    
                    # Normalisiere R_start auf Summe 1 (wichtig für LRP-Konservierung)
                    if normalize == "sum1":
                        R_start_full = R_start_full / (R_start_full.abs().sum() + 1e-12)
                    elif normalize == "sumAbs1":
                        R_start_full = R_start_full / (R_start_full.abs().sum() + 1e-12)
                    # Bei "none" keine Normalisierung
                    
                    if self.verbose and query_idx < 3:
                        logger.debug(f"Query {query_idx}: R_start sum={R_start_full.sum().item():.6f}, shape={R_start_full.shape}")
                    
                    # Backward Pass - mit which_module Parameter für Decoder-only Propagation
                    # WICHTIG: clear_activations=False damit Aktivierungen für alle Queries erhalten bleiben
                    result = self.backward_pass(
                        R_start_full, 
                        which_module=which_module,
                        clear_activations=False,  # Aktivierungen für folgende Queries erhalten
                    )
                    
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
                    
                    # Speicher freigeben - nur alle gc_interval Queries
                    del result, R_start_full, decoder_output
                    if (query_idx + 1) % gc_interval == 0:
                        gc.collect()
                    
                except Exception as e:
                    logger.warning(f"Fehler bei Query {query_idx}: {e}")
                    results.append(LRPResult())  # Leeres Ergebnis
            
            return results
            
        finally:
            self.cleanup()
            gc.collect()
            
    # Erstellt Start-Relevanz für Encoder-LRP basierend auf einem spezifischen Encoder-Layer-Output
    
    def _get_encoder_start_relevance(
        self,
        feature_index: int = 0,
        normalize: str = "sum1",
        target_layer_index: Optional[int] = None,
        target_layer_name: Optional[str] = None,
    ) -> Tuple[Tensor, int]:
        # Hole alle Encoder-Layer in Propagationsreihenfolge (rückwärts = letzter zuerst)
        encoder_layers = self.graph.get_lrp_propagation_order(which_module="encoder")
        if not encoder_layers:
            raise RuntimeError("Keine Encoder-Layer gefunden")
        
        # Wähle den richtigen Layer
        prop_start_index = 0
        target_module = None
        target_name = None
        
        if target_layer_name is not None:
            # Suche Layer nach Name
            for idx, (name, module) in enumerate(encoder_layers):
                if name == target_layer_name or target_layer_name in name:
                    target_name = name
                    target_module = module
                    prop_start_index = idx
                    break
            if target_module is None:
                raise ValueError(
                    f"Layer '{target_layer_name}' nicht in Propagationsreihenfolge gefunden. "
                    f"Verfügbare Layer: {[n for n, _ in encoder_layers[:10]]}..."
                )
        elif target_layer_index is not None:
            if target_layer_index < 0 or target_layer_index >= len(encoder_layers):
                raise IndexError(
                    f"target_layer_index {target_layer_index} außerhalb [0, {len(encoder_layers)-1}]"
                )
            target_name, target_module = encoder_layers[target_layer_index]
            prop_start_index = target_layer_index
        else:
            # letzter Encoder-Layer
            target_name, target_module = encoder_layers[0]
            prop_start_index = 0
        
        if self.verbose:
            logger.debug(f"Verwende Output von {target_name} als Encoder-Start (Prop-Index: {prop_start_index})")
        
        # Hole die gespeicherten Aktivierungen
        if not hasattr(target_module, 'activations') or target_module.activations is None:
            raise RuntimeError(f"Keine Aktivierungen für {target_name} gespeichert")
        
        activations = target_module.activations
        if activations.output is None:
            raise RuntimeError(f"Kein Output für {target_name} gespeichert")
        
        encoder_output = activations.output
        
        if self.verbose:
            logger.debug(f"Encoder-Output Shape: {encoder_output.shape}")
        
        # Behalte das ursprüngliche Format (T, B, C) für die Propagation bei.
        R_start = torch.zeros_like(encoder_output)
        if encoder_output.dim() == 3:
            if encoder_output.shape[1] == 1:
                if feature_index >= encoder_output.shape[2]:
                     raise IndexError(f"feature_index {feature_index} out of bounds for dimension 2 with size {encoder_output.shape[2]}")
                R_start[:, :, feature_index] = encoder_output[:, :, feature_index].abs()
            else:
                if feature_index >= encoder_output.shape[2]:
                     raise IndexError(f"feature_index {feature_index} out of bounds for dimension 2 with size {encoder_output.shape[2]}")
                R_start[:, :, feature_index] = encoder_output[:, :, feature_index].abs()
        elif encoder_output.dim() == 2:
            if feature_index >= encoder_output.shape[1]:
                 raise IndexError(f"feature_index {feature_index} out of bounds for dimension 1 with size {encoder_output.shape[1]}")
            R_start[:, feature_index] = encoder_output[:, feature_index].abs()
        if normalize == "sum1":
            total = R_start.sum()
            if total > 1e-12:
                R_start = R_start / total
        elif normalize == "sumAbs1":
            total = R_start.abs().sum()
            if total > 1e-12:
                R_start = R_start / total
        
        if self.verbose:
            logger.debug(f"Encoder R_start Shape: {R_start.shape}, Sum: {R_start.sum().item():.6f}")
            if R_start.dim() == 3:
                channel_sums = R_start.sum(dim=(0, 1))
                non_zero = (channel_sums > 1e-12).sum().item()
                logger.debug(f"Anzahl Kanäle mit Relevanz: {non_zero} (sollte 1 sein für feature_index={feature_index})")
        
        return R_start, prop_start_index
    
    # Nun erstellen wir Start-Relevanz für Decoder Layer-zu-Layer LRP
    
    def _get_decoder_start_relevance(
        self,
        query_index: int = 0,
        normalize: str = "sum1",
        target_layer_name: Optional[str] = None,
    ) -> Tuple[Tensor, int]:

        # Hole alle Decoder-Layer in Propagationsreihenfolge
        decoder_layers = self.graph.get_lrp_propagation_order(which_module="decoder")
        if not decoder_layers:
            raise RuntimeError("Keine Decoder-Layer gefunden")
        
        num_queries_expected = 300
        
        # Wähle den richtigen Layer
        prop_start_index = 0
        target_module = None
        target_name = None
        
        if target_layer_name is not None:
            for idx, (name, module) in enumerate(decoder_layers):
                if name == target_layer_name or target_layer_name in name:
                    target_name = name
                    target_module = module
                    prop_start_index = idx
                    break
            if target_module is None:
                raise ValueError(
                    f"Layer '{target_layer_name}' nicht in Decoder-Propagationsreihenfolge gefunden. "
                    f"Verfügbare Layer: {[n for n, _ in decoder_layers[:10]]}..."
                )
        else:
            # letzter Decoder-Layer
            target_name, target_module = decoder_layers[0]
            prop_start_index = 0
        
        if self.verbose:
            logger.debug(f"Verwende Output von {target_name} als Decoder-Start (Prop-Index: {prop_start_index})")
        
        decoder_output = None
        actual_layer_name = target_name
        actual_prop_index = prop_start_index

        layers_to_try = [(target_name, target_module, prop_start_index)]

        target_block = '.'.join(target_layer_name.split('.')[:-1]) if target_layer_name else ""
        for idx, (name, module) in enumerate(decoder_layers):
            if name != target_name and target_block and target_block in name:
                layers_to_try.append((name, module, idx))
        
        for try_name, try_module, try_idx in layers_to_try:
            if not hasattr(try_module, 'activations') or try_module.activations is None:
                continue
            
            activations = try_module.activations
            if activations.output is None:
                continue
            output = activations.output
            
            # Prüfe ob die Shape passt
            if output.dim() == 3:
                if output.shape[0] == num_queries_expected:
                    # Format (Q, B, C) - perfekt!
                    decoder_output = output
                    actual_layer_name = try_name
                    actual_prop_index = try_idx
                    break
                elif output.shape[1] == num_queries_expected:
                    # Format (B, Q, C) - transponiere (erzeugt neuen Tensor)
                    decoder_output = output.permute(1, 0, 2)
                    actual_layer_name = try_name
                    actual_prop_index = try_idx
                    break
            
            # Versuche auch die Input-Aktivierung
            if activations.input is not None:
                inp = activations.input
                if inp.dim() == 3:
                    if inp.shape[0] == num_queries_expected:
                        decoder_output = inp
                        actual_layer_name = try_name
                        actual_prop_index = try_idx
                        if self.verbose:
                            logger.debug(f"Verwende Input statt Output von {try_name}")
                        break
                    elif inp.shape[1] == num_queries_expected:
                        # permute() erzeugt neuen Tensor
                        decoder_output = inp.permute(1, 0, 2)
                        actual_layer_name = try_name
                        actual_prop_index = try_idx
                        break
        
        if decoder_output is None:
            # Suche JEDEN Decoder-Layer mit passender Shape
            for idx, (name, module) in enumerate(decoder_layers):
                if not hasattr(module, 'activations') or module.activations is None:
                    continue
                activations = module.activations
                for act_name, act in [('output', activations.output), ('input', activations.input)]:
                    if act is None:
                        continue
                    if act.dim() == 3 and act.shape[0] == num_queries_expected:
                        decoder_output = act
                        actual_layer_name = name
                        actual_prop_index = idx
                        logger.warning(f"Fallback: Verwende {act_name} von {name} (Shape: {act.shape})")
                        break
                    elif act.dim() == 3 and act.shape[1] == num_queries_expected:
                        # permute() erzeugt neuen Tensor
                        decoder_output = act.permute(1, 0, 2)
                        actual_layer_name = name
                        actual_prop_index = idx
                        logger.warning(f"Fallback: Verwende transponiertes {act_name} von {name}")
                        break
                if decoder_output is not None:
                    break
        
        if decoder_output is None:
            # Debug: Zeige alle verfügbaren Shapes
            shapes_info = []
            for name, module in decoder_layers[:10]:
                if hasattr(module, 'activations') and module.activations is not None:
                    acts = module.activations
                    out_shape = tuple(acts.output.shape) if acts.output is not None else None
                    inp_shape = tuple(acts.input.shape) if acts.input is not None else None
                    shapes_info.append(f"{name}: out={out_shape}, in={inp_shape}")
            raise RuntimeError(
                f"Kein Decoder-Layer mit Query-Dimension {num_queries_expected} gefunden.\n"
                f"Verfügbare Layer und Shapes:\n" + "\n".join(shapes_info)
            )
        
        if self.verbose:
            logger.debug(f"Decoder-Output von {actual_layer_name}, Shape: {decoder_output.shape}")
        
        # Nur die angegebene Query erhält Relevanz, alle anderen = 0
        R_start = torch.zeros_like(decoder_output)
        
        # Decoder-Output hat Shape (Q, B, C) = (300, 1, 256)
        if decoder_output.dim() == 3:
            # Validiere query_index
            num_queries = decoder_output.shape[0]
            if query_index < 0 or query_index >= num_queries:
                raise IndexError(f"query_index {query_index} außerhalb [0, {num_queries-1}]")
            
            # Alle Dimensionen dieser Query erhalten die Aktivierungswerte
            R_start[query_index, :, :] = decoder_output[query_index, :, :].abs()
        elif decoder_output.dim() == 2:
            # (Q, C) -> behandle wie (Q, 1, C)
            R_start[query_index, :] = decoder_output[query_index, :].abs()
        
        # Normalisieren
        if normalize == "sum1":
            total = R_start.sum()
            if total > 1e-12:
                R_start = R_start / total
        elif normalize == "sumAbs1":
            total = R_start.abs().sum()
            if total > 1e-12:
                R_start = R_start / total
        
        if self.verbose:
            logger.debug(f"Decoder R_start Shape: {R_start.shape}, Sum: {R_start.sum().item():.6f}")
            # Debug: Zeige welche Queries Relevanz haben
            if R_start.dim() == 3:
                query_sums = R_start.sum(dim=(1, 2))
                non_zero = (query_sums > 1e-12).sum().item()
                logger.debug(f"Anzahl Queries mit Relevanz: {non_zero} (sollte 1 sein für query_index={query_index})")
        
        return R_start, actual_prop_index
    
    # Extrahiert die Decoder-Ausgabe für LRP-Start. Nutzt zuerst die via Hook gespeicherten Raw-Outputs
    
    def _get_decoder_output(
        self,
        outputs: Dict,
        target_query: int,
    ) -> Tensor:
        # Gespeicherte Raw-Outputs vom Decoder-Hook
        logits = None
        if self._decoder_raw_outputs is not None:
            if "pred_logits" in self._decoder_raw_outputs:
                logits = self._decoder_raw_outputs["pred_logits"]
                if self.verbose:
                    logger.debug(f"Verwende Raw-Decoder-Outputs: pred_logits {tuple(logits.shape)}")
        
        # Versuche verschiedene Output-Strukturen
        if logits is None and isinstance(outputs, dict):
            if "pred_logits" in outputs:
                logits = outputs["pred_logits"]
        
        if logits is None:
            # Verwende ersten Tensor
            if isinstance(outputs, Tensor):
                return outputs
            for key, value in outputs.items():
                if isinstance(value, Tensor):
                    return value[:, target_query:target_query+1, :] if value.dim() == 3 else value
            raise ValueError("Konnte keinen Tensor in outputs dict finden.")
        logits_selected = logits[:, target_query:target_query+1, :]  # (1, 1, num_classes)
        
        # Finde den class_embed Layer
        class_embed = None
        for name, module in self.model.named_modules():
            if name.endswith("class_embed") or name == "sem_seg_head.predictor.class_embed":
                class_embed = module
                break
        
        if class_embed is not None and hasattr(class_embed, 'weight'):
            weight = class_embed.weight.detach().clone()  # (num_classes, hidden_dim)
            logits_for_proj = logits_selected.detach().clone()
            # Aktivierungen des Layers während des Forward Pass speichern
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

__all__ = [
    "LRPController",
    "LRPResult",
    "LayerRelevance",
]
