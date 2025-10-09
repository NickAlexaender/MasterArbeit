from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
import cv2
import os
import csv
import warnings
warnings.filterwarnings("ignore", message=".*torch\\.meshgrid.*indexing.*", category=UserWarning)


def register_decoder_hooks(model: torch.nn.Module) -> Tuple[Dict[str, List[torch.Tensor]], List]:
    """
    Registriert Forward-Hooks nur auf echten Transformer-Decoder-Layern für Network Dissection.
    Sammelt Hidden-States pro Query je Layer.
    """
    hidden_states_per_layer = {}
    hook_handles = []
    
    def create_hook(layer_name: str):
        def hook_fn(module, input, output):
            if layer_name not in hidden_states_per_layer:
                hidden_states_per_layer[layer_name] = []
            
            # Speichere Hidden-States (für Network Dissection benötigt)
            hidden_states = output[0] if isinstance(output, tuple) else output
            hidden_states_per_layer[layer_name].append(hidden_states.detach().clone())
            
        return hook_fn
    
    # Erkenne nur echte Decoder-Layer (nicht Encoder!)
    decoder_count = 0
    for name, module in model.named_modules():
        # Spezifisch für MaskDINO/DETR: Nur predictor.decoder.layers
        if 'predictor.decoder.layers.' in name and name.count('.') == 4:  # Hauptlayer-Ebene
            # Nur die Hauptlayer registrieren, nicht Sub-Module
            if name.endswith('.layers.0') or name.endswith('.layers.1') or name.endswith('.layers.2'):
                handle = module.register_forward_hook(create_hook(name))
                hook_handles.append(handle)
                decoder_count += 1
                print(f"Hook registriert: {name}")
    
    print(f"✅ {decoder_count} Decoder-Layer gefunden und registriert.")
    return hidden_states_per_layer, hook_handles


def extract_pixel_embedding_map(model: torch.nn.Module, batched_input: Dict[str, Any], 
                               image_id: str, output_dir: str) -> Optional[np.ndarray]:
    """
    Extrahiert Pixel-Embedding-Map für Network Dissection mit korrektem MaskDINO-Input.
    
    Args:
        image_id: Eindeutige ID für das Bild (z.B. "image_1")
    """
    if batched_input is None:
        print(f"⚠️  Kein Input für Pixel-Embedding-Extraktion bei Bild {image_id}")
        return None
    
    encoder_features = {}
    hook_handles = []
    
    def encoder_hook(name):
        def hook_fn(module, input, output):
            features = output[0] if isinstance(output, tuple) else output
            if len(features.shape) == 4:  # Nur räumliche Features [B, C, H, W]
                encoder_features[name] = features.detach().clone()
        return hook_fn
    
    # Registriere Hooks für Encoder/Backbone (spezifisch für MaskDINO)
    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in [
            'backbone.res', 'backbone.layer', 'input_proj', 'pixel_decoder.input_proj'
        ]):
            # Filtere nur die Hauptlayer, nicht alle Sub-Module
            if any(end in name for end in ['.res2', '.res3', '.res4', '.res5', 'input_proj.0', 'input_proj.1', 'input_proj.2']):
                handle = module.register_forward_hook(encoder_hook(name))
                hook_handles.append(handle)
                print(f"🔗 Encoder-Hook registriert: {name}")
    
    try:
        # Forward-Pass mit korrektem MaskDINO-Input-Format
        model.eval()
        with torch.no_grad():
            _ = model([batched_input])  # Liste von batched_inputs
        
        # Cleanup Hooks
        for handle in hook_handles:
            handle.remove()
        
        # Finde beste Pixel-Embedding-Map
        pixel_embedding_map = None
        best_layer_name = None
        
        # Priorität für aussagekräftige Features
        priority_patterns = ['input_proj', 'res5', 'res4', 'res3', 'res2']
        
        for pattern in priority_patterns:
            for layer_name, features in encoder_features.items():
                if pattern in layer_name and len(features.shape) == 4:
                    pixel_embedding_map = features[0].cpu().numpy()  # [C, H, W]
                    best_layer_name = layer_name
                    print(f"✅ Pixel-Embedding gewählt: {layer_name} - Shape: {pixel_embedding_map.shape}")
                    break
            if pixel_embedding_map is not None:
                break
        
        # Fallback: Nimm erste verfügbare Feature-Map
        if pixel_embedding_map is None and encoder_features:
            layer_name, features = next(iter(encoder_features.items()))
            pixel_embedding_map = features[0].cpu().numpy()
            best_layer_name = layer_name
            print(f"🔄 Fallback Pixel-Embedding: {layer_name} - Shape: {pixel_embedding_map.shape}")
        
        if pixel_embedding_map is not None:
            # Speichere für Network Dissection
            os.makedirs(os.path.join(output_dir, "pixel_embeddings"), exist_ok=True)
            
            # Speichere als NPY mit image_id
            npy_path = os.path.join(output_dir, "pixel_embeddings", f"pixel_embed_{image_id}.npy")
            np.save(npy_path, pixel_embedding_map)
            
            # Speichere Metadaten
            metadata = {
                "image_id": image_id,  # Neue eindeutige ID
                "embedding_id": f"embed_{image_id}",  # Embedding-ID für spätere Zuordnung
                "layer_name": best_layer_name,
                "shape": list(pixel_embedding_map.shape),
                "channels": int(pixel_embedding_map.shape[0]),
                "height": int(pixel_embedding_map.shape[1]),
                "width": int(pixel_embedding_map.shape[2]),
                "npy_file": f"pixel_embed_{image_id}.npy",
                # Zusätzliche Felder für exakte Skalierung
                "embed_h": int(pixel_embedding_map.shape[1]),
                "embed_w": int(pixel_embedding_map.shape[2]),
                "input_h": 800,  # MaskDINO Standard
                "input_w": 800,
                "stride": 32
            }
            
            import json
            metadata_path = os.path.join(output_dir, "pixel_embeddings", f"metadata_{image_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Pixel-Embedding gespeichert: {os.path.basename(npy_path)}")
            print(f"   📊 Shape: {pixel_embedding_map.shape} ({pixel_embedding_map.shape[0]} Kanäle)")
            print(f"   📋 Metadaten: {os.path.basename(metadata_path)}")
            
        else:
            print(f"⚠️  Keine geeignete Pixel-Embedding-Map für Bild {image_id} gefunden.")
            if encoder_features:
                print("Verfügbare Features:")
                for name, features in encoder_features.items():
                    print(f"  - {name}: {features.shape}")
            else:
                print("Keine Encoder-Features gefunden - prüfen Sie die Hook-Registrierung.")
        
        return pixel_embedding_map
        
    except Exception as e:
        # Cleanup Hooks bei Fehler
        for handle in hook_handles:
            handle.remove()
        print(f"❌ Fehler bei Pixel-Embedding-Extraktion für Bild {image_id}: {str(e)}")
        return None


def save_queries_to_csv(hidden_states_per_layer: Dict[str, List[torch.Tensor]], 
                       num_queries: int, image_id: str, output_dir: str):
    """
    Speichert Query-Embeddings pro Layer in organisierter Struktur.
    Format: layer0/Query.csv, layer1/Query.csv, etc. mit allen Bildern in einer Datei.
    
    Args:
        image_id: Eindeutige ID für das Bild (z.B. "image_1")
    """
    
    for layer_name, hidden_states_list in hidden_states_per_layer.items():
        if not hidden_states_list:
            continue
            
        # Nimm letzten Hidden-State
        hidden_states = hidden_states_list[-1]
        
        print(f"Layer {layer_name}: Shape = {hidden_states.shape}")
        
        # Für MaskDINO Format: [300, 1, 256] -> [num_queries, batch_size, hidden_dim]
        if len(hidden_states.shape) == 3:
            actual_num_queries, batch_size, hidden_dim = hidden_states.shape
            
            if batch_size == 1:
                # Korrekte Extraktion: Alle 300 Queries, erste (und einzige) Batch-Dimension
                queries = hidden_states[:, 0, :].cpu().numpy()  # [300, 256]
                print(f"✅ Extrahiert: {actual_num_queries} Queries mit {hidden_dim} Dimensionen")
            else:
                print(f"⚠️  Unerwartete Batch-Größe: {batch_size}")
                continue
                
        elif len(hidden_states.shape) == 2:
            # Fallback für 2D-Tensoren
            queries = hidden_states.cpu().numpy()
            print(f"✅ 2D-Tensor extrahiert: Shape {queries.shape}")
        else:
            print(f"⚠️  Unerwartete Tensor-Dimensionen: {hidden_states.shape}")
            continue
        
        # Bestimme Layer-Nummer aus Layer-Name (z.B. "layers.0" -> "layer0")
        layer_number = "unknown"
        if "layers.0" in layer_name:
            layer_number = "layer0"
        elif "layers.1" in layer_name:
            layer_number = "layer1"
        elif "layers.2" in layer_name:
            layer_number = "layer2"
        else:
            # Fallback: Extrahiere Zahl aus Layer-Name
            import re
            match = re.search(r'layers[._](\d+)', layer_name)
            if match:
                layer_number = f"layer{match.group(1)}"
            else:
                layer_number = f"layer_{layer_name.split('.')[-1]}"
        
        # Erstelle Layer-spezifischen Ordner
        layer_dir = os.path.join(output_dir, layer_number)
        os.makedirs(layer_dir, exist_ok=True)
        
        # CSV-Pfad: layerX/Query.csv
        csv_path = os.path.join(layer_dir, "Query.csv")
        
        num_queries_actual, hidden_dim = queries.shape
        
        # Prüfe ob Datei bereits existiert (für weitere Bilder)
        file_exists = os.path.exists(csv_path)
        
        # Bestimme Modus: append für weitere Bilder, write für erstes Bild
        mode = 'a' if file_exists else 'w'
        
        with open(csv_path, mode) as f:
            # Schreibe Header nur beim ersten Bild
            if not file_exists:
                # Erstelle Header: "Name,Gewicht 1,Gewicht 2,Gewicht 3,..."
                header_parts = ["Name"]
                for i in range(1, hidden_dim + 1):
                    header_parts.append(f"Gewicht {i}")
                header = ",".join(header_parts)
                f.write(header + '\n')
            
            # Schreibe alle Queries für dieses Bild
            for query_idx in range(num_queries_actual):
                query_name = f'"{image_id}, Query{query_idx + 1}"'  # Mit image_id und 1-basierte Query-Nummerierung
                weights = queries[query_idx, :]
                
                # Formatiere Gewichte als kommagetrennte Liste
                weight_strings = [f"{weight:.6f}" for weight in weights]
                
                # Schreibe Zeile: Name,weight1,weight2,weight3,...
                line = query_name + "," + ",".join(weight_strings)
                f.write(line + '\n')
        
        print(f"✅ {'Erweitert' if file_exists else 'Erstellt'}: {layer_number}/Query.csv")
        print(f"   📊 {image_id}: {num_queries_actual} Queries x {hidden_dim} Gewichte")
        print(f"   📂 Pfad: {csv_path}")
        
        # Keine Metadaten-JSONs mehr erstellen


def load_and_preprocess_image(image_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Lädt und preprocesst Bild für MaskDINO-Modell-Eingabe.
    Erstellt das richtige Input-Format wie das Modell es erwartet.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Standard-Preprocessing für MaskDINO
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    
    # Resize auf Modell-Eingabegröße (800x800 für MaskDINO)
    image_resized = cv2.resize(image_rgb, (800, 800))
    
    # Konvertiere zu Tensor
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
    
    # ImageNet-Normalisierung
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Erstelle das Input-Format wie MaskDINO es erwartet
    batched_input = {
        "image": image_tensor,  # [3, 800, 800]
        "height": original_height,
        "width": original_width,
        "file_name": os.path.basename(image_path)
    }
    
    return batched_input


def accept_weights_model_images(weights_path: str, model: torch.nn.Module, 
                               image_list: List[Tuple[str, str]], num_queries: int = 300,  # Geändert: Liste von (image_id, image_path)
                               output_dir: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/decoder") -> Dict[str, Any]:
    """
    Hauptfunktion für Network Dissection auf Transformer-Decoder.
    Extrahiert Queries pro Layer (CSV) und Pixel-Embedding-Maps (NPY).
    
    Args:
        image_list: Liste von (image_id, image_path) Tupeln
    """
    print(f"Starte Network Dissection Datenextraktion für {len(image_list)} Bilder...")
    
    # Lade Gewichte falls vorhanden
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model', checkpoint))
        print("Modell-Gewichte geladen.")
    
    model.eval()
    
    # Registriere Decoder-Hooks
    hidden_states_per_layer, hook_handles = register_decoder_hooks(model)
    
    # Erstelle Ausgabeverzeichnisse
    os.makedirs(output_dir, exist_ok=True)
    
    results = {"processed": 0, "failed": 0}
    
    try:
        for image_id, image_path in image_list:
            print(f"Verarbeite {image_id}: {os.path.basename(image_path)}")
            
            try:
                # Lade Bild
                batched_input = load_and_preprocess_image(image_path)
                if batched_input is None:
                    results["failed"] += 1
                    continue
                
                # Leere Hidden-States für neues Bild
                for layer_states in hidden_states_per_layer.values():
                    layer_states.clear()
                
                # 1. Extrahiere Pixel-Embedding-Map mit image_id
                extract_pixel_embedding_map(model, batched_input, image_id, output_dir)
                
                # 2. Forward-Pass für Decoder-Hidden-States mit korrektem Input-Format
                with torch.no_grad():
                    _ = model([batched_input])  # Liste von Inputs wie erwartet
                
                # 3. Speichere Queries als CSV mit image_id
                save_queries_to_csv(hidden_states_per_layer, num_queries, image_id, output_dir)
                
                results["processed"] += 1
                print(f"✅ {image_id} erfolgreich verarbeitet")
                
            except Exception as e:
                print(f"❌ Fehler beim Verarbeiten von {image_id}: {str(e)}")
                results["failed"] += 1
                continue
                
    except Exception as e:
        print(f"❌ Allgemeiner Fehler: {e}")
        results["failed"] += len(image_list) - results["processed"]
    
    finally:
        # Cleanup Hooks
        for handle in hook_handles:
            handle.remove()
        print(f"{len(hook_handles)} Hooks entfernt.")
    
    print(f"Abgeschlossen: {results['processed']} erfolgreich, {results['failed']} fehlgeschlagen")
    return results