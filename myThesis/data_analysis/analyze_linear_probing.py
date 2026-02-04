"""
Analyse der Linear-Probing-Ergebnisse aus den Experimenten.
Erstellt Tabelle 6: Linear Probing Precision pro Modell, Farbe und Finetuning-Schritt.

Extrahiert die Precision-Werte für orange, grau, blau und background aus den JSON-Dateien.
Sowohl für balanced als auch unbalanced Experimente.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from glob import glob
import re
import warnings

warnings.filterwarnings('ignore')


# Konfiguration
BASE_PATH = Path("/Volumes/Untitled/Master-Arbeit_Ergebnisse/output")
OUTPUT_PATH = Path("/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/data_analysis")

MODELS = ["car", "butterfly"]
FINETUNES = ["finetune1", "finetune2", "finetune3"]
COLORS = ["orange", "grau", "blau", "background"]
ENCODER_DECODER = ["encoder", "decoder"]
ENCODER_LAYERS = ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"]
DECODER_LAYERS = ["layer0", "layer1", "layer2"]


def get_layers(enc_dec: str) -> List[str]:
    """Gibt die Layer-Liste für Encoder oder Decoder zurück."""
    return ENCODER_LAYERS if enc_dec == "encoder" else DECODER_LAYERS


def find_linear_probe_json(model: str, finetune: str, enc_dec: str, layer: str, balanced: bool) -> Optional[Path]:
    """
    Findet die passende Linear-Probe JSON-Datei.
    Falls mehrere existieren, wird die erste genommen.
    """
    results_dir = BASE_PATH / model / finetune / "linear_probing" / f"{enc_dec}_results" / layer
    
    if not results_dir.exists():
        return None
    
    # Pattern für balanced oder unbalanced
    if balanced:
        pattern = f"linear_probe_results_{layer}_balanced_*.json"
    else:
        # Unbalanced: ohne "balanced" im Namen
        pattern = f"linear_probe_results_{layer}_*.json"
    
    json_files = list(results_dir.glob(pattern))
    
    # Filter: für unbalanced nur Dateien ohne "balanced"
    if not balanced:
        json_files = [f for f in json_files if "balanced" not in f.name]
    
    if json_files:
        # Nehme die erste Datei (oder sortiere nach Datum, falls gewünscht)
        return sorted(json_files)[0]
    
    return None


def load_linear_probe_results(json_path: Path) -> Optional[Dict]:
    """Lädt die Linear-Probe-Ergebnisse aus einer JSON-Datei."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden von {json_path}: {e}")
        return None


def extract_precision_values(results: Dict, colors: List[str]) -> Dict[str, Optional[float]]:
    """
    Extrahiert die Precision-Werte für die angegebenen Farben aus den Ergebnissen.
    """
    precision_dict = {}
    
    if results is None or "per_class_metrics" not in results:
        return {color: None for color in colors}
    
    per_class = results["per_class_metrics"]
    
    for color in colors:
        if color in per_class and "precision" in per_class[color]:
            precision_dict[color] = per_class[color]["precision"]
        else:
            precision_dict[color] = None
    
    return precision_dict


def create_linear_probing_table() -> pd.DataFrame:
    """
    Erstellt die Haupttabelle mit allen Linear-Probing Precision-Werten.
    
    Struktur:
    - model: car, butterfly
    - finetune: finetune1, finetune2, finetune3
    - encoder_decoder: encoder, decoder
    - layer: layer0, layer1, ...
    - Für jede Farbe (orange, grau, blau, background):
      - precision_unbalanced, precision_balanced
    """
    rows = []
    
    for model in MODELS:
        print(f"\nVerarbeite Modell: {model}")
        
        for finetune in FINETUNES:
            for enc_dec in ENCODER_DECODER:
                layers = get_layers(enc_dec)
                
                for layer in layers:
                    row = {
                        "model": model,
                        "finetune": finetune,
                        "encoder_decoder": enc_dec,
                        "layer": layer
                    }
                    
                    # Unbalanced
                    json_path = find_linear_probe_json(model, finetune, enc_dec, layer, balanced=False)
                    if json_path:
                        results = load_linear_probe_results(json_path)
                        precision_values = extract_precision_values(results, COLORS)
                        for color in COLORS:
                            col_name = f"{color}_precision_unbalanced"
                            row[col_name] = precision_values.get(color)
                    else:
                        for color in COLORS:
                            col_name = f"{color}_precision_unbalanced"
                            row[col_name] = None
                    
                    # Balanced
                    json_path_balanced = find_linear_probe_json(model, finetune, enc_dec, layer, balanced=True)
                    if json_path_balanced:
                        results_balanced = load_linear_probe_results(json_path_balanced)
                        precision_values_balanced = extract_precision_values(results_balanced, COLORS)
                        for color in COLORS:
                            col_name = f"{color}_precision_balanced"
                            row[col_name] = precision_values_balanced.get(color)
                    else:
                        for color in COLORS:
                            col_name = f"{color}_precision_balanced"
                            row[col_name] = None
                    
                    rows.append(row)
                    print(f"  {finetune}/{enc_dec}/{layer}: Daten extrahiert")
    
    # DataFrame erstellen
    df = pd.DataFrame(rows)
    
    # Spalten sortieren
    base_cols = ["model", "finetune", "encoder_decoder", "layer"]
    color_cols = []
    for color in COLORS:
        color_cols.append(f"{color}_precision_unbalanced")
        color_cols.append(f"{color}_precision_balanced")
    
    # Nur existierende Spalten verwenden
    all_cols = base_cols + [c for c in color_cols if c in df.columns]
    df = df[all_cols]
    
    return df


def add_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt zusammenfassende Statistiken hinzu.
    Da Finetune-Schritte jetzt in den Zeilen sind, werden keine zusätzlichen
    Durchschnittsspalten benötigt - die Aggregation erfolgt bei Bedarf.
    """
    return df.copy()


def print_summary(df: pd.DataFrame):
    """Druckt eine Zusammenfassung der Ergebnisse."""
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG: Linear Probing Precision")
    print("=" * 80)
    
    for model in MODELS:
        print(f"\n--- Modell: {model.upper()} ---")
        model_df = df[df["model"] == model]
        
        for finetune in FINETUNES:
            ft_df = model_df[model_df["finetune"] == finetune]
            if ft_df.empty:
                continue
            
            print(f"\n  {finetune.upper()}:")
            
            for enc_dec in ENCODER_DECODER:
                enc_df = ft_df[ft_df["encoder_decoder"] == enc_dec]
                if enc_df.empty:
                    continue
                
                print(f"    {enc_dec.upper()}:")
                
                for color in COLORS:
                    unbal_col = f"{color}_precision_unbalanced"
                    bal_col = f"{color}_precision_balanced"
                    
                    if unbal_col in enc_df.columns:
                        unbal_mean = enc_df[unbal_col].mean()
                        print(f"      {color} (unbalanced): {unbal_mean:.4f}" if pd.notna(unbal_mean) else f"      {color} (unbalanced): N/A")
                    
                    if bal_col in enc_df.columns:
                        bal_mean = enc_df[bal_col].mean()
                        print(f"      {color} (balanced):   {bal_mean:.4f}" if pd.notna(bal_mean) else f"      {color} (balanced): N/A")


def main():
    """Hauptfunktion zum Erstellen der Linear-Probing-Tabelle."""
    print("=" * 80)
    print("Erstelle Tabelle 6: Linear Probing Precision")
    print("=" * 80)
    
    # Prüfe ob Basispfad existiert
    if not BASE_PATH.exists():
        print(f"FEHLER: Basispfad nicht gefunden: {BASE_PATH}")
        print("Bitte stellen Sie sicher, dass das externe Volume gemountet ist.")
        return
    
    # Erstelle Output-Verzeichnis falls nicht vorhanden
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Erstelle Haupttabelle
    df = create_linear_probing_table()
    
    # Füge Zusammenfassungsstatistiken hinzu
    df = add_summary_statistics(df)
    
    # Speichere CSV (deutsches Format: ; als Trennzeichen, , als Dezimaltrennzeichen)
    output_file = OUTPUT_PATH / "table6_linear_probing.csv"
    df.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"\nTabelle gespeichert: {output_file}")
    
    # Drucke Zusammenfassung
    print_summary(df)
    
    # Zeige die ersten Zeilen
    print("\n" + "=" * 80)
    print("Vorschau der Tabelle:")
    print("=" * 80)
    print(df.head(10).to_string())
    
    print(f"\nGesamt: {len(df)} Zeilen")
    print(f"Spalten: {list(df.columns)}")


if __name__ == "__main__":
    main()
