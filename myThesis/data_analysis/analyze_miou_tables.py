"""
Analyse der mIoU-Werte aus Network Dissection Experimenten.
Erstellt 5 verschiedene Analysetabellen gemäß der spezifizierten Anforderungen.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Konfiguration
BASE_PATH = Path("/Volumes/Untitled/Master-Arbeit_Ergebnisse/output")
OUTPUT_PATH = Path("/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/data_analysis")

MODELS = ["car", "butterfly"]
FINETUNES = ["finetune1", "finetune2", "finetune3"]
COLORS = ["grau", "orange", "blau"]
ENCODER_DECODER = ["encoder", "decoder"]
ENCODER_LAYERS = ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"]
DECODER_LAYERS = ["layer0", "layer1", "layer2"]

MIOU_THRESHOLD = 0.04


def get_layers(enc_dec: str) -> List[str]:
    """Gibt die Layer-Liste für Encoder oder Decoder zurück."""
    return ENCODER_LAYERS if enc_dec == "encoder" else DECODER_LAYERS


def load_miou_csv(model: str, finetune: str, color: str, enc_dec: str, layer: str) -> Optional[pd.DataFrame]:
    """Lädt eine mIoU CSV-Datei (Encoder oder Decoder Format)."""
    if enc_dec == "encoder":
        csv_path = BASE_PATH / model / finetune / color / enc_dec / layer / "miou_network_dissection.csv"
    else:  # decoder
        csv_path = BASE_PATH / model / finetune / color / enc_dec / layer / "mIoU_per_Query.csv"
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Normalisiere Spaltennamen für einheitliche Verarbeitung
            if enc_dec == "decoder":
                # Decoder hat andere Spaltennamen: query_idx -> feature_idx, mean_iou -> miou
                df = df.rename(columns={'query_idx': 'feature_idx', 'mean_iou': 'miou'})
            return df
        except Exception as e:
            print(f"Fehler beim Laden von {csv_path}: {e}")
            return None
    return None


def compute_layer_stats(df: pd.DataFrame) -> Dict:
    """Berechnet Statistiken für einen Layer."""
    miou_values = df['miou'].values
    
    count_above_threshold = np.sum(miou_values > MIOU_THRESHOLD)
    mean_miou = np.mean(miou_values)
    var_miou = np.var(miou_values)
    best_miou = np.max(miou_values)
    best_idx = np.argmax(miou_values)
    best_feature = df.iloc[best_idx]['feature_idx']
    
    return {
        'count_above_threshold': int(count_above_threshold),
        'mean_miou': mean_miou,
        'variance_miou': var_miou,
        'best_miou': best_miou,
        'best_feature': int(best_feature)
    }


def create_table1_iou_over_layers():
    """
    Tabelle 1: IoU über Layer
    Pro Modell, pro Farbe, pro Finetuning, pro Layer (108 Stück)
    -> Anzahl mIoU über 0.04, mIoU, Varianz der IoU, best IoU, Feature of best IoU
    """
    print("Erstelle Tabelle 1: IoU über Layer...")
    
    results = []
    
    for model in MODELS:
        for finetune in FINETUNES:
            for color in COLORS:
                for enc_dec in ENCODER_DECODER:
                    layers = get_layers(enc_dec)
                    for layer in layers:
                        df = load_miou_csv(model, finetune, color, enc_dec, layer)
                        if df is not None and len(df) > 0:
                            stats = compute_layer_stats(df)
                            results.append({
                                'model': model,
                                'finetune': finetune,
                                'color': color,
                                'encoder_decoder': enc_dec,
                                'layer': layer,
                                'count_above_0.04': stats['count_above_threshold'],
                                'mean_miou': round(stats['mean_miou'], 6),
                                'variance_miou': round(stats['variance_miou'], 8),
                                'best_miou': round(stats['best_miou'], 6),
                                'best_feature': stats['best_feature']
                            })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table1_iou_per_layer.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def create_table2_iou_over_finetuning():
    """
    Tabelle 2: IoU über Finetuning
    Pro Modell, pro Farbe, pro Finetuning (12 Stück)
    -> Anzahl mIoU über 0.04, mIoU, Varianz der IoU, best IoU, Spanne der Varianz, 
       Feature of best IoU, Layer of best IoU
    """
    print("Erstelle Tabelle 2: IoU über Finetuning...")
    
    results = []
    
    for model in MODELS:
        for finetune in FINETUNES:
            for color in COLORS:
                all_miou_values = []
                layer_variances = []
                best_overall_miou = -1
                best_overall_feature = None
                best_overall_layer = None
                best_overall_enc_dec = None
                count_above_total = 0
                
                for enc_dec in ENCODER_DECODER:
                    layers = get_layers(enc_dec)
                    for layer in layers:
                        df = load_miou_csv(model, finetune, color, enc_dec, layer)
                        if df is not None and len(df) > 0:
                            miou_values = df['miou'].values
                            all_miou_values.extend(miou_values)
                            layer_variances.append(np.var(miou_values))
                            count_above_total += np.sum(miou_values > MIOU_THRESHOLD)
                            
                            max_miou = np.max(miou_values)
                            if max_miou > best_overall_miou:
                                best_overall_miou = max_miou
                                best_idx = np.argmax(miou_values)
                                best_overall_feature = int(df.iloc[best_idx]['feature_idx'])
                                best_overall_layer = f"{enc_dec}/{layer}"
                
                if all_miou_values:
                    variance_span = max(layer_variances) - min(layer_variances) if layer_variances else 0
                    results.append({
                        'model': model,
                        'finetune': finetune,
                        'color': color,
                        'count_above_0.04': int(count_above_total),
                        'mean_miou': round(np.mean(all_miou_values), 6),
                        'variance_miou': round(np.var(all_miou_values), 8),
                        'best_miou': round(best_overall_miou, 6),
                        'variance_span': round(variance_span, 8),
                        'best_feature': best_overall_feature,
                        'best_layer': best_overall_layer
                    })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table2_iou_per_finetuning.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def create_table3_iou_per_color_and_model():
    """
    Tabelle 3: IoU pro Farbe und Modell
    Pro Modell, pro Farbe (4+2 = 6 Stück)
    -> Anzahl mIoU über 0.04, mIoU, Varianz der IoU, best IoU, Spanne der Varianz,
       Feature of best IoU, Layer of best IoU
    """
    print("Erstelle Tabelle 3: IoU pro Farbe und Modell...")
    
    results = []
    
    for model in MODELS:
        for color in COLORS:
            all_miou_values = []
            layer_variances = []
            best_overall_miou = -1
            best_overall_feature = None
            best_overall_layer = None
            count_above_total = 0
            
            for finetune in FINETUNES:
                for enc_dec in ENCODER_DECODER:
                    layers = get_layers(enc_dec)
                    for layer in layers:
                        df = load_miou_csv(model, finetune, color, enc_dec, layer)
                        if df is not None and len(df) > 0:
                            miou_values = df['miou'].values
                            all_miou_values.extend(miou_values)
                            layer_variances.append(np.var(miou_values))
                            count_above_total += np.sum(miou_values > MIOU_THRESHOLD)
                            
                            max_miou = np.max(miou_values)
                            if max_miou > best_overall_miou:
                                best_overall_miou = max_miou
                                best_idx = np.argmax(miou_values)
                                best_overall_feature = int(df.iloc[best_idx]['feature_idx'])
                                best_overall_layer = f"{finetune}/{enc_dec}/{layer}"
            
            if all_miou_values:
                variance_span = max(layer_variances) - min(layer_variances) if layer_variances else 0
                results.append({
                    'model': model,
                    'color': color,
                    'count_above_0.04': int(count_above_total),
                    'mean_miou': round(np.mean(all_miou_values), 6),
                    'variance_miou': round(np.var(all_miou_values), 8),
                    'best_miou': round(best_overall_miou, 6),
                    'variance_span': round(variance_span, 8),
                    'best_feature': best_overall_feature,
                    'best_layer': best_overall_layer
                })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table3_iou_per_color_model.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def create_table4_iou_feature_comparison():
    """
    Tabelle 4: IoU pro Feature über Layer (Vergleich der 3 Finetuning Schritte)
    Pro Modell, pro Farbe, pro Layer (36 Stück)
    -> Spanne der Varianz (in den 3 Features), 10 Feature mit höchster Varianz,
       10 Feature mit niedrigster Varianz, 10 Feature mit höchstem mIoU,
       prozentuale Überschneidung, Anzahl mIoU über 0.04 pro Finetuning-Schritt,
       prozentzahl gleiche über 0.04 im zweiten wie im ersten,
       prozentzahl gleiche über 0.04 im dritten wie im zweiten
    """
    print("Erstelle Tabelle 4: IoU pro Feature über Layer (Finetuning Vergleich)...")
    
    results = []
    
    for model in MODELS:
        for color in COLORS:
            for enc_dec in ENCODER_DECODER:
                layers = get_layers(enc_dec)
                for layer in layers:
                    # Lade Daten für alle 3 Finetuning-Schritte
                    dfs = {}
                    for finetune in FINETUNES:
                        df = load_miou_csv(model, finetune, color, enc_dec, layer)
                        if df is not None:
                            dfs[finetune] = df
                    
                    if len(dfs) < 3:
                        continue
                    
                    # Erstelle Feature-basierte Analyse
                    # Für jeden Finetuning-Schritt: Feature -> mIoU
                    feature_data = {}
                    for finetune, df in dfs.items():
                        for _, row in df.iterrows():
                            feat_idx = int(row['feature_idx'])
                            if feat_idx not in feature_data:
                                feature_data[feat_idx] = {}
                            feature_data[feat_idx][finetune] = row['miou']
                    
                    # Berechne Varianz pro Feature über die 3 Finetuning-Schritte
                    feature_variances = {}
                    feature_mean_miou = {}
                    for feat_idx, ft_miou in feature_data.items():
                        if len(ft_miou) == 3:  # Nur Features die in allen 3 vorkommen
                            values = [ft_miou[ft] for ft in FINETUNES]
                            feature_variances[feat_idx] = np.var(values)
                            feature_mean_miou[feat_idx] = np.mean(values)
                    
                    if not feature_variances:
                        continue
                    
                    # Spanne der Varianz
                    variance_values = list(feature_variances.values())
                    variance_span = max(variance_values) - min(variance_values)
                    
                    # Top 10 Features nach Varianz (höchste)
                    sorted_by_var_desc = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
                    top_10_high_var = [f[0] for f in sorted_by_var_desc[:10]]
                    
                    # Top 10 Features nach Varianz (niedrigste)
                    sorted_by_var_asc = sorted(feature_variances.items(), key=lambda x: x[1])
                    top_10_low_var = [f[0] for f in sorted_by_var_asc[:10]]
                    
                    # Top 10 Features nach mIoU
                    sorted_by_miou = sorted(feature_mean_miou.items(), key=lambda x: x[1], reverse=True)
                    top_10_miou = [f[0] for f in sorted_by_miou[:10]]
                    
                    # Prozentuale Überschneidung: Wie viel % der top10_high_variance_features haben mIoU > 0.04
                    high_var_above_threshold = sum(
                        1 for feat in top_10_high_var 
                        if feature_mean_miou.get(feat, 0) > MIOU_THRESHOLD
                    )
                    overlap_percent = (high_var_above_threshold / len(top_10_high_var)) * 100 if top_10_high_var else 0
                    
                    # Anzahl über Threshold pro Finetuning
                    counts_above = {}
                    above_threshold_sets = {}
                    for finetune in FINETUNES:
                        df = dfs[finetune]
                        above_mask = df['miou'] > MIOU_THRESHOLD
                        counts_above[finetune] = int(above_mask.sum())
                        above_threshold_sets[finetune] = set(df[above_mask]['feature_idx'].astype(int).tolist())
                    
                    # Prozentzahl gleiche über 0.04 im zweiten wie im ersten
                    set1 = above_threshold_sets['finetune1']
                    set2 = above_threshold_sets['finetune2']
                    set3 = above_threshold_sets['finetune3']
                    
                    same_1_2 = len(set1.intersection(set2)) / len(set1) * 100 if set1 else 0
                    same_2_3 = len(set2.intersection(set3)) / len(set2) * 100 if set2 else 0
                    
                    # Top 40 Features nach mIoU
                    top_40_miou = [f[0] for f in sorted_by_miou[:40]]
                    
                    # Top 40 Features pro Finetuning-Schritt
                    top40_sets = {}
                    for finetune in FINETUNES:
                        df = dfs[finetune]
                        top40_df = df.nlargest(40, 'miou')
                        top40_sets[finetune] = set(top40_df['feature_idx'].astype(int).tolist())
                    
                    # Prozentzahl gleiche Top 40 im zweiten wie im ersten
                    top40_set1 = top40_sets['finetune1']
                    top40_set2 = top40_sets['finetune2']
                    top40_set3 = top40_sets['finetune3']
                    
                    top40_same_1_2 = len(top40_set1.intersection(top40_set2)) / len(top40_set1) * 100 if top40_set1 else 0
                    top40_same_2_3 = len(top40_set2.intersection(top40_set3)) / len(top40_set2) * 100 if top40_set2 else 0
                    
                    results.append({
                        'model': model,
                        'color': color,
                        'encoder_decoder': enc_dec,
                        'layer': layer,
                        'variance_span': round(variance_span, 10),
                        'top10_high_variance_features': str(top_10_high_var),
                        'top10_low_variance_features': str(top_10_low_var),
                        'top10_high_miou_features': str(top_10_miou),
                        'top40_high_miou_features': str(top_40_miou),
                        'overlap_percent': round(overlap_percent, 2),
                        'count_above_0.04_ft1': counts_above['finetune1'],
                        'count_above_0.04_ft2': counts_above['finetune2'],
                        'count_above_0.04_ft3': counts_above['finetune3'],
                        'percent_same_ft1_ft2': round(same_1_2, 2),
                        'percent_same_ft2_ft3': round(same_2_3, 2),
                        'top40_percent_same_ft1_ft2': round(top40_same_1_2, 2),
                        'top40_percent_same_ft2_ft3': round(top40_same_2_3, 2)
                    })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table4_iou_feature_over_finetuning.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def create_table5_iou_feature_over_colors_models():
    """
    Tabelle 5: IoU pro Feature über Farben und Modelle
    Pro Modell, pro Farbe * pro Modell (6 Stück)
    -> Spanne der Varianz, Layer mit größter Varianzspanne, Layer mit niedrigster Varianzspanne,
       Anzahl mIoU über 0.04 pro Finetuning-Schritt,
       prozentzahl gleiche über 0.04 im zweiten wie im ersten,
       prozentzahl gleiche über 0.04 im dritten wie im zweiten
    """
    print("Erstelle Tabelle 5: IoU pro Feature über Farben und Modelle...")
    
    results = []
    
    for model in MODELS:
        for color in COLORS:
            layer_variance_spans = {}
            total_counts_above = {ft: 0 for ft in FINETUNES}
            total_above_sets = {ft: set() for ft in FINETUNES}
            total_top40_sets = {ft: set() for ft in FINETUNES}
            all_variance_spans = []
            
            for enc_dec in ENCODER_DECODER:
                layers = get_layers(enc_dec)
                for layer in layers:
                    layer_key = f"{enc_dec}/{layer}"
                    
                    # Lade Daten für alle 3 Finetuning-Schritte
                    dfs = {}
                    for finetune in FINETUNES:
                        df = load_miou_csv(model, finetune, color, enc_dec, layer)
                        if df is not None:
                            dfs[finetune] = df
                    
                    if len(dfs) < 3:
                        continue
                    
                    # Feature-basierte Varianz
                    feature_data = {}
                    for finetune, df in dfs.items():
                        for _, row in df.iterrows():
                            feat_idx = int(row['feature_idx'])
                            if feat_idx not in feature_data:
                                feature_data[feat_idx] = {}
                            feature_data[feat_idx][finetune] = row['miou']
                    
                    # Varianz pro Feature
                    feature_variances = {}
                    for feat_idx, ft_miou in feature_data.items():
                        if len(ft_miou) == 3:
                            values = [ft_miou[ft] for ft in FINETUNES]
                            feature_variances[feat_idx] = np.var(values)
                    
                    if feature_variances:
                        variance_values = list(feature_variances.values())
                        layer_variance_span = max(variance_values) - min(variance_values)
                        layer_variance_spans[layer_key] = layer_variance_span
                        all_variance_spans.append(layer_variance_span)
                    
                    # Counts und Sets für Überschneidungen
                    for finetune in FINETUNES:
                        df = dfs[finetune]
                        above_mask = df['miou'] > MIOU_THRESHOLD
                        total_counts_above[finetune] += int(above_mask.sum())
                        # Add layer info to feature idx for unique identification
                        above_features = set(
                            (layer_key, int(f)) for f in df[above_mask]['feature_idx'].tolist()
                        )
                        total_above_sets[finetune].update(above_features)
                        
                        # Top 40 Features pro Finetuning-Schritt
                        top40_df = df.nlargest(40, 'miou')
                        top40_features = set(
                            (layer_key, int(f)) for f in top40_df['feature_idx'].tolist()
                        )
                        total_top40_sets[finetune].update(top40_features)
            
            if all_variance_spans and layer_variance_spans:
                overall_variance_span = max(all_variance_spans) - min(all_variance_spans)
                
                # Layer mit größter und kleinster Varianzspanne
                sorted_layers = sorted(layer_variance_spans.items(), key=lambda x: x[1], reverse=True)
                layer_max_var = sorted_layers[0][0] if sorted_layers else None
                layer_min_var = sorted_layers[-1][0] if sorted_layers else None
                
                # Prozentzahlen
                set1 = total_above_sets['finetune1']
                set2 = total_above_sets['finetune2']
                set3 = total_above_sets['finetune3']
                
                same_1_2 = len(set1.intersection(set2)) / len(set1) * 100 if set1 else 0
                same_2_3 = len(set2.intersection(set3)) / len(set2) * 100 if set2 else 0
                
                # Top 40 Prozentzahlen
                top40_set1 = total_top40_sets['finetune1']
                top40_set2 = total_top40_sets['finetune2']
                top40_set3 = total_top40_sets['finetune3']
                
                top40_same_1_2 = len(top40_set1.intersection(top40_set2)) / len(top40_set1) * 100 if top40_set1 else 0
                top40_same_2_3 = len(top40_set2.intersection(top40_set3)) / len(top40_set2) * 100 if top40_set2 else 0
                
                results.append({
                    'model': model,
                    'color': color,
                    'variance_span': round(overall_variance_span, 10),
                    'layer_max_variance_span': layer_max_var,
                    'layer_min_variance_span': layer_min_var,
                    'count_above_0.04_ft1': total_counts_above['finetune1'],
                    'count_above_0.04_ft2': total_counts_above['finetune2'],
                    'count_above_0.04_ft3': total_counts_above['finetune3'],
                    'percent_same_ft1_ft2': round(same_1_2, 2),
                    'percent_same_ft2_ft3': round(same_2_3, 2),
                    'top40_percent_same_ft1_ft2': round(top40_same_1_2, 2),
                    'top40_percent_same_ft2_ft3': round(top40_same_2_3, 2)
                })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table5_iou_feature_over_colors_models.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def main():
    """Hauptfunktion - erstellt alle 5 Tabellen."""
    print("=" * 60)
    print("mIoU Datenanalyse - Erstellung der Analysetabellen")
    print("=" * 60)
    print(f"Datenquelle: {BASE_PATH}")
    print(f"Ausgabe: {OUTPUT_PATH}")
    print(f"mIoU Schwellenwert: {MIOU_THRESHOLD}")
    print("=" * 60)
    
    # Stelle sicher, dass Output-Verzeichnis existiert
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Erstelle alle Tabellen
    table1 = create_table1_iou_over_layers()
    table2 = create_table2_iou_over_finetuning()
    table3 = create_table3_iou_per_color_and_model()
    table4 = create_table4_iou_feature_comparison()
    table5 = create_table5_iou_feature_over_colors_models()
    
    print("=" * 60)
    print("Alle Tabellen erfolgreich erstellt!")
    print("=" * 60)
    
    # Zusammenfassung
    print("\nZusammenfassung:")
    print(f"  Tabelle 1 (IoU über Layer):             {len(table1)} Zeilen")
    print(f"  Tabelle 2 (IoU über Finetuning):        {len(table2)} Zeilen")
    print(f"  Tabelle 3 (IoU pro Farbe/Modell):       {len(table3)} Zeilen")
    print(f"  Tabelle 4 (Feature Vergleich):          {len(table4)} Zeilen")
    print(f"  Tabelle 5 (Feature über Farben/Modelle): {len(table5)} Zeilen")


if __name__ == "__main__":
    main()
