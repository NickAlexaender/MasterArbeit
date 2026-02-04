"""
Analyse der LRP-Werte (Layer-wise Relevance Propagation) aus den Experimenten.
Erstellt 3 verschiedene Analysetabellen (Tabellen 7, 8, 9).

WICHTIG: Relevanzwerte werden in positive und negative getrennt behandelt.
- Positive Relevanz: Summe aller positiven normalized_relevance Werte
- Negative Relevanz: Summe aller negativen normalized_relevance Werte (als Absolutwert)
- Für jede Coverage-Berechnung (10%, 20%, 50%) gibt es zwei Werte:
  - Anzahl positiver Werte für X% der positiven Relevanz
  - Anzahl negativer Werte für X% der negativen Relevanz
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from scipy.stats import pearsonr, spearmanr
import warnings
import re
from glob import glob

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


def get_prev_layer(layer: str) -> Optional[str]:
    """Gibt den Vorlayer zurück (layer1 -> layer0, etc.)."""
    match = re.match(r'layer(\d+)', layer)
    if match:
        layer_num = int(match.group(1))
        if layer_num > 0:
            return f"layer{layer_num - 1}"
    return None


def load_miou_csv(model: str, finetune: str, color: str, enc_dec: str, layer: str) -> Optional[pd.DataFrame]:
    """Lädt eine mIoU CSV-Datei (Encoder oder Decoder Format)."""
    if enc_dec == "encoder":
        csv_path = BASE_PATH / model / finetune / color / enc_dec / layer / "miou_network_dissection.csv"
    else:  # decoder
        csv_path = BASE_PATH / model / finetune / color / enc_dec / layer / "mIoU_per_Query.csv"
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if enc_dec == "decoder":
                df = df.rename(columns={'query_idx': 'feature_idx', 'mean_iou': 'miou'})
            return df
        except Exception as e:
            print(f"Fehler beim Laden von {csv_path}: {e}")
            return None
    return None


def get_miou_dict(model: str, finetune: str, color: str, enc_dec: str, layer: str) -> Dict[int, float]:
    """Erstellt ein Dictionary feature_idx -> mIoU für einen Layer."""
    df = load_miou_csv(model, finetune, color, enc_dec, layer)
    if df is not None:
        return dict(zip(df['feature_idx'].astype(int), df['miou']))
    return {}


def load_lrp_csv(model: str, finetune: str, color: str, enc_dec: str, layer: str, feature: int) -> Optional[pd.DataFrame]:
    """Lädt eine LRP CSV-Datei."""
    csv_path = BASE_PATH / model / finetune / color / "lrp" / enc_dec / f"{layer}_feat{feature}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Fehler beim Laden von {csv_path}: {e}")
            return None
    return None


def get_lrp_files(model: str, finetune: str, color: str, enc_dec: str) -> List[Tuple[str, int]]:
    """Findet alle LRP-Dateien und gibt (layer, feature) Tupel zurück."""
    lrp_dir = BASE_PATH / model / finetune / color / "lrp" / enc_dec
    if not lrp_dir.exists():
        return []
    
    results = []
    pattern = re.compile(r'layer(\d+)_feat(\d+)\.csv')
    
    for f in lrp_dir.glob("*.csv"):
        match = pattern.match(f.name)
        if match:
            layer = f"layer{match.group(1)}"
            feature = int(match.group(2))
            results.append((layer, feature))
    
    return results


def split_positive_negative(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Teilt DataFrame in positive und negative normalized_relevance Werte.
    Gibt (positive_df, negative_df) zurück.
    Bei negative_df werden die Werte als Absolutwerte behandelt (für Sortierung).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    positive_df = df[df['normalized_relevance'] > 0].copy()
    negative_df = df[df['normalized_relevance'] < 0].copy()
    
    # Sortiere positive absteigend nach Relevanz
    positive_df = positive_df.sort_values('normalized_relevance', ascending=False).reset_index(drop=True)
    
    # Sortiere negative nach Absolutwert absteigend (also nach Wert aufsteigend, da negativ)
    negative_df = negative_df.sort_values('normalized_relevance', ascending=True).reset_index(drop=True)
    # Füge eine Spalte mit Absolutwerten hinzu für einfachere Berechnung
    negative_df['abs_relevance'] = negative_df['normalized_relevance'].abs()
    
    return positive_df, negative_df


def get_connections_for_coverage_positive(df: pd.DataFrame, coverage: float) -> int:
    """
    Berechnet die Anzahl der positiven Connections für eine bestimmte Coverage.
    df sollte bereits nach normalized_relevance absteigend sortiert sein.
    """
    if df is None or len(df) == 0:
        return 0
    
    total = df['normalized_relevance'].sum()
    if total == 0:
        return len(df)
    
    cumsum = df['normalized_relevance'].cumsum()
    threshold = coverage * total
    
    connections_needed = (cumsum >= threshold).idxmax() + 1 if (cumsum >= threshold).any() else len(df)
    return int(connections_needed)


def get_connections_for_coverage_negative(df: pd.DataFrame, coverage: float) -> int:
    """
    Berechnet die Anzahl der negativen Connections für eine bestimmte Coverage.
    df sollte bereits nach abs_relevance absteigend sortiert sein.
    """
    if df is None or len(df) == 0 or 'abs_relevance' not in df.columns:
        return 0
    
    total = df['abs_relevance'].sum()
    if total == 0:
        return len(df)
    
    cumsum = df['abs_relevance'].cumsum()
    threshold = coverage * total
    
    connections_needed = (cumsum >= threshold).idxmax() + 1 if (cumsum >= threshold).any() else len(df)
    return int(connections_needed)


def get_features_for_coverage_positive(df: pd.DataFrame, coverage: float) -> Set[int]:
    """Gibt Feature-Indizes für positive Coverage zurück."""
    if df is None or len(df) == 0:
        return set()
    
    n = get_connections_for_coverage_positive(df, coverage)
    return set(df.iloc[:n]['prev_feature_idx'].astype(int).tolist())


def get_features_for_coverage_negative(df: pd.DataFrame, coverage: float) -> Set[int]:
    """Gibt Feature-Indizes für negative Coverage zurück."""
    if df is None or len(df) == 0 or 'abs_relevance' not in df.columns:
        return set()
    
    n = get_connections_for_coverage_negative(df, coverage)
    return set(df.iloc[:n]['prev_feature_idx'].astype(int).tolist())


def compute_correlation(lrp_df: pd.DataFrame, miou_dict: Dict[int, float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Berechnet die Korrelation zwischen Relevanz und mIoU des Vorlayers.
    Gibt (Pearson r für positive, p-Wert positive, Pearson r für negative, p-Wert negative) zurück.
    """
    if lrp_df is None or len(lrp_df) == 0 or not miou_dict:
        return None, None, None, None
    
    positive_df, negative_df = split_positive_negative(lrp_df)
    
    # Positive Korrelation
    pos_corr = None
    pos_pvalue = None
    if len(positive_df) >= 3:
        relevances = []
        mious = []
        for _, row in positive_df.iterrows():
            feat_idx = int(row['prev_feature_idx'])
            if feat_idx in miou_dict:
                relevances.append(row['normalized_relevance'])
                mious.append(miou_dict[feat_idx])
        if len(relevances) >= 3:
            try:
                pos_corr, pos_pvalue = pearsonr(relevances, mious)
            except:
                pass
    
    # Negative Korrelation (mit Absolutwerten)
    neg_corr = None
    neg_pvalue = None
    if len(negative_df) >= 3:
        relevances = []
        mious = []
        for _, row in negative_df.iterrows():
            feat_idx = int(row['prev_feature_idx'])
            if feat_idx in miou_dict:
                relevances.append(row['abs_relevance'])
                mious.append(miou_dict[feat_idx])
        if len(relevances) >= 3:
            try:
                neg_corr, neg_pvalue = pearsonr(relevances, mious)
            except:
                pass
    
    return pos_corr, pos_pvalue, neg_corr, neg_pvalue


def get_percent_above_threshold_at_coverage(features: Set[int], miou_dict: Dict[int, float]) -> Optional[float]:
    """
    Berechnet den Prozentsatz der Features, die einen mIoU über dem Threshold haben.
    """
    if not features or not miou_dict:
        return None
    
    count_above = sum(1 for f in features if miou_dict.get(f, 0) > MIOU_THRESHOLD)
    return (count_above / len(features)) * 100


def compute_outliers(df: pd.DataFrame, column: str = 'normalized_relevance') -> Dict[str, any]:
    """
    Führt eine Ausreißer-Analyse basierend auf IQR (Interquartile Range) durch.
    Gibt Dictionary mit Ausreißer-Statistiken zurück.
    """
    if df is None or len(df) == 0 or column not in df.columns:
        return {'n_outliers_high': 0, 'n_outliers_low': 0, 'outlier_pct': 0, 'outlier_indices': []}
    
    values = df[column].dropna()
    if len(values) < 4:
        return {'n_outliers_high': 0, 'n_outliers_low': 0, 'outlier_pct': 0, 'outlier_indices': []}
    
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_high = values[values > upper_bound]
    outliers_low = values[values < lower_bound]
    
    n_outliers = len(outliers_high) + len(outliers_low)
    outlier_pct = (n_outliers / len(values)) * 100 if len(values) > 0 else 0
    
    # Feature-Indizes der Ausreißer
    outlier_mask = (values > upper_bound) | (values < lower_bound)
    outlier_indices = df.loc[outlier_mask.index[outlier_mask], 'prev_feature_idx'].astype(int).tolist() if 'prev_feature_idx' in df.columns else []
    
    return {
        'n_outliers_high': len(outliers_high),
        'n_outliers_low': len(outliers_low),
        'outlier_pct': outlier_pct,
        'outlier_indices': outlier_indices,
        'iqr': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def compute_information_bottleneck(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Berechnet Information Bottleneck Metriken für die Relevanzverteilung.
    
    Metriken:
    - Gini-Koeffizient: Maß für Ungleichverteilung (0=gleichverteilt, 1=maximal konzentriert)
    - Entropie: Informationsgehalt der Verteilung (normalisiert auf 0-1)
    - Effective Features: Anzahl der Features, die effektiv zur Relevanz beitragen
    - Compression Ratio: Verhältnis von effektiven zu totalen Features
    """
    if df is None or len(df) == 0:
        return {'gini': None, 'entropy': None, 'effective_features': None, 'compression_ratio': None}
    
    positive_df, negative_df = split_positive_negative(df)
    
    results = {}
    
    for suffix, sub_df, col in [('_pos', positive_df, 'normalized_relevance'), 
                                  ('_neg', negative_df, 'abs_relevance')]:
        if len(sub_df) == 0 or col not in sub_df.columns:
            results[f'gini{suffix}'] = None
            results[f'entropy{suffix}'] = None
            results[f'effective_features{suffix}'] = None
            results[f'compression_ratio{suffix}'] = None
            continue
        
        values = sub_df[col].values
        total = values.sum()
        
        if total == 0:
            results[f'gini{suffix}'] = None
            results[f'entropy{suffix}'] = None
            results[f'effective_features{suffix}'] = None
            results[f'compression_ratio{suffix}'] = None
            continue
        
        # Normalisierte Wahrscheinlichkeiten
        probs = values / total
        
        # Gini-Koeffizient
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        cumulative = np.cumsum(sorted_probs)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n if n > 0 else 0
        
        # Entropie (normalisiert)
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
        max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Effective Features (basierend auf exponentieller Entropie)
        effective_features = 2 ** entropy if entropy > 0 else 1
        
        # Compression Ratio
        compression_ratio = effective_features / len(values) if len(values) > 0 else 0
        
        results[f'gini{suffix}'] = gini
        results[f'entropy{suffix}'] = normalized_entropy
        results[f'effective_features{suffix}'] = effective_features
        results[f'compression_ratio{suffix}'] = compression_ratio
    
    return results


def create_table7_lrp_per_feature():
    """
    Tabelle 7: LRP über Feature
    Pro Modell, pro Farbe, pro Finetune, pro Layer, pro Feature
    Mit separaten Werten für positive und negative Relevanz.
    """
    print("Erstelle Tabelle 7: LRP über Feature...")
    
    results = []
    
    for model in MODELS:
        for color in COLORS:
            for finetune in FINETUNES:
                for enc_dec in ENCODER_DECODER:
                    lrp_files = get_lrp_files(model, finetune, color, enc_dec)
                    
                    for layer, feature in lrp_files:
                        lrp_df = load_lrp_csv(model, finetune, color, enc_dec, layer, feature)
                        
                        if lrp_df is None or len(lrp_df) == 0:
                            continue
                        
                        # Vorlayer mIoU-Daten
                        prev_layer = get_prev_layer(layer)
                        miou_dict = {}
                        if prev_layer:
                            miou_dict = get_miou_dict(model, finetune, color, enc_dec, prev_layer)
                        
                        # Teile in positive und negative
                        positive_df, negative_df = split_positive_negative(lrp_df)
                        
                        # Connections für positive Relevanz
                        conn_10_pos = get_connections_for_coverage_positive(positive_df, 0.10)
                        conn_20_pos = get_connections_for_coverage_positive(positive_df, 0.20)
                        conn_50_pos = get_connections_for_coverage_positive(positive_df, 0.50)
                        
                        # Connections für negative Relevanz
                        conn_10_neg = get_connections_for_coverage_negative(negative_df, 0.10)
                        conn_20_neg = get_connections_for_coverage_negative(negative_df, 0.20)
                        conn_50_neg = get_connections_for_coverage_negative(negative_df, 0.50)
                        
                        # Korrelationen mit p-Werten
                        corr_pos, pval_pos, corr_neg, pval_neg = compute_correlation(lrp_df, miou_dict)
                        
                        # Ausreißer-Analyse
                        outliers_pos = compute_outliers(positive_df, 'normalized_relevance')
                        outliers_neg = compute_outliers(negative_df, 'abs_relevance') if len(negative_df) > 0 else {'n_outliers_high': 0, 'n_outliers_low': 0, 'outlier_pct': 0}
                        
                        # Information Bottleneck Metriken
                        bottleneck = compute_information_bottleneck(lrp_df)
                        
                        # Prozent über Threshold bei verschiedenen Coverage-Levels (POSITIVE)
                        features_10_pos = get_features_for_coverage_positive(positive_df, 0.10)
                        features_20_pos = get_features_for_coverage_positive(positive_df, 0.20)
                        features_50_pos = get_features_for_coverage_positive(positive_df, 0.50)
                        
                        pct_10_pos = get_percent_above_threshold_at_coverage(features_10_pos, miou_dict)
                        pct_20_pos = get_percent_above_threshold_at_coverage(features_20_pos, miou_dict)
                        pct_50_pos = get_percent_above_threshold_at_coverage(features_50_pos, miou_dict)
                        
                        # Prozent über Threshold bei verschiedenen Coverage-Levels (NEGATIVE)
                        features_10_neg = get_features_for_coverage_negative(negative_df, 0.10)
                        features_20_neg = get_features_for_coverage_negative(negative_df, 0.20)
                        features_50_neg = get_features_for_coverage_negative(negative_df, 0.50)
                        
                        pct_10_neg = get_percent_above_threshold_at_coverage(features_10_neg, miou_dict)
                        pct_20_neg = get_percent_above_threshold_at_coverage(features_20_neg, miou_dict)
                        pct_50_neg = get_percent_above_threshold_at_coverage(features_50_neg, miou_dict)
                        
                        # Beste Relevanz (positive)
                        best_relevance_pos = positive_df['normalized_relevance'].max() if len(positive_df) > 0 else None
                        best_feature_pos = int(positive_df.loc[positive_df['normalized_relevance'].idxmax(), 'prev_feature_idx']) if len(positive_df) > 0 else None
                        
                        # Stärkste negative Relevanz (als Absolutwert)
                        best_relevance_neg = negative_df['abs_relevance'].max() if len(negative_df) > 0 else None
                        best_feature_neg = int(negative_df.loc[negative_df['abs_relevance'].idxmax(), 'prev_feature_idx']) if len(negative_df) > 0 else None
                        
                        # Spannen
                        relevance_span_pos = positive_df['normalized_relevance'].max() - positive_df['normalized_relevance'].min() if len(positive_df) > 1 else 0
                        relevance_span_neg = negative_df['abs_relevance'].max() - negative_df['abs_relevance'].min() if len(negative_df) > 1 else 0
                        
                        results.append({
                            'model': model,
                            'color': color,
                            'finetune': finetune,
                            'encoder_decoder': enc_dec,
                            'layer': layer,
                            'feature': feature,
                            # Positive
                            'conn_10pct_pos': conn_10_pos,
                            'conn_20pct_pos': conn_20_pos,
                            'conn_50pct_pos': conn_50_pos,
                            'correlation_pos': round(corr_pos, 6) if corr_pos is not None else None,
                            'correlation_pvalue_pos': round(pval_pos, 8) if pval_pos is not None else None,
                            'correlation_significant_pos': pval_pos < 0.05 if pval_pos is not None else None,
                            'pct_above_0.04_at_10pct_pos': round(pct_10_pos, 2) if pct_10_pos is not None else None,
                            'pct_above_0.04_at_20pct_pos': round(pct_20_pos, 2) if pct_20_pos is not None else None,
                            'pct_above_0.04_at_50pct_pos': round(pct_50_pos, 2) if pct_50_pos is not None else None,
                            'best_relevance_pos': round(best_relevance_pos, 8) if best_relevance_pos is not None else None,
                            'best_feature_pos': best_feature_pos,
                            'relevance_span_pos': round(relevance_span_pos, 8) if relevance_span_pos else None,
                            # Ausreißer (Positive)
                            'n_outliers_high_pos': outliers_pos['n_outliers_high'],
                            'n_outliers_low_pos': outliers_pos['n_outliers_low'],
                            'outlier_pct_pos': round(outliers_pos['outlier_pct'], 2),
                            # Information Bottleneck (Positive)
                            'gini_pos': round(bottleneck['gini_pos'], 6) if bottleneck.get('gini_pos') is not None else None,
                            'entropy_pos': round(bottleneck['entropy_pos'], 6) if bottleneck.get('entropy_pos') is not None else None,
                            'effective_features_pos': round(bottleneck['effective_features_pos'], 2) if bottleneck.get('effective_features_pos') is not None else None,
                            'compression_ratio_pos': round(bottleneck['compression_ratio_pos'], 6) if bottleneck.get('compression_ratio_pos') is not None else None,
                            # Negative
                            'conn_10pct_neg': conn_10_neg,
                            'conn_20pct_neg': conn_20_neg,
                            'conn_50pct_neg': conn_50_neg,
                            'correlation_neg': round(corr_neg, 6) if corr_neg is not None else None,
                            'correlation_pvalue_neg': round(pval_neg, 8) if pval_neg is not None else None,
                            'correlation_significant_neg': pval_neg < 0.05 if pval_neg is not None else None,
                            'pct_above_0.04_at_10pct_neg': round(pct_10_neg, 2) if pct_10_neg is not None else None,
                            'pct_above_0.04_at_20pct_neg': round(pct_20_neg, 2) if pct_20_neg is not None else None,
                            'pct_above_0.04_at_50pct_neg': round(pct_50_neg, 2) if pct_50_neg is not None else None,
                            'best_relevance_neg': round(best_relevance_neg, 8) if best_relevance_neg is not None else None,
                            'best_feature_neg': best_feature_neg,
                            'relevance_span_neg': round(relevance_span_neg, 8) if relevance_span_neg else None,
                            # Ausreißer (Negative)
                            'n_outliers_high_neg': outliers_neg['n_outliers_high'],
                            'n_outliers_low_neg': outliers_neg['n_outliers_low'],
                            'outlier_pct_neg': round(outliers_neg['outlier_pct'], 2),
                            # Information Bottleneck (Negative)
                            'gini_neg': round(bottleneck['gini_neg'], 6) if bottleneck.get('gini_neg') is not None else None,
                            'entropy_neg': round(bottleneck['entropy_neg'], 6) if bottleneck.get('entropy_neg') is not None else None,
                            'effective_features_neg': round(bottleneck['effective_features_neg'], 2) if bottleneck.get('effective_features_neg') is not None else None,
                            'compression_ratio_neg': round(bottleneck['compression_ratio_neg'], 6) if bottleneck.get('compression_ratio_neg') is not None else None,
                        })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table7_lrp_per_feature.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def create_table8_lrp_per_layer():
    """
    Tabelle 8: LRP pro Layer
    Pro Modell, pro Farbe, pro Layer
    Mit separaten Werten für positive und negative Relevanz.
    """
    print("Erstelle Tabelle 8: LRP pro Layer...")
    
    results = []
    
    for model in MODELS:
        for color in COLORS:
            for enc_dec in ENCODER_DECODER:
                layers = get_layers(enc_dec)
                
                for layer in layers:
                    prev_layer = get_prev_layer(layer)
                    # Layer 0 hat keinen Vorlayer, aber wir nehmen ihn trotzdem mit auf
                    # (einige Werte werden dann None sein)
                    
                    finetune_data = {ft: {
                        'mious_20pct_pos': [], 'mious_20pct_neg': [],
                        'correlations_pos': [], 'correlations_neg': [],
                        'pvalues_pos': [], 'pvalues_neg': [],
                        'features_10_pos': set(), 'features_10_neg': set(),
                        'features_20_pos': set(), 'features_20_neg': set(),
                        'features_50_pos': set(), 'features_50_neg': set(),
                        # Information Bottleneck
                        'gini_pos': [], 'gini_neg': [],
                        'entropy_pos': [], 'entropy_neg': [],
                        'compression_ratio_pos': [], 'compression_ratio_neg': [],
                        # Ausreißer
                        'outlier_pcts_pos': [], 'outlier_pcts_neg': [],
                    } for ft in FINETUNES}
                    
                    for finetune in FINETUNES:
                        # Bei Layer 0 gibt es keinen Vorlayer -> leeres miou_dict
                        miou_dict = get_miou_dict(model, finetune, color, enc_dec, prev_layer) if prev_layer else {}
                        lrp_files = get_lrp_files(model, finetune, color, enc_dec)
                        layer_lrp_files = [(l, f) for l, f in lrp_files if l == layer]
                        
                        for _, feature in layer_lrp_files:
                            lrp_df = load_lrp_csv(model, finetune, color, enc_dec, layer, feature)
                            if lrp_df is None:
                                continue
                            
                            positive_df, negative_df = split_positive_negative(lrp_df)
                            
                            # Korrelationen mit p-Werten
                            corr_pos, pval_pos, corr_neg, pval_neg = compute_correlation(lrp_df, miou_dict)
                            if corr_pos is not None:
                                finetune_data[finetune]['correlations_pos'].append(corr_pos)
                                finetune_data[finetune]['pvalues_pos'].append(pval_pos)
                            if corr_neg is not None:
                                finetune_data[finetune]['correlations_neg'].append(corr_neg)
                                finetune_data[finetune]['pvalues_neg'].append(pval_neg)
                            
                            # Information Bottleneck
                            bottleneck = compute_information_bottleneck(lrp_df)
                            if bottleneck.get('gini_pos') is not None:
                                finetune_data[finetune]['gini_pos'].append(bottleneck['gini_pos'])
                                finetune_data[finetune]['entropy_pos'].append(bottleneck['entropy_pos'])
                                finetune_data[finetune]['compression_ratio_pos'].append(bottleneck['compression_ratio_pos'])
                            if bottleneck.get('gini_neg') is not None:
                                finetune_data[finetune]['gini_neg'].append(bottleneck['gini_neg'])
                                finetune_data[finetune]['entropy_neg'].append(bottleneck['entropy_neg'])
                                finetune_data[finetune]['compression_ratio_neg'].append(bottleneck['compression_ratio_neg'])
                            
                            # Ausreißer-Analyse
                            positive_df_local, negative_df_local = split_positive_negative(lrp_df)
                            outliers_pos = compute_outliers(positive_df_local, 'normalized_relevance')
                            finetune_data[finetune]['outlier_pcts_pos'].append(outliers_pos['outlier_pct'])
                            if len(negative_df_local) > 0:
                                outliers_neg = compute_outliers(negative_df_local, 'abs_relevance')
                                finetune_data[finetune]['outlier_pcts_neg'].append(outliers_neg['outlier_pct'])
                            
                            # Features für verschiedene Coverage (POSITIVE)
                            f_10_pos = get_features_for_coverage_positive(positive_df, 0.10)
                            f_20_pos = get_features_for_coverage_positive(positive_df, 0.20)
                            f_50_pos = get_features_for_coverage_positive(positive_df, 0.50)
                            
                            finetune_data[finetune]['features_10_pos'].update(f_10_pos)
                            finetune_data[finetune]['features_20_pos'].update(f_20_pos)
                            finetune_data[finetune]['features_50_pos'].update(f_50_pos)
                            
                            # mIoU für 20% Coverage (POSITIVE)
                            for feat in f_20_pos:
                                if feat in miou_dict:
                                    finetune_data[finetune]['mious_20pct_pos'].append(miou_dict[feat])
                            
                            # Features für verschiedene Coverage (NEGATIVE)
                            f_10_neg = get_features_for_coverage_negative(negative_df, 0.10)
                            f_20_neg = get_features_for_coverage_negative(negative_df, 0.20)
                            f_50_neg = get_features_for_coverage_negative(negative_df, 0.50)
                            
                            finetune_data[finetune]['features_10_neg'].update(f_10_neg)
                            finetune_data[finetune]['features_20_neg'].update(f_20_neg)
                            finetune_data[finetune]['features_50_neg'].update(f_50_neg)
                            
                            # mIoU für 20% Coverage (NEGATIVE)
                            for feat in f_20_neg:
                                if feat in miou_dict:
                                    finetune_data[finetune]['mious_20pct_neg'].append(miou_dict[feat])
                    
                    # Berechne Aggregierte Werte
                    # Durchschnittlicher mIoU bei 20% Coverage
                    avg_miou_20_ft1_pos = np.mean(finetune_data['finetune1']['mious_20pct_pos']) if finetune_data['finetune1']['mious_20pct_pos'] else None
                    avg_miou_20_ft2_pos = np.mean(finetune_data['finetune2']['mious_20pct_pos']) if finetune_data['finetune2']['mious_20pct_pos'] else None
                    avg_miou_20_ft3_pos = np.mean(finetune_data['finetune3']['mious_20pct_pos']) if finetune_data['finetune3']['mious_20pct_pos'] else None
                    
                    avg_miou_20_ft1_neg = np.mean(finetune_data['finetune1']['mious_20pct_neg']) if finetune_data['finetune1']['mious_20pct_neg'] else None
                    avg_miou_20_ft2_neg = np.mean(finetune_data['finetune2']['mious_20pct_neg']) if finetune_data['finetune2']['mious_20pct_neg'] else None
                    avg_miou_20_ft3_neg = np.mean(finetune_data['finetune3']['mious_20pct_neg']) if finetune_data['finetune3']['mious_20pct_neg'] else None
                    
                    # Durchschnittliche Korrelation
                    all_corr_pos = []
                    all_corr_neg = []
                    for ft in FINETUNES:
                        all_corr_pos.extend(finetune_data[ft]['correlations_pos'])
                        all_corr_neg.extend(finetune_data[ft]['correlations_neg'])
                    
                    avg_corr_pos = np.mean(all_corr_pos) if all_corr_pos else None
                    corr_span_pos = max(all_corr_pos) - min(all_corr_pos) if len(all_corr_pos) > 1 else None
                    
                    avg_corr_neg = np.mean(all_corr_neg) if all_corr_neg else None
                    corr_span_neg = max(all_corr_neg) - min(all_corr_neg) if len(all_corr_neg) > 1 else None
                    
                    # p-Werte aggregieren
                    all_pval_pos = []
                    all_pval_neg = []
                    for ft in FINETUNES:
                        all_pval_pos.extend(finetune_data[ft]['pvalues_pos'])
                        all_pval_neg.extend(finetune_data[ft]['pvalues_neg'])
                    
                    pct_significant_pos = (sum(1 for p in all_pval_pos if p < 0.05) / len(all_pval_pos) * 100) if all_pval_pos else None
                    pct_significant_neg = (sum(1 for p in all_pval_neg if p < 0.05) / len(all_pval_neg) * 100) if all_pval_neg else None
                    
                    # Information Bottleneck aggregieren
                    all_gini_pos = [g for ft in FINETUNES for g in finetune_data[ft]['gini_pos']]
                    all_gini_neg = [g for ft in FINETUNES for g in finetune_data[ft]['gini_neg']]
                    all_entropy_pos = [e for ft in FINETUNES for e in finetune_data[ft]['entropy_pos']]
                    all_entropy_neg = [e for ft in FINETUNES for e in finetune_data[ft]['entropy_neg']]
                    all_compression_pos = [c for ft in FINETUNES for c in finetune_data[ft]['compression_ratio_pos']]
                    all_compression_neg = [c for ft in FINETUNES for c in finetune_data[ft]['compression_ratio_neg']]
                    
                    avg_gini_pos = np.mean(all_gini_pos) if all_gini_pos else None
                    avg_gini_neg = np.mean(all_gini_neg) if all_gini_neg else None
                    avg_entropy_pos = np.mean(all_entropy_pos) if all_entropy_pos else None
                    avg_entropy_neg = np.mean(all_entropy_neg) if all_entropy_neg else None
                    avg_compression_pos = np.mean(all_compression_pos) if all_compression_pos else None
                    avg_compression_neg = np.mean(all_compression_neg) if all_compression_neg else None
                    
                    # Ausreißer aggregieren
                    all_outlier_pcts_pos = [o for ft in FINETUNES for o in finetune_data[ft]['outlier_pcts_pos']]
                    all_outlier_pcts_neg = [o for ft in FINETUNES for o in finetune_data[ft]['outlier_pcts_neg']]
                    avg_outlier_pct_pos = np.mean(all_outlier_pcts_pos) if all_outlier_pcts_pos else None
                    avg_outlier_pct_neg = np.mean(all_outlier_pcts_neg) if all_outlier_pcts_neg else None
                    
                    # Überlappung zwischen Finetuning-Schritten (POSITIVE)
                    def overlap_pct(set1, set2):
                        if not set1:
                            return 0
                        return len(set1.intersection(set2)) / len(set1) * 100
                    
                    # Positive
                    overlap_10_ft1_ft2_pos = overlap_pct(finetune_data['finetune1']['features_10_pos'], finetune_data['finetune2']['features_10_pos'])
                    overlap_10_ft2_ft3_pos = overlap_pct(finetune_data['finetune2']['features_10_pos'], finetune_data['finetune3']['features_10_pos'])
                    overlap_20_ft1_ft2_pos = overlap_pct(finetune_data['finetune1']['features_20_pos'], finetune_data['finetune2']['features_20_pos'])
                    overlap_20_ft2_ft3_pos = overlap_pct(finetune_data['finetune2']['features_20_pos'], finetune_data['finetune3']['features_20_pos'])
                    overlap_50_ft1_ft2_pos = overlap_pct(finetune_data['finetune1']['features_50_pos'], finetune_data['finetune2']['features_50_pos'])
                    overlap_50_ft2_ft3_pos = overlap_pct(finetune_data['finetune2']['features_50_pos'], finetune_data['finetune3']['features_50_pos'])
                    
                    # Negative
                    overlap_10_ft1_ft2_neg = overlap_pct(finetune_data['finetune1']['features_10_neg'], finetune_data['finetune2']['features_10_neg'])
                    overlap_10_ft2_ft3_neg = overlap_pct(finetune_data['finetune2']['features_10_neg'], finetune_data['finetune3']['features_10_neg'])
                    overlap_20_ft1_ft2_neg = overlap_pct(finetune_data['finetune1']['features_20_neg'], finetune_data['finetune2']['features_20_neg'])
                    overlap_20_ft2_ft3_neg = overlap_pct(finetune_data['finetune2']['features_20_neg'], finetune_data['finetune3']['features_20_neg'])
                    overlap_50_ft1_ft2_neg = overlap_pct(finetune_data['finetune1']['features_50_neg'], finetune_data['finetune2']['features_50_neg'])
                    overlap_50_ft2_ft3_neg = overlap_pct(finetune_data['finetune2']['features_50_neg'], finetune_data['finetune3']['features_50_neg'])
                    
                    # Gleiche Features bei 20% die auch mIoU > 0.04 haben
                    def overlap_with_miou_threshold(set1, set2, miou_dict1, miou_dict2):
                        if not set1:
                            return 0
                        common = set1.intersection(set2)
                        count_both_above = sum(1 for f in common if miou_dict1.get(f, 0) > MIOU_THRESHOLD and miou_dict2.get(f, 0) > MIOU_THRESHOLD)
                        return count_both_above / len(set1) * 100 if set1 else 0
                    
                    # Bei Layer 0 gibt es keinen Vorlayer -> leere miou_dicts
                    miou_dict_ft1 = get_miou_dict(model, 'finetune1', color, enc_dec, prev_layer) if prev_layer else {}
                    miou_dict_ft2 = get_miou_dict(model, 'finetune2', color, enc_dec, prev_layer) if prev_layer else {}
                    miou_dict_ft3 = get_miou_dict(model, 'finetune3', color, enc_dec, prev_layer) if prev_layer else {}
                    
                    # Positive
                    overlap_20_miou_ft1_ft2_pos = overlap_with_miou_threshold(
                        finetune_data['finetune1']['features_20_pos'], 
                        finetune_data['finetune2']['features_20_pos'],
                        miou_dict_ft1, miou_dict_ft2
                    )
                    overlap_20_miou_ft2_ft3_pos = overlap_with_miou_threshold(
                        finetune_data['finetune2']['features_20_pos'], 
                        finetune_data['finetune3']['features_20_pos'],
                        miou_dict_ft2, miou_dict_ft3
                    )
                    
                    # Negative
                    overlap_20_miou_ft1_ft2_neg = overlap_with_miou_threshold(
                        finetune_data['finetune1']['features_20_neg'], 
                        finetune_data['finetune2']['features_20_neg'],
                        miou_dict_ft1, miou_dict_ft2
                    )
                    overlap_20_miou_ft2_ft3_neg = overlap_with_miou_threshold(
                        finetune_data['finetune2']['features_20_neg'], 
                        finetune_data['finetune3']['features_20_neg'],
                        miou_dict_ft2, miou_dict_ft3
                    )
                    
                    results.append({
                        'model': model,
                        'color': color,
                        'encoder_decoder': enc_dec,
                        'layer': layer,
                        # Positive
                        'avg_miou_20pct_ft1_pos': round(avg_miou_20_ft1_pos, 6) if avg_miou_20_ft1_pos is not None else None,
                        'avg_miou_20pct_ft2_pos': round(avg_miou_20_ft2_pos, 6) if avg_miou_20_ft2_pos is not None else None,
                        'avg_miou_20pct_ft3_pos': round(avg_miou_20_ft3_pos, 6) if avg_miou_20_ft3_pos is not None else None,
                        'avg_correlation_pos': round(avg_corr_pos, 6) if avg_corr_pos is not None else None,
                        'correlation_span_pos': round(corr_span_pos, 6) if corr_span_pos is not None else None,
                        'pct_significant_pos': round(pct_significant_pos, 2) if pct_significant_pos is not None else None,
                        'overlap_10pct_ft1_ft2_pos': round(overlap_10_ft1_ft2_pos, 2),
                        'overlap_10pct_ft2_ft3_pos': round(overlap_10_ft2_ft3_pos, 2),
                        'overlap_20pct_ft1_ft2_pos': round(overlap_20_ft1_ft2_pos, 2),
                        'overlap_20pct_ft2_ft3_pos': round(overlap_20_ft2_ft3_pos, 2),
                        'overlap_50pct_ft1_ft2_pos': round(overlap_50_ft1_ft2_pos, 2),
                        'overlap_50pct_ft2_ft3_pos': round(overlap_50_ft2_ft3_pos, 2),
                        'overlap_20pct_miou_ft1_ft2_pos': round(overlap_20_miou_ft1_ft2_pos, 2),
                        'overlap_20pct_miou_ft2_ft3_pos': round(overlap_20_miou_ft2_ft3_pos, 2),
                        # Information Bottleneck (Positive)
                        'avg_gini_pos': round(avg_gini_pos, 6) if avg_gini_pos is not None else None,
                        'avg_entropy_pos': round(avg_entropy_pos, 6) if avg_entropy_pos is not None else None,
                        'avg_compression_ratio_pos': round(avg_compression_pos, 6) if avg_compression_pos is not None else None,
                        # Ausreißer (Positive)
                        'avg_outlier_pct_pos': round(avg_outlier_pct_pos, 2) if avg_outlier_pct_pos is not None else None,
                        # Negative
                        'avg_miou_20pct_ft1_neg': round(avg_miou_20_ft1_neg, 6) if avg_miou_20_ft1_neg is not None else None,
                        'avg_miou_20pct_ft2_neg': round(avg_miou_20_ft2_neg, 6) if avg_miou_20_ft2_neg is not None else None,
                        'avg_miou_20pct_ft3_neg': round(avg_miou_20_ft3_neg, 6) if avg_miou_20_ft3_neg is not None else None,
                        'avg_correlation_neg': round(avg_corr_neg, 6) if avg_corr_neg is not None else None,
                        'correlation_span_neg': round(corr_span_neg, 6) if corr_span_neg is not None else None,
                        'pct_significant_neg': round(pct_significant_neg, 2) if pct_significant_neg is not None else None,
                        'overlap_10pct_ft1_ft2_neg': round(overlap_10_ft1_ft2_neg, 2),
                        'overlap_10pct_ft2_ft3_neg': round(overlap_10_ft2_ft3_neg, 2),
                        'overlap_20pct_ft1_ft2_neg': round(overlap_20_ft1_ft2_neg, 2),
                        'overlap_20pct_ft2_ft3_neg': round(overlap_20_ft2_ft3_neg, 2),
                        'overlap_50pct_ft1_ft2_neg': round(overlap_50_ft1_ft2_neg, 2),
                        'overlap_50pct_ft2_ft3_neg': round(overlap_50_ft2_ft3_neg, 2),
                        'overlap_20pct_miou_ft1_ft2_neg': round(overlap_20_miou_ft1_ft2_neg, 2),
                        'overlap_20pct_miou_ft2_ft3_neg': round(overlap_20_miou_ft2_ft3_neg, 2),
                        # Information Bottleneck (Negative)
                        'avg_gini_neg': round(avg_gini_neg, 6) if avg_gini_neg is not None else None,
                        'avg_entropy_neg': round(avg_entropy_neg, 6) if avg_entropy_neg is not None else None,
                        'avg_compression_ratio_neg': round(avg_compression_neg, 6) if avg_compression_neg is not None else None,
                        # Ausreißer (Negative)
                        'avg_outlier_pct_neg': round(avg_outlier_pct_neg, 2) if avg_outlier_pct_neg is not None else None,
                    })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table8_lrp_per_layer.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def create_table9_lrp_per_color_model():
    """
    Tabelle 9: LRP pro Farbe und Modell
    Pro Modell, pro Farbe
    Mit separaten Werten für positive und negative Relevanz.
    """
    print("Erstelle Tabelle 9: LRP pro Farbe und Modell...")
    
    results = []
    
    for model in MODELS:
        for color in COLORS:
            all_correlations_pos = []
            all_correlations_neg = []
            all_pvalues_pos = []
            all_pvalues_neg = []
            all_gini_pos = []
            all_gini_neg = []
            all_entropy_pos = []
            all_entropy_neg = []
            all_outlier_pcts_pos = []
            all_outlier_pcts_neg = []
            best_corr_pos = None
            best_corr_neg = None
            best_pval_pos = None
            best_pval_neg = None
            best_finetune_pos = None
            best_finetune_neg = None
            best_layer_pos = None
            best_layer_neg = None
            
            for finetune in FINETUNES:
                for enc_dec in ENCODER_DECODER:
                    layers = get_layers(enc_dec)
                    
                    for layer in layers:
                        prev_layer = get_prev_layer(layer)
                        if prev_layer is None:
                            continue
                        
                        miou_dict = get_miou_dict(model, finetune, color, enc_dec, prev_layer)
                        lrp_files = get_lrp_files(model, finetune, color, enc_dec)
                        layer_lrp_files = [(l, f) for l, f in lrp_files if l == layer]
                        
                        for _, feature in layer_lrp_files:
                            lrp_df = load_lrp_csv(model, finetune, color, enc_dec, layer, feature)
                            if lrp_df is None:
                                continue
                            
                            corr_pos, pval_pos, corr_neg, pval_neg = compute_correlation(lrp_df, miou_dict)
                            
                            if corr_pos is not None:
                                all_correlations_pos.append(corr_pos)
                                all_pvalues_pos.append(pval_pos)
                                if best_corr_pos is None or corr_pos > best_corr_pos:
                                    best_corr_pos = corr_pos
                                    best_pval_pos = pval_pos
                                    best_finetune_pos = finetune
                                    best_layer_pos = f"{enc_dec}/{layer}"
                            
                            if corr_neg is not None:
                                all_correlations_neg.append(corr_neg)
                                all_pvalues_neg.append(pval_neg)
                                if best_corr_neg is None or corr_neg > best_corr_neg:
                                    best_corr_neg = corr_neg
                                    best_pval_neg = pval_neg
                                    best_finetune_neg = finetune
                                    best_layer_neg = f"{enc_dec}/{layer}"
                            
                            # Information Bottleneck
                            bottleneck = compute_information_bottleneck(lrp_df)
                            if bottleneck.get('gini_pos') is not None:
                                all_gini_pos.append(bottleneck['gini_pos'])
                                all_entropy_pos.append(bottleneck['entropy_pos'])
                            if bottleneck.get('gini_neg') is not None:
                                all_gini_neg.append(bottleneck['gini_neg'])
                                all_entropy_neg.append(bottleneck['entropy_neg'])
                            
                            # Ausreißer
                            positive_df_local, negative_df_local = split_positive_negative(lrp_df)
                            outliers_pos = compute_outliers(positive_df_local, 'normalized_relevance')
                            all_outlier_pcts_pos.append(outliers_pos['outlier_pct'])
                            if len(negative_df_local) > 0:
                                outliers_neg = compute_outliers(negative_df_local, 'abs_relevance')
                                all_outlier_pcts_neg.append(outliers_neg['outlier_pct'])
            
            # Aggregierte Metriken berechnen
            pct_significant_pos = (sum(1 for p in all_pvalues_pos if p < 0.05) / len(all_pvalues_pos) * 100) if all_pvalues_pos else None
            pct_significant_neg = (sum(1 for p in all_pvalues_neg if p < 0.05) / len(all_pvalues_neg) * 100) if all_pvalues_neg else None
            
            results.append({
                'model': model,
                'color': color,
                # Positive
                'avg_correlation_pos': round(np.mean(all_correlations_pos), 6) if all_correlations_pos else None,
                'correlation_span_pos': round(max(all_correlations_pos) - min(all_correlations_pos), 6) if len(all_correlations_pos) > 1 else None,
                'best_correlation_pos': round(best_corr_pos, 6) if best_corr_pos is not None else None,
                'best_correlation_pvalue_pos': round(best_pval_pos, 8) if best_pval_pos is not None else None,
                'pct_significant_pos': round(pct_significant_pos, 2) if pct_significant_pos is not None else None,
                'best_finetune_pos': best_finetune_pos,
                'best_layer_pos': best_layer_pos,
                # Information Bottleneck (Positive)
                'avg_gini_pos': round(np.mean(all_gini_pos), 6) if all_gini_pos else None,
                'avg_entropy_pos': round(np.mean(all_entropy_pos), 6) if all_entropy_pos else None,
                'gini_span_pos': round(max(all_gini_pos) - min(all_gini_pos), 6) if len(all_gini_pos) > 1 else None,
                # Ausreißer (Positive)
                'avg_outlier_pct_pos': round(np.mean(all_outlier_pcts_pos), 2) if all_outlier_pcts_pos else None,
                # Negative
                'avg_correlation_neg': round(np.mean(all_correlations_neg), 6) if all_correlations_neg else None,
                'correlation_span_neg': round(max(all_correlations_neg) - min(all_correlations_neg), 6) if len(all_correlations_neg) > 1 else None,
                'best_correlation_neg': round(best_corr_neg, 6) if best_corr_neg is not None else None,
                'best_correlation_pvalue_neg': round(best_pval_neg, 8) if best_pval_neg is not None else None,
                'pct_significant_neg': round(pct_significant_neg, 2) if pct_significant_neg is not None else None,
                'best_finetune_neg': best_finetune_neg,
                'best_layer_neg': best_layer_neg,
                # Information Bottleneck (Negative)
                'avg_gini_neg': round(np.mean(all_gini_neg), 6) if all_gini_neg else None,
                'avg_entropy_neg': round(np.mean(all_entropy_neg), 6) if all_entropy_neg else None,
                'gini_span_neg': round(max(all_gini_neg) - min(all_gini_neg), 6) if len(all_gini_neg) > 1 else None,
                # Ausreißer (Negative)
                'avg_outlier_pct_neg': round(np.mean(all_outlier_pcts_neg), 2) if all_outlier_pcts_neg else None,
            })
    
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH / "table9_lrp_per_color_model.csv"
    df_results.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"  -> Gespeichert: {output_file} ({len(df_results)} Zeilen)")
    return df_results


def main():
    """Hauptfunktion - erstellt alle 3 LRP-Tabellen."""
    print("=" * 60)
    print("LRP Datenanalyse - Erstellung der Analysetabellen")
    print("=" * 60)
    print(f"Datenquelle: {BASE_PATH}")
    print(f"Ausgabe: {OUTPUT_PATH}")
    print(f"mIoU Schwellenwert: {MIOU_THRESHOLD}")
    print("HINWEIS: Positive und negative Relevanz werden separat behandelt!")
    print("=" * 60)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    table7 = create_table7_lrp_per_feature()
    table8 = create_table8_lrp_per_layer()
    table9 = create_table9_lrp_per_color_model()
    
    print("=" * 60)
    print("Alle LRP-Tabellen erfolgreich erstellt!")
    print("=" * 60)
    
    print("\nZusammenfassung:")
    print(f"  Tabelle 7 (LRP pro Feature):       {len(table7)} Zeilen")
    print(f"  Tabelle 8 (LRP pro Layer):         {len(table8)} Zeilen")
    print(f"  Tabelle 9 (LRP pro Farbe/Modell):  {len(table9)} Zeilen")


if __name__ == "__main__":
    main()
