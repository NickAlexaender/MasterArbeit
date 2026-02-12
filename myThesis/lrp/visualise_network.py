import argparse
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd

# Configure matplotlib for scientific publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'text.usetex': False,  # Set True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Layer configuration for different modules
LAYER_CONFIG = {
    "encoder": {
        "num_data_layers": 6,  # Layer 0-5 in data
        "layer_names": ["Pre-Layer", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6"],
        "num_visual_layers": 7,  # Including pre-layer
    },
    "decoder": {
        "num_data_layers": 3,  # Layer 0-2 in data
        "layer_names": ["Pre-Layer", "Layer 1", "Layer 2", "Layer 3"],
        "num_visual_layers": 4,  # Including pre-layer
    },
}

# External data paths (for analysis tables)
BASE_PATH = Path("/Volumes/Untitled/Master-Arbeit_Ergebnisse/output")
LOCAL_OUTPUT_PATH = Path(os.path.dirname(__file__)) / ".." / "output" / "data_analysis"


@dataclass
class LayerStatistics:
    """Statistics for a single layer from various analyses."""
    layer_idx: int
    layer_name: str
    mean_miou: float = 0.0
    variance_miou: float = 0.0
    count_above_threshold: int = 0
    best_miou: float = 0.0
    best_feature: int = -1
    # Linear probing metrics
    linear_probe_precision: Optional[float] = None
    # Information bottleneck metrics  
    gini_coefficient: Optional[float] = None
    effective_features: Optional[float] = None
    compression_ratio: Optional[float] = None
    # Coverage metrics (how many features for X% relevance)
    coverage_50_pct: Optional[int] = None
    # Distribution data
    miou_distribution: Optional[np.ndarray] = None


@dataclass
class NeuralUnit:
    """Represents a neural unit (feature/query) with its IoU value."""
    layer_idx: int  # data layer index (0-based)
    feature_idx: int  # 0..255
    miou: float
    is_top_feature: bool = False  # Marked as top feature (red node)
    # Additional metrics from LRP analysis
    gini: Optional[float] = None
    effective_features: Optional[float] = None


@dataclass
class TopFeature:
    layer_idx: int  # encoder data layer index (0-based)
    feature_idx: int  # 0..255
    miou: float
    # Coverage info
    coverage_50: int = 0  # Features needed for 50% relevance


@dataclass
class Edge:
    from_layer_vis: int  # visual layer index
    to_layer_vis: int  # visual layer index  
    from_feature: int  # 0..255
    to_feature: int  # 0..255
    relevance: float
    normalized_relevance: float = 0.0


# ============================================================================
# mIoU Loading Functions (same approach as analyze_lrp_tables.py)
# ============================================================================

def load_miou_csv_external(
    model: str, 
    finetune: str, 
    color: str, 
    enc_dec: str, 
    layer: str
) -> Optional[pd.DataFrame]:
    """
    Load mIoU CSV file from external storage.
    Uses same path structure as analyze_lrp_tables.py.
    
    Path: BASE_PATH / model / finetune / color / enc_dec / layer / miou_network_dissection.csv
    """
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
            print(f"Warning: Could not load {csv_path}: {e}")
            return None
    return None


def get_miou_dict(
    model: str, 
    finetune: str, 
    color: str, 
    enc_dec: str, 
    layer: str
) -> Dict[int, float]:
    """
    Create a dictionary feature_idx -> mIoU for a layer.
    Same approach as analyze_lrp_tables.py.
    """
    df = load_miou_csv_external(model, finetune, color, enc_dec, layer)
    if df is not None:
        return dict(zip(df['feature_idx'].astype(int), df['miou']))
    return {}


def load_all_neural_units_external(
    model: str,
    finetune: str,
    color: str,
    module: str,
    miou_threshold: float = 0.04
) -> Dict[int, List[NeuralUnit]]:
    """
    Load all neural units from external mIoU CSV files.
    Uses same path structure as analyze_lrp_tables.py.
    
    Returns a dict mapping layer_idx -> list of NeuralUnits above threshold.
    """
    module = module.lower()
    config = LAYER_CONFIG.get(module, LAYER_CONFIG["encoder"])
    num_layers = config["num_data_layers"]
    
    units_by_layer: Dict[int, List[NeuralUnit]] = {}
    
    for layer_idx in range(num_layers):
        layer_name = f"layer{layer_idx}"
        
        df = load_miou_csv_external(model, finetune, color, module, layer_name)
        if df is None:
            continue
        
        units: List[NeuralUnit] = []
        for _, row in df.iterrows():
            try:
                feat_idx = int(row['feature_idx'])
                miou_val = float(row['miou'])
                if miou_val >= miou_threshold:
                    units.append(NeuralUnit(
                        layer_idx=layer_idx,
                        feature_idx=feat_idx,
                        miou=miou_val,
                        is_top_feature=False,
                    ))
            except Exception:
                continue
        
        if units:
            units_by_layer[layer_idx] = units
    
    return units_by_layer


def load_layer_statistics_external(
    model: str,
    finetune: str,
    color: str,
    module: str,
    miou_threshold: float = 0.04
) -> Dict[int, LayerStatistics]:
    """
    Load comprehensive layer statistics from external storage.
    Uses same path structure as analyze_lrp_tables.py.
    """
    config = LAYER_CONFIG.get(module, LAYER_CONFIG["encoder"])
    layer_names = config["layer_names"]
    num_layers = config["num_data_layers"]
    
    stats_by_layer: Dict[int, LayerStatistics] = {}
    
    for layer_idx in range(num_layers):
        layer_name = f"layer{layer_idx}"
        
        layer_stat = LayerStatistics(
            layer_idx=layer_idx,
            layer_name=layer_names[layer_idx + 1] if layer_idx + 1 < len(layer_names) else f"Layer {layer_idx + 1}"
        )
        
        df = load_miou_csv_external(model, finetune, color, module, layer_name)
        if df is not None and len(df) > 0:
            miou_values = df['miou'].values
            
            layer_stat.mean_miou = float(np.mean(miou_values))
            layer_stat.variance_miou = float(np.var(miou_values))
            layer_stat.count_above_threshold = int(np.sum(miou_values > miou_threshold))
            layer_stat.best_miou = float(np.max(miou_values))
            layer_stat.best_feature = int(df.iloc[np.argmax(miou_values)]['feature_idx'])
            layer_stat.miou_distribution = miou_values
        
        # Try to load linear probing precision
        try:
            lp_path = BASE_PATH / model / finetune / "linear_probing" / f"{module}_results" / layer_name
            if lp_path.exists():
                json_files = list(lp_path.glob("linear_probe_results_*.json"))
                if json_files:
                    with open(sorted(json_files)[0], 'r') as f:
                        lp_data = json.load(f)
                        if "overall_accuracy" in lp_data:
                            layer_stat.linear_probe_precision = lp_data["overall_accuracy"]
        except Exception:
            pass
        
        stats_by_layer[layer_idx] = layer_stat
    
    return stats_by_layer


# ============================================================================
# Legacy Local Loading Functions (for backwards compatibility)
# ============================================================================

def load_all_neural_units(
    miou_base_dir: str, 
    module: str, 
    miou_threshold: float = 0.04
) -> Dict[int, List[NeuralUnit]]:
    """
    Load all neural units from mIoU CSV files.
    Returns a dict mapping layer_idx -> list of NeuralUnits above threshold.
    """
    module = module.lower()
    units_by_layer: Dict[int, List[NeuralUnit]] = {}
    
    if not os.path.isdir(miou_base_dir):
        print(f"Warning: mIoU directory not found: {miou_base_dir}")
        return units_by_layer
    
    # Find layer directories
    for name in os.listdir(miou_base_dir):
        if not name.startswith("layer"):
            continue
        try:
            layer_idx = int(name.replace("layer", ""))
        except ValueError:
            continue
            
        layer_dir = os.path.join(miou_base_dir, name)
        if not os.path.isdir(layer_dir):
            continue
        
        # Determine CSV filename based on module
        if module == "encoder":
            csv_name = "miou_network_dissection.csv"
            idx_col = "feature_idx"
            miou_col = "miou"
        else:  # decoder
            csv_name = "mIoU_per_Query.csv"
            idx_col = "query_idx"
            miou_col = "mean_iou"
        
        csv_path = os.path.join(layer_dir, csv_name)
        if not os.path.isfile(csv_path):
            continue
        
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        
        units: List[NeuralUnit] = []
        for _, row in df.iterrows():
            try:
                feat_idx = int(row[idx_col])
                miou_val = float(row.get(miou_col, 0))
                if miou_val >= miou_threshold:
                    units.append(NeuralUnit(
                        layer_idx=layer_idx,
                        feature_idx=feat_idx,
                        miou=miou_val,
                        is_top_feature=False,
                    ))
            except Exception:
                continue
        
        if units:
            units_by_layer[layer_idx] = units
    
    return units_by_layer


def load_layer_statistics(
    miou_base_dir: str,
    module: str,
    model: str = "car",
    finetune: str = "finetune3",
    color: str = "grau",
    miou_threshold: float = 0.04
) -> Dict[int, LayerStatistics]:
    """
    Load comprehensive statistics for each layer including mIoU distribution,
    linear probing precision, and information bottleneck metrics.
    """
    config = LAYER_CONFIG.get(module, LAYER_CONFIG["encoder"])
    layer_names = config["layer_names"]
    num_layers = config["num_data_layers"]
    
    stats_by_layer: Dict[int, LayerStatistics] = {}
    
    # Load mIoU data for each layer
    for layer_idx in range(num_layers):
        layer_name = f"layer{layer_idx}"
        
        if module == "encoder":
            csv_path = os.path.join(miou_base_dir, layer_name, "miou_network_dissection.csv")
            idx_col = "feature_idx"
            miou_col = "miou"
        else:
            csv_path = os.path.join(miou_base_dir, layer_name, "mIoU_per_Query.csv")
            idx_col = "query_idx"
            miou_col = "mean_iou"
        
        layer_stat = LayerStatistics(
            layer_idx=layer_idx,
            layer_name=layer_names[layer_idx + 1] if layer_idx + 1 < len(layer_names) else f"Layer {layer_idx + 1}"
        )
        
        if os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path)
                miou_values = df[miou_col].values
                
                layer_stat.mean_miou = float(np.mean(miou_values))
                layer_stat.variance_miou = float(np.var(miou_values))
                layer_stat.count_above_threshold = int(np.sum(miou_values > miou_threshold))
                layer_stat.best_miou = float(np.max(miou_values))
                layer_stat.best_feature = int(df.iloc[np.argmax(miou_values)][idx_col])
                layer_stat.miou_distribution = miou_values
            except Exception as e:
                print(f"Warning: Could not load mIoU for {layer_name}: {e}")
        
        # Try to load linear probing precision from external data
        try:
            lp_path = BASE_PATH / model / finetune / "linear_probing" / f"{module}_results" / layer_name
            if lp_path.exists():
                json_files = list(lp_path.glob("linear_probe_results_*.json"))
                if json_files:
                    with open(sorted(json_files)[0], 'r') as f:
                        lp_data = json.load(f)
                        if "overall_accuracy" in lp_data:
                            layer_stat.linear_probe_precision = lp_data["overall_accuracy"]
        except Exception:
            pass
        
        stats_by_layer[layer_idx] = layer_stat
    
    return stats_by_layer


def load_top_features(top_features_csv: str, module: str) -> List[TopFeature]:
    """Load top features from the summary CSV."""
    if not os.path.isfile(top_features_csv):
        print(f"Warning: Top features CSV not found: {top_features_csv}")
        return []
    
    df = pd.read_csv(top_features_csv)
    df = df[df["module"].str.lower() == module.lower()]
    top_features: List[TopFeature] = []
    for _, row in df.iterrows():
        try:
            top_features.append(
                TopFeature(
                    layer_idx=int(row["layer_idx"]),
                    feature_idx=int(row["feature_idx"]),
                    miou=float(row["miou"]),
                )
            )
        except Exception:
            continue
    return top_features


def map_layer_to_visual(module: str, layer_idx: int) -> int:
    """
    Map data layer index to visual layer index.
    Data layer 0 -> Visual layer 2 (first actual layer after pre-layer)
    Pre-layer is visual layer 1.
    """
    # Visual layers: 1=Pre-Layer, 2=Layer1, 3=Layer2, etc.
    return layer_idx + 2


def get_layer_feature_csv(root_dir: str, layer_idx: int, feature_idx: int) -> str:
    """Get path to LRP relevance CSV for a specific feature."""
    return os.path.join(root_dir, f"layer{layer_idx}_feat{feature_idx}.csv")


def split_positive_negative(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into positive and negative normalized_relevance values.
    Returns (positive_df, negative_df).
    Same logic as analyze_lrp_tables.py.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Check which column to use
    rel_col = 'normalized_relevance' if 'normalized_relevance' in df.columns else 'relevance'
    
    positive_df = df[df[rel_col] > 0].copy()
    negative_df = df[df[rel_col] < 0].copy()
    
    # Sort positive descending by relevance
    positive_df = positive_df.sort_values(rel_col, ascending=False).reset_index(drop=True)
    
    # Sort negative by absolute value descending (i.e., by value ascending since negative)
    negative_df = negative_df.sort_values(rel_col, ascending=True).reset_index(drop=True)
    # Add absolute value column for easier calculation
    negative_df['abs_relevance'] = negative_df[rel_col].abs()
    
    return positive_df, negative_df


def get_connections_for_coverage_positive(df: pd.DataFrame, coverage: float) -> int:
    """
    Calculate number of positive connections for a given coverage.
    df should already be sorted by normalized_relevance descending.
    Same logic as analyze_lrp_tables.py.
    """
    if df is None or len(df) == 0:
        return 0
    
    rel_col = 'normalized_relevance' if 'normalized_relevance' in df.columns else 'relevance'
    
    total = df[rel_col].sum()
    if total == 0:
        return len(df)
    
    cumsum = df[rel_col].cumsum()
    threshold = coverage * total
    
    connections_needed = (cumsum >= threshold).idxmax() + 1 if (cumsum >= threshold).any() else len(df)
    return int(connections_needed)


def get_connections_for_coverage_negative(df: pd.DataFrame, coverage: float) -> int:
    """
    Calculate number of negative connections for a given coverage.
    df should already be sorted by abs_relevance descending.
    Same logic as analyze_lrp_tables.py.
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


def load_coverage_edges_for_node(
    csv_path: str, coverage: float = 0.10
) -> List[Tuple[int, float, bool]]:
    """
    Load edges that cover X% of positive and X% of negative relevance.
    Returns list of (prev_feature_idx, relevance, is_positive).
    Same logic as analyze_lrp_tables.py (conn_10pct_pos, conn_10pct_neg).
    """
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    # Check which column to use for feature index
    feat_col = 'prev_feature_idx' if 'prev_feature_idx' in df.columns else None
    rel_col = 'normalized_relevance' if 'normalized_relevance' in df.columns else 'relevance'
    
    if feat_col is None or rel_col not in df.columns:
        return []

    # Split into positive and negative
    positive_df, negative_df = split_positive_negative(df)
    
    pairs: List[Tuple[int, float, bool]] = []
    
    # Get positive connections for coverage
    n_pos = get_connections_for_coverage_positive(positive_df, coverage)
    for _, row in positive_df.head(n_pos).iterrows():
        try:
            pairs.append((int(row[feat_col]), float(row[rel_col]), True))
        except Exception:
            continue
    
    # Get negative connections for coverage
    n_neg = get_connections_for_coverage_negative(negative_df, coverage)
    for _, row in negative_df.head(n_neg).iterrows():
        try:
            pairs.append((int(row[feat_col]), float(row[rel_col]), False))
        except Exception:
            continue
    
    return pairs


def load_top_k_edges_for_node(
    csv_path: str, k: int
) -> List[Tuple[int, float]]:
    """Load top-k relevance connections from previous layer for a node (legacy)."""
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    feat_col = 'prev_feature_idx' if 'prev_feature_idx' in df.columns else None
    rel_col = 'normalized_relevance' if 'normalized_relevance' in df.columns else 'relevance'
    
    if feat_col is None or rel_col not in df.columns:
        return []

    # Sort by absolute relevance descending
    df_sorted = df.sort_values(by=rel_col, key=lambda s: s.abs(), ascending=False)
    df_top = df_sorted.head(k)
    pairs: List[Tuple[int, float]] = []
    for _, row in df_top.iterrows():
        try:
            pairs.append((int(row[feat_col]), float(row[rel_col])))
        except Exception:
            continue
    return pairs


def build_edges(
    root_dir: str, top_nodes: List[TopFeature], coverage: float, module: str
) -> List[Edge]:
    """
    Build edge list from top features using coverage-based selection.
    Uses conn_10pct_pos and conn_10pct_neg logic from analyze_lrp_tables.py.
    
    Args:
        root_dir: Directory with LRP CSV files
        top_nodes: List of top features to trace
        coverage: Coverage percentage (0.10 = 10%)
        module: 'encoder' or 'decoder'
    """
    edges: List[Edge] = []
    for tf in top_nodes:
        to_layer_vis = map_layer_to_visual(module, tf.layer_idx)
        from_layer_vis = to_layer_vis - 1
        csv_path = get_layer_feature_csv(root_dir, tf.layer_idx, tf.feature_idx)
        pairs = load_coverage_edges_for_node(csv_path, coverage)
        for prev_feature, rel, is_positive in pairs:
            edges.append(
                Edge(
                    from_layer_vis=from_layer_vis,
                    to_layer_vis=to_layer_vis,
                    from_feature=prev_feature,
                    to_feature=tf.feature_idx,
                    relevance=rel,
                )
            )
    return edges


def compute_layout(
    num_layers: int,
    features_per_layer: int = 256,
    horizontal_spacing: float = 1.0,
    vertical_spacing: float = 1.5,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Compute node positions for the VERTICAL network graph.
    X-axis: features (0-255), Y-axis: layers (bottom to top).
    """
    coords: Dict[Tuple[int, int], Tuple[float, float]] = {}
    
    for layer in range(1, num_layers + 1):
        # Y position: bottom to top (layer 1 at bottom)
        y = (layer - 1) / (num_layers - 1) if num_layers > 1 else 0.5
        for feat in range(features_per_layer):
            # X position: left to right
            x = feat / (features_per_layer - 1) if features_per_layer > 1 else 0.5
            coords[(layer, feat)] = (x, y)
    
    return coords


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for inequality measure (0=equal, 1=concentrated)."""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(np.abs(values))
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_coverage_features(relevances: np.ndarray, coverage: float = 0.5) -> int:
    """Compute how many top features are needed for X% of total relevance."""
    if len(relevances) == 0:
        return 0
    abs_rel = np.abs(relevances)
    sorted_idx = np.argsort(abs_rel)[::-1]
    sorted_rel = abs_rel[sorted_idx]
    cumsum = np.cumsum(sorted_rel)
    total = cumsum[-1]
    if total == 0:
        return len(relevances)
    threshold = coverage * total
    count = np.searchsorted(cumsum, threshold) + 1
    return min(count, len(relevances))


def draw_visualisation(
    output_path_base: str,
    top_nodes: List[TopFeature],
    edges: List[Edge],
    all_units: Dict[int, List[NeuralUnit]],
    num_layers: int,
    features_per_layer: int = 256,
    base_node_size: float = 4.0,
    fig_width_in: float = 16.0,
    fig_height_in: float = 20.0,
    module: str = "encoder",
    miou_threshold: float = 0.04,
    layer_stats: Optional[Dict[int, LayerStatistics]] = None,
):
    """
    Draw a publication-quality VERTICAL network visualisation.
    
    Layout:
    - Vertical network with layers stacked bottom-to-top
    - Features arranged horizontally (0-255)
    
    Features:
    - Nodes colored by mIoU value (viridis colormap)
    - Edge thickness by relevance magnitude
    - Edge color: red=positive, blue=negative relevance
    """
    from collections import defaultdict
    
    config = LAYER_CONFIG.get(module, LAYER_CONFIG["encoder"])
    layer_names = config["layer_names"]
    
    # Create figure - simple single axis
    fig, ax_main = plt.subplots(figsize=(fig_width_in, fig_height_in))
    
    # === Compute Layout ===
    coords = compute_layout(
        num_layers=num_layers,
        features_per_layer=features_per_layer,
    )
    
    # Build lookup for units by (layer_vis, feature_idx)
    units_lookup: Dict[Tuple[int, int], NeuralUnit] = {}
    for layer_idx, units in all_units.items():
        vis_layer = map_layer_to_visual(module, layer_idx)
        for unit in units:
            units_lookup[(vis_layer, unit.feature_idx)] = unit
    
    # Mark top features
    top_feature_set = {(map_layer_to_visual(module, tf.layer_idx), tf.feature_idx): tf.miou 
                       for tf in top_nodes}
    
    # === Style Main Axis ===
    ax_main.set_aspect('auto')
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_facecolor('#f8f9fa')
    
    # === Create colormap for mIoU values ===
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=0.3)  # mIoU typically 0-0.3
    
    # === Draw Edges First (behind nodes) ===
    edges_by_target: Dict[Tuple[int, int], List[Edge]] = defaultdict(list)
    for e in edges:
        edges_by_target[(e.to_layer_vis, e.to_feature)].append(e)
    
    # Collect all edges for LineCollection (more efficient)
    pos_segments, pos_colors, pos_widths = [], [], []
    neg_segments, neg_colors, neg_widths = [], [], []
    
    for (to_layer, to_feat), e_list in edges_by_target.items():
        abs_vals = np.array([abs(e.relevance) for e in e_list], dtype=float)
        if len(abs_vals) == 0:
            continue
        max_abs = abs_vals.max() if abs_vals.max() > 0 else 1.0
        
        for e in e_list:
            coord_from = coords.get((e.from_layer_vis, e.from_feature))
            coord_to = coords.get((e.to_layer_vis, e.to_feature))
            if coord_from is None or coord_to is None:
                continue
            
            rel_strength = abs(e.relevance) / max_abs
            lw = 0.5 + 3.0 * float(rel_strength)
            alpha = 0.3 + 0.5 * rel_strength
            
            if e.relevance >= 0:
                pos_segments.append([coord_from, coord_to])
                pos_colors.append((*mcolors.to_rgb('#C41E3A'), alpha))
                pos_widths.append(lw)
            else:
                neg_segments.append([coord_from, coord_to])
                neg_colors.append((*mcolors.to_rgb('#1E90FF'), alpha))
                neg_widths.append(lw)
    
    # Draw edges as LineCollections
    if pos_segments:
        lc_pos = LineCollection(pos_segments, colors=pos_colors, linewidths=pos_widths, zorder=1)
        ax_main.add_collection(lc_pos)
    if neg_segments:
        lc_neg = LineCollection(neg_segments, colors=neg_colors, linewidths=neg_widths, zorder=1)
        ax_main.add_collection(lc_neg)
    
    # === Draw Nodes Layer by Layer ===
    color_inactive = '#e0e0e0'
    color_top_edge = '#8B0000'
    
    for layer in range(1, num_layers + 1):
        xs, ys, sizes, colors, edgecolors = [], [], [], [], []
        
        for feat in range(features_per_layer):
            if (layer, feat) not in coords:
                continue
            x, y = coords[(layer, feat)]
            xs.append(x)
            ys.append(y)
            
            is_top = (layer, feat) in top_feature_set
            unit = units_lookup.get((layer, feat))
            
            if is_top:
                miou = top_feature_set[(layer, feat)]
                scale = 1.5 + miou * 6.0
                sizes.append(base_node_size * scale)
                colors.append('#E74C3C')  # Bright red for top features
                edgecolors.append(color_top_edge)
            elif unit is not None and unit.miou >= miou_threshold:
                # Color by mIoU value using colormap
                scale = 0.8 + unit.miou * 4.0
                sizes.append(base_node_size * scale)
                colors.append(cmap(norm(unit.miou)))
                edgecolors.append('#404040')
            else:
                sizes.append(base_node_size * 0.2)
                colors.append(color_inactive)
                edgecolors.append('none')
        
        if not xs:
            continue
        
        s = (np.array(sizes) ** 2).astype(float)
        ax_main.scatter(xs, ys, s=s, c=colors, edgecolors=edgecolors,
                       linewidths=0.4, zorder=2 + layer)
    
    # === Layer Labels on Main Plot ===
    for layer in range(1, num_layers + 1):
        y = (layer - 1) / (num_layers - 1) if num_layers > 1 else 0.5
        label = layer_names[layer - 1] if layer - 1 < len(layer_names) else f"Layer {layer}"
        ax_main.text(
            -0.02, y, label,
            va='center', ha='right',
            fontsize=11, fontweight='bold',
            color='#2c3e50',
            transform=ax_main.transAxes,
        )
    
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(-0.05, 1.05)
    
    # === Title ===
    title = f"{module.capitalize()} Network Architecture"
    subtitle = f"Vertical Layout • {features_per_layer} Features/Layer • mIoU Threshold: {miou_threshold}"
    ax_main.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', 
                     color='#2c3e50', pad=15)
    
    # === Legend ===
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='Top Features', markeredgecolor='#8B0000', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.7),
               markersize=8, label='High mIoU', markeredgecolor='#404040'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.3),
               markersize=6, label='Medium mIoU', markeredgecolor='#404040'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0e0e0',
               markersize=4, label='Inactive', markeredgecolor='none'),
        Line2D([0], [0], color='#C41E3A', linewidth=2.5, label='Positive Relevance'),
        Line2D([0], [0], color='#1E90FF', linewidth=2.5, label='Negative Relevance'),
    ]
    ax_main.legend(handles=legend_elements, loc='lower right', framealpha=0.95,
                  edgecolor='#bdc3c7', fontsize=9, ncol=2, 
                  bbox_to_anchor=(1.0, -0.02))
    
    # === Colorbar for mIoU ===
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.02, 0.02, 0.25, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('mIoU Value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # === Save outputs ===
    fig.patch.set_facecolor('white')
    os.makedirs(os.path.dirname(output_path_base) if os.path.dirname(output_path_base) else '.', exist_ok=True)
    png_path = output_path_base + ".png"
    svg_path = output_path_base + ".svg"
    pdf_path = output_path_base + ".pdf"
    
    plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return png_path, svg_path, pdf_path


def main(
    module: str = "encoder",
    model: str = "car",
    finetune: str = "finetune3",
    color: str = "grau",
    top_features: str = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "car",
        "finetune6",
        "lrp",
        "top_features.csv",
    ),
    encoder_dir: str = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "car",
        "finetune6",
        "lrp",
        "encoder",
    ),
    decoder_dir: str = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "car",
        "finetune6",
        "lrp",
        "decoder",
    ),
    encoder_miou_dir: str = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "car",
        "finetune6",
        "encoder",
        "rot",
    ),
    decoder_miou_dir: str = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "car",
        "finetune6",
        "decoder",
    ),
    out: str = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "visualisations",
        "encoder_graph",
    ),
    coverage: float = 0.10,
    miou_threshold: float = 0.04,
    base_node_size: float = 4.0,
    fig_width: float = 16.0,
    fig_height: float = 20.0,
):
    """
    Main function to generate publication-quality VERTICAL network visualisation.
    
    This enhanced version includes:
    - Vertical layout (layers stacked bottom-to-top)
    - mIoU-based node coloring with viridis colormap
    - Side panels with layer statistics and mIoU distributions
    - Efficient edge rendering with LineCollections
    
    Args:
        module: 'encoder' or 'decoder'
        model: Model name ('car' or 'butterfly') for loading external stats
        finetune: Finetuning step ('finetune1', 'finetune2', 'finetune3')
        color: Color condition ('grau', 'orange', 'blau')
        top_features: Path to CSV with top features
        encoder_dir: Directory with encoder LRP CSV files
        decoder_dir: Directory with decoder LRP CSV files
        encoder_miou_dir: Directory with encoder mIoU results
        decoder_miou_dir: Directory with decoder mIoU results
        out: Output path base (without extension)
        coverage: Coverage percentage for edge selection (0.10 = 10% of relevance)
        miou_threshold: Threshold for showing units as active
        base_node_size: Base size for nodes
        fig_width: Figure width in inches
        fig_height: Figure height in inches
    """
    module = module.lower()
    config = LAYER_CONFIG.get(module, LAYER_CONFIG["encoder"])
    num_layers = config["num_visual_layers"]
    
    print(f"\n{'='*60}")
    print(f"🎨 Generating Vertical Network Visualisation")
    print(f"{'='*60}")
    print(f"   Module:      {module}")
    print(f"   Model:       {model}")
    print(f"   Finetune:    {finetune}")
    print(f"   Color:       {color}")
    print(f"   Coverage:    {coverage*100:.0f}%")
    print(f"   Threshold:   {miou_threshold}")
    print(f"{'='*60}\n")
    
    # Load top features for highlighting
    top_nodes = load_top_features(top_features, module)
    print(f"📊 Loaded {len(top_nodes)} top features for {module}")
    
    # Filter to valid range
    top_nodes = [
        tf for tf in top_nodes
        if 0 <= tf.feature_idx < 256 and 1 <= map_layer_to_visual(module, tf.layer_idx) <= num_layers
    ]
    print(f"   → {len(top_nodes)} valid top features after filtering")
    
    # Try to load mIoU data from external storage first (same as analyze_lrp_tables.py)
    print(f"\n📂 Loading mIoU data from external storage...")
    print(f"   Path: {BASE_PATH / model / finetune / color / module}")
    
    all_units = load_all_neural_units_external(
        model=model,
        finetune=finetune,
        color=color,
        module=module,
        miou_threshold=miou_threshold
    )
    
    # Fallback to local directory if external not available
    if not all_units:
        print(f"   ⚠️ External data not found, trying local directory...")
        miou_dir = encoder_miou_dir if module == "encoder" else decoder_miou_dir
        all_units = load_all_neural_units(miou_dir, module, miou_threshold=miou_threshold)
    
    total_units = sum(len(units) for units in all_units.values())
    print(f"🧠 Loaded {total_units} neural units above mIoU threshold {miou_threshold}")
    for layer_idx in sorted(all_units.keys()):
        print(f"   Layer {layer_idx}: {len(all_units[layer_idx])} units")
    
    # Load comprehensive layer statistics from external storage
    layer_stats = load_layer_statistics_external(
        model=model,
        finetune=finetune,
        color=color,
        module=module,
        miou_threshold=miou_threshold
    )
    print(f"📈 Loaded layer statistics for {len(layer_stats)} layers")
    
    # Build edges from LRP data using coverage-based selection
    lrp_dir = encoder_dir if module == "encoder" else decoder_dir
    edges = build_edges(lrp_dir, top_nodes, coverage, module)
    print(f"🔗 Built {len(edges)} edges using {coverage*100:.0f}% coverage (conn_10pct_pos/neg)")

    # Determine output path
    out_base = out
    default_encoder_out = os.path.join(
        os.path.dirname(__file__), "..", "output", "visualisations", "encoder_graph"
    )
    if module == "decoder" and os.path.normpath(out_base) == os.path.normpath(default_encoder_out):
        out_base = os.path.join(
            os.path.dirname(__file__), "..", "output", "visualisations", "decoder_graph"
        )

    # Draw and save
    png_path, svg_path, pdf_path = draw_visualisation(
        output_path_base=out_base,
        top_nodes=top_nodes,
        edges=edges,
        all_units=all_units,
        num_layers=num_layers,
        base_node_size=base_node_size,
        fig_width_in=fig_width,
        fig_height_in=fig_height,
        module=module,
        miou_threshold=miou_threshold,
        layer_stats=layer_stats,
    )

    print(f"\n{'='*60}")
    print(f"✅ Saved {module} visualisation:")
    print(f"   📄 PNG: {png_path}")
    print(f"   📄 SVG: {svg_path}")
    print(f"   📄 PDF: {pdf_path}")
    print(f"{'='*60}\n")
    
    return png_path, svg_path, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication-quality VERTICAL network visualisation for encoder/decoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualise encoder with default settings (vertical layout)
  python visualise_network.py --module encoder

  # Visualise decoder with custom threshold
  python visualise_network.py --module decoder --miou_threshold 0.05

  # Specify model and finetune for loading stats
  python visualise_network.py --module encoder --model car --finetune finetune3 --color grau

  # Custom output path and figure size
  python visualise_network.py --module encoder --out ./figures/encoder_vis --fig_width 18 --fig_height 24
        """
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["encoder", "decoder"],
        default="encoder",
        help="Which module to visualise (encoder: 6 layers + pre-layer, decoder: 3 layers + pre-layer)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["car", "butterfly"],
        default="car",
        help="Model name for loading external statistics",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        choices=["finetune1", "finetune2", "finetune3"],
        default="finetune3",
        help="Finetuning step",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["grau", "orange", "blau"],
        default="grau",
        help="Color condition",
    )
    parser.add_argument(
        "--top_features",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "car",
            "finetune6",
            "lrp",
            "top_features.csv",
        ),
        help="Path to top_features.csv with highlighted features",
    )
    parser.add_argument(
        "--encoder_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "car",
            "finetune6",
            "lrp",
            "encoder",
        ),
        help="Directory with encoder LRP layer*_feat*.csv files",
    )
    parser.add_argument(
        "--decoder_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "car",
            "finetune6",
            "lrp",
            "decoder",
        ),
        help="Directory with decoder LRP layer*_feat*.csv files",
    )
    parser.add_argument(
        "--encoder_miou_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "car",
            "finetune6",
            "encoder",
            "rot",
        ),
        help="Directory with encoder mIoU results (miou_network_dissection.csv per layer)",
    )
    parser.add_argument(
        "--decoder_miou_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "car",
            "finetune6",
            "decoder",
        ),
        help="Directory with decoder mIoU results (mIoU_per_Query.csv per layer)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "visualisations",
            "encoder_graph_vertical",
        ),
        help="Output path base (without extension). Will generate .png, .svg, and .pdf",
    )
    parser.add_argument(
        "--coverage", 
        type=float, 
        default=0.10, 
        help="Coverage percentage for edge selection (0.10 = 10%%, uses conn_10pct_pos/neg logic)"
    )
    parser.add_argument(
        "--miou_threshold",
        type=float,
        default=0.04,
        help="mIoU threshold for showing units as active (colored nodes)",
    )
    parser.add_argument(
        "--base_node_size", 
        type=float, 
        default=4.0, 
        help="Base node size in points"
    )
    parser.add_argument(
        "--fig_width", 
        type=float, 
        default=16.0, 
        help="Figure width in inches"
    )
    parser.add_argument(
        "--fig_height", 
        type=float, 
        default=20.0, 
        help="Figure height in inches (taller for vertical layout)"
    )

    cli_args = parser.parse_args()
    main(
        module=cli_args.module,
        model=cli_args.model,
        finetune=cli_args.finetune,
        color=cli_args.color,
        top_features=cli_args.top_features,
        encoder_dir=cli_args.encoder_dir,
        decoder_dir=cli_args.decoder_dir,
        encoder_miou_dir=cli_args.encoder_miou_dir,
        decoder_miou_dir=cli_args.decoder_miou_dir,
        out=cli_args.out,
        coverage=cli_args.coverage,
        miou_threshold=cli_args.miou_threshold,
        base_node_size=cli_args.base_node_size,
        fig_width=cli_args.fig_width,
        fig_height=cli_args.fig_height,
    )
