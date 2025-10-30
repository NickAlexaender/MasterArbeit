import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class TopFeature:
    layer_idx: int  # encoder data layer index (0-based)
    feature_idx: int  # 0..255
    miou: float


@dataclass
class Edge:
    from_layer_vis: int  # visual layer (1..7)
    to_layer_vis: int  # visual layer (1..7)
    from_feature: int  # 0..255
    to_feature: int  # 0..255
    relevance: float


def load_top_features(top_features_csv: str, module: str) -> List[TopFeature]:
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
            # Skip malformed rows
            continue
    return top_features


def map_layer_to_visual(module: str, layer_idx: int) -> int:
    """Map a data layer index to a visual layer number depending on module.

    Rules:
    - encoder: visual_layer = layer_idx + 2 (visual layers 1..7 expected)
    - decoder: visual_layer = layer_idx + 2 (visual layers 1..4 for decoder)
    """
    module = module.lower()
    if module == "encoder":
        return layer_idx + 2
    elif module == "decoder":
        return layer_idx + 2
    else:
        # default fallback
        return layer_idx + 2


def get_layer_feature_csv(root_dir: str, encoder_layer_idx: int, feature_idx: int) -> str:
    # Example: myThesis/output/car/finetune6/lrp/encoder/layer0_feat3.csv
    return os.path.join(root_dir, f"layer{encoder_layer_idx}_feat{feature_idx}.csv")


def load_top_k_edges_for_node(
    csv_path: str, k: int
) -> List[Tuple[int, float]]:
    """Return top-k (prev_feature_idx, relevance) by absolute relevance from a node CSV.

    If file missing or empty, return [].
    """
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if "prev_feature_idx" not in df.columns or "relevance" not in df.columns:
        return []

    # Sort by absolute relevance descending
    df_sorted = df.sort_values(by="relevance", key=lambda s: s.abs(), ascending=False)
    df_top = df_sorted.head(k)
    pairs: List[Tuple[int, float]] = []
    for _, row in df_top.iterrows():
        try:
            pairs.append((int(row["prev_feature_idx"]), float(row["relevance"])))
        except Exception:
            continue
    return pairs


def build_edges(
    root_dir: str, top_nodes: List[TopFeature], k: int, module: str
) -> List[Edge]:
    edges: List[Edge] = []
    for tf in top_nodes:
        to_layer_vis = map_layer_to_visual(module, tf.layer_idx)
        from_layer_vis = to_layer_vis - 1
        csv_path = get_layer_feature_csv(root_dir, tf.layer_idx, tf.feature_idx)
        pairs = load_top_k_edges_for_node(csv_path, k)
        for prev_feature, rel in pairs:
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


def compute_layout(num_layers: int = 7, features_per_layer: int = 256, width: int = 256) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Compute (x,y) for each (visual_layer, feature_idx).

    - Layers are numbered 1..num_layers vertically from bottom (1) to top (num_layers).
    - x is feature index normalized to [0, 1]. y is normalized to [0, 1].
    - width is used only to scale marker sizes aesthetically later.
    """
    coords: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for layer in range(1, num_layers + 1):
        y = (layer - 1) / (num_layers - 1) if num_layers > 1 else 0.5
        for feat in range(features_per_layer):
            x = feat / (features_per_layer - 1) if features_per_layer > 1 else 0.5
            coords[(layer, feat)] = (x, y)
    return coords


def draw_visualisation(
    output_path_base: str,
    top_nodes: List[TopFeature],
    edges: List[Edge],
    num_layers: int = 7,
    features_per_layer: int = 256,
    base_node_size: float = 10.0,
    fig_width_in: float = 18.0,
    fig_height_in: float = 10.0,
    module: str = "encoder",
):
    coords = compute_layout(num_layers=num_layers, features_per_layer=features_per_layer)

    # Build quick lookup for red nodes (size scaling)
    red_nodes_by_layer: Dict[int, Dict[int, float]] = {l: {} for l in range(1, num_layers + 1)}
    for tf in top_nodes:
        vis_layer = map_layer_to_visual(module, tf.layer_idx)
        if 1 <= vis_layer <= num_layers and 0 <= tf.feature_idx < features_per_layer:
            red_nodes_by_layer[vis_layer][tf.feature_idx] = tf.miou

    # Prepare figure
    plt.figure(figsize=(fig_width_in, fig_height_in), dpi=150)
    ax = plt.gca()
    ax.set_axis_off()

    # Draw edges first (so nodes overlay)
    # For each node's edges, scale linewidths locally among its top-k
    # Group edges by (to_layer, to_feature)
    from collections import defaultdict

    edges_by_target: Dict[Tuple[int, int], List[Edge]] = defaultdict(list)
    for e in edges:
        edges_by_target[(e.to_layer_vis, e.to_feature)].append(e)

    for (to_layer, to_feat), e_list in edges_by_target.items():
        # scale by local max abs relevance
        abs_vals = np.array([abs(e.relevance) for e in e_list], dtype=float)
        if len(abs_vals) == 0:
            continue
        max_abs = abs_vals.max() if abs_vals.max() > 0 else 1.0
        for e in e_list:
            (x1, y1) = coords.get((e.from_layer_vis, e.from_feature), (None, None))
            (x2, y2) = coords.get((e.to_layer_vis, e.to_feature), (None, None))
            if x1 is None or x2 is None:
                continue
            # line width between 0.5 and 5.0 based on relative strength
            rel_strength = abs(e.relevance) / max_abs
            lw = 0.5 + 4.5 * float(rel_strength)
            # color by sign
            color = (0.85, 0.0, 0.0, 0.7) if e.relevance >= 0 else (0.0, 0.4, 0.9, 0.6)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="round", alpha=0.8)

    # Draw nodes layer by layer
    for layer in range(1, num_layers + 1):
        xs = []
        ys = []
        sizes = []
        colors = []
        for feat in range(features_per_layer):
            x, y = coords[(layer, feat)]
            xs.append(x)
            ys.append(y)
            miou = red_nodes_by_layer[layer].get(feat)
            if miou is not None:
                scale = 1.0 + (miou * 2.0)
                sizes.append(base_node_size * scale)
                colors.append((0.9, 0.1, 0.1, 0.95))  # red nodes
            else:
                sizes.append(base_node_size)
                colors.append((0.6, 0.6, 0.6, 0.6))  # grey nodes

        # matplotlib scatter sizes are in points^2
        s = (np.array(sizes) ** 2).astype(float)
        ax.scatter(xs, ys, s=s, c=colors, edgecolors="none")

    # Layer labels on the left
    for layer in range(1, num_layers + 1):
        y = (layer - 1) / (num_layers - 1) if num_layers > 1 else 0.5
        ax.text(
            -0.01,
            y,
            f"Layer {layer}",
            va="center",
            ha="right",
            fontsize=10,
        )

    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.05, 1.05)

    # Save outputs
    os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
    png_path = output_path_base + ".png"
    svg_path = output_path_base + ".svg"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    return png_path, svg_path


def main(
    module: str = "encoder",
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
    out: str = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "visualisations",
        "encoder_graph",
    ),
    k: int = 5,
    base_node_size: float = 6.0,
    fig_width: float = 20.0,
    fig_height: float = 12.0,
):
    """Run the visualisation pipeline.

    Parameters mirror the previous CLI flags so this function can be called externally.
    Returns a tuple of (png_path, svg_path).
    """
    # Load data for selected module
    module = module.lower()
    top_nodes = load_top_features(top_features, module)

    # Set visual layer count per module
    if module == "encoder":
        num_layers = 7
    else:
        # decoder -> show 4 visual layers (there are 3 data layers but we offset by +2)
        num_layers = 4

    # Filter to valid range of features (0..255) and layers that map into visual range
    top_nodes = [
        tf
        for tf in top_nodes
        if 0 <= tf.feature_idx < 256 and 1 <= map_layer_to_visual(module, tf.layer_idx) <= num_layers
    ]

    # Build edges from the selected directory
    node_dir = encoder_dir if module == "encoder" else decoder_dir
    edges = build_edges(node_dir, top_nodes, k, module)

    # Draw and save
    out_base = out
    # If user keeps default encoder_graph but asked for decoder, switch name for convenience
    default_encoder_out = os.path.join(
        os.path.dirname(__file__), "..", "output", "visualisations", "encoder_graph"
    )
    if module == "decoder" and os.path.normpath(out_base) == os.path.normpath(default_encoder_out):
        out_base = os.path.join(
            os.path.dirname(__file__), "..", "output", "visualisations", "decoder_graph"
        )

    png_path, svg_path = draw_visualisation(
        output_path_base=out_base,
        top_nodes=top_nodes,
        edges=edges,
        num_layers=num_layers,
        base_node_size=base_node_size,
        fig_width_in=fig_width,
        fig_height_in=fig_height,
        module=module,
    )

    print(f"Saved {module} visualisation to:\n- {png_path}\n- {svg_path}")
    return png_path, svg_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise encoder/decoder layers and top connections.")
    parser.add_argument(
        "--module",
        type=str,
        choices=["encoder", "decoder"],
        default="encoder",
        help="Which module to visualise (affects filtering and directory)",
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
        help="Path to top_features.csv",
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
        help="Directory with encoder layer*_feat*.csv files",
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
        help="Directory with decoder layer*_feat*.csv files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "visualisations",
            "encoder_graph",
        ),
        help="Output path base (without extension)",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k connections per red node")
    parser.add_argument("--base_node_size", type=float, default=6.0, help="Base node size (points)")
    parser.add_argument("--fig_width", type=float, default=20.0, help="Figure width in inches")
    parser.add_argument("--fig_height", type=float, default=12.0, help="Figure height in inches")

    cli_args = parser.parse_args()
    main(
        module=cli_args.module,
        top_features=cli_args.top_features,
        encoder_dir=cli_args.encoder_dir,
        decoder_dir=cli_args.decoder_dir,
        out=cli_args.out,
        k=cli_args.k,
        base_node_size=cli_args.base_node_size,
        fig_width=cli_args.fig_width,
        fig_height=cli_args.fig_height,
    )
