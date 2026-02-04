import os
from typing import Optional, Literal

from myThesis.linear_probing.lp_on_transformer_encoder import main as encoder_extract
from myThesis.linear_probing.lp_on_transformer_decoder import main as decoder_extract
from myThesis.linear_probing.linear_probing_trainer import (
    train_all_layers,
    run_both_experiments as run_both_exp,
)

# Nun führen wir Linear Probing Pipeline für Transformer-LAyer durch

def run_pipeline(
    images_dir: str,
    weights_path: str,
    output_dir: str,
    model: str = "butterfly",
    # Modus
    mode: Literal["encoder", "decoder", "all"] = "all",
    layer: Optional[str] = None,  # z.B. "layer1" oder None für alle
    # Training-Optionen
    samples_per_class: int = 10000,  # Encoder: 10k, Decoder: 0 (auto)
    natural_subsample: float = 0.1,  # Encoder: 10%, Decoder: 1.0 (alle)
    test_size: float = 0.20,
    max_iter: int = 1000,
    use_sgd: bool = True,
    save_model: bool = True,
):
    _print_header(model, weights_path, images_dir, mode, layer)
    
    encoder_features_dir = os.path.join(output_dir, "encoder_features")
    encoder_results_dir = os.path.join(output_dir, "encoder_results")
    decoder_features_dir = os.path.join(output_dir, "decoder_features")
    decoder_results_dir = os.path.join(output_dir, "decoder_results")
    base_train_kwargs = dict(
        test_size=test_size,
        max_iter=max_iter,
        use_sgd=use_sgd,
        save_model=save_model,
        verbose=True,
    )
    # Balanced: 10.000 pro Klasse
    if mode in ("encoder", "all"):
        encoder_kwargs = {
            **base_train_kwargs,
            "samples_per_class": samples_per_class,  # 10.000
            "natural_subsample": natural_subsample,  # 10%
        }
        _run_encoder_pipeline(
            images_dir=images_dir,
            weights_path=weights_path,
            model=model,
            layer=layer if mode == "encoder" else None,  # Bei "all" alle Layer
            features_dir=encoder_features_dir,
            results_dir=encoder_results_dir,
            **encoder_kwargs,
        )
    
    # Decoder: Auto: alle Daten
    if mode in ("decoder", "all"):
        decoder_kwargs = {
            **base_train_kwargs,
            "samples_per_class": 0,   # Auto: min(verfügbare pro Klasse)
            "natural_subsample": 1.0,  # Alle Daten (kein Subsample)
        }
        _run_decoder_pipeline(
            images_dir=images_dir,
            weights_path=weights_path,
            model=model,
            layer=layer if mode == "decoder" else None,  # Bei "all" alle Layer
            features_dir=decoder_features_dir,
            results_dir=decoder_results_dir,
            **decoder_kwargs,
        )
    
    # Zusammenfassung
    _print_summary(mode, output_dir, encoder_features_dir, encoder_results_dir,
                   decoder_features_dir, decoder_results_dir)

# Pipeline für nur Encoder

def _run_encoder_pipeline(
    images_dir: str,
    weights_path: str,
    model: str,
    layer: Optional[str],
    features_dir: str,
    results_dir: str,
    **train_kwargs,
):
    print("\n" + "─" * 70)
    print(f"🔷 ENCODER: Feature-Extraktion (Layer: {layer or 'Alle'})")
    print("─" * 70)
    
    encoder_extract(
        images_dir=images_dir,
        weights_path=weights_path,
        output_dir=features_dir,
        model=model,
        layer=layer,
    )
    
    print("\n" + "─" * 70)
    print("🔷 ENCODER: Linear Probing Training")
    print("─" * 70)
    
    train_all_layers(
        base_dir=features_dir,
        output_dir=results_dir,
        layer=layer,
        run_both=True,
        **train_kwargs,
    )

# Pipeline für nur Decoder

def _run_decoder_pipeline(
    images_dir: str,
    weights_path: str,
    model: str,
    layer: Optional[str],
    features_dir: str,
    results_dir: str,
    **train_kwargs,
):
    print("\n" + "─" * 70)
    print(f"🔶 DECODER: Feature-Extraktion (Layer: {layer or 'Alle'})")
    print("─" * 70)
    
    decoder_extract(
        images_dir=images_dir,
        weights_path=weights_path,
        output_dir=features_dir,
        model=model,
    )
    
    print("\n" + "─" * 70)
    print("🔶 DECODER: Linear Probing Training")
    print("─" * 70)

    train_all_layers(
        base_dir=features_dir,
        output_dir=results_dir,
        layer=layer,
        run_both=True,
        csv_name="queries.csv",  # Decoder verwendet queries.csv
        **train_kwargs,
    )


# Drucken des Headers

def _print_header(model: str, weights_path: str, images_dir: str, 
                  mode: str, layer: Optional[str]):
    mode_text = {
        "encoder": "Nur Encoder",
        "decoder": "Nur Decoder",
        "all": "Encoder + Decoder",
    }
    
    print("=" * 70)
    print("🔬 LINEAR PROBING PIPELINE")
    print("=" * 70)
    print(f"   Modus:       {mode_text.get(mode, mode)}")
    print(f"   Layer:       {layer or 'Alle'}")
    print(f"   Model:       {model}")
    print(f"   Weights:     {weights_path}")
    print(f"   Images:      {images_dir}")
    print("=" * 70)
    print("   Regeln:")
    print("   • Natural:   10% zufällige Samples")
    print("   • Balanced:  10.000 Samples pro Klasse")
    print("=" * 70 + "\n")

# Printen der Zusammenfassung

def _print_summary(mode: str, output_dir: str, 
                   encoder_features_dir: str, encoder_results_dir: str,
                   decoder_features_dir: str, decoder_results_dir: str):
    print("\n" + "=" * 70)
    print("✅ LINEAR PROBING ABGESCHLOSSEN")
    print("=" * 70)
    
    if mode in ("encoder", "all"):
        print(f"   Encoder Features: {encoder_features_dir}")
        print(f"   Encoder Results:  {encoder_results_dir}")
    
    if mode in ("decoder", "all"):
        print(f"   Decoder Features: {decoder_features_dir}")
        print(f"   Decoder Results:  {decoder_results_dir}")
    
    print("=" * 70)
