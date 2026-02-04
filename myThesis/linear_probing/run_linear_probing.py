import os
import sys

# Projekt-Root zu sys.path hinzufügen
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myThesis.linear_probing.linear_probing_trainer import train_all_layers



BASE_DIR = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/test"
OUTPUT_DIR = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/linear_probing_results"
LAYER = "layer1"
TEST_SIZE = 0.20
MAX_ITER = 1000
C = 1.0
CLASS_WEIGHT = "balanced"
USE_SGD = True
SUBSAMPLE = None
RUN_BOTH = True
SAMPLES_PER_CLASS = 10000


if __name__ == "__main__":
    print("=" * 70)
    print("🔬 LINEAR PROBING TRAINING")
    print("=" * 70)
    print(f"   Base Dir:    {BASE_DIR}")
    print(f"   Output Dir:  {OUTPUT_DIR}")
    print(f"   Layer:       {LAYER if LAYER else 'Alle'}")
    print(f"   Train/Val:   {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
    if RUN_BOTH:
        print(f"   Experimente: Natural + Balanced ({SAMPLES_PER_CLASS:,}/Klasse)")
    print("=" * 70 + "\n")
    
    results = train_all_layers(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        layer=LAYER,
        run_both=RUN_BOTH,
        samples_per_class=SAMPLES_PER_CLASS,
        test_size=TEST_SIZE,
        max_iter=MAX_ITER,
        C=C,
        class_weight=CLASS_WEIGHT,
        use_sgd=USE_SGD,
        subsample=SUBSAMPLE,
        save_model=True,
        verbose=True,
    )
    
    print("\n✅ Linear Probing abgeschlossen!")
    print(f"   Ergebnisse gespeichert in: {OUTPUT_DIR}")
