"""
Linear Probing für Transformer Encoder und Decoder.

Modi:
    - "encoder": Nur Encoder (ein Layer oder alle)
    - "decoder": Nur Decoder (ein Layer oder alle)  
    - "all": Alle Encoder + Decoder Layer

Usage:
    python3 myThesis/calculate_linear_smv.py
"""

from myThesis.linear_probing.linear_probing_pipeline import run_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

# Pfade

images_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/1images"
weights_path="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_0001199.pth"
model = "car"
train_state="finetune4"
concept="orange"
local_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/"
output_root="/Volumes/Untitled/Master-Arbeit_Ergebnisse/output/"

MODE = "all"

LAYER = None

run_pipeline(
        images_dir=images_dir,
        weights_path=weights_path,
        output_dir=f"{output_root}{model}/{train_state}/linear_probing",
        model=model,
        mode=MODE,
        layer=LAYER,
    )