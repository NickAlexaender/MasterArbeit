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
train_state="finetune3"
local_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/"
output_root="/Volumes/Untitled/Master-Arbeit_Ergebnisse/output/"

# Modus: "encoder", "decoder" oder "all"
MODE = "all" # "encoder", "decoder" oder "all"

# Layer: z.B. "layer0", "layer1", ... oder None für alle Layer
LAYER = None # z.B. "layer0", "layer1" oder None für alle


# ─────────────────────────────────────────────────────────────────────────────
# Ausführung
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(
        images_dir=images_dir,
        weights_path=weights_path,
        output_dir=f"{output_root}{model}/{train_state}/linear_probing",
        model=model,
        mode=MODE,
        layer=LAYER,
    )


