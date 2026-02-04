from __future__ import annotations

# Paar basic Konfigurationen

# Standardwert in Network Dissection-Paper
DEFAULT_PERCENTILE: float = 99.5

# Bilder dürfen nicht zu groß sein
MAX_INPUT_SIZE: int = 2048

# Auch bei den files geben wir Standardwerte an
EXPORT_SUBDIR: str = "iou_results"
MASK_DEFAULT_SUBDIR: str = "image/rot"
DECODER_DEFAULT_SUBDIR: str = "output/decoder"
CSV_MIOU_FILENAME: str = "mIoU_per_Query.csv"
