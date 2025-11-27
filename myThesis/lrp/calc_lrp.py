"""
LRP/Attributions-Analyse für MaskDINO-Transformer Encoder

Funktion:
- Lädt das MaskDINO-Modell (wie in myThesis/fine-tune.py konfiguriert)
- Führt eine Attribution (LRP) für ein wählbares Encoder-/Decoder-Layer
	und ein bestimmtes Feature (Kanalindex) durch
- Aggregiert Beiträge der vorherigen Features (Kanäle) über alle Bilder im Ordner
- Exportiert Ergebnisse als CSV-Datei

"""

from __future__ import annotations

# WICHTIG: Kompatibilitäts-Patches müssen zuerst geladen werden
from myThesis.lrp.calc.compat import *  # Pillow/NumPy monkey patches

import logging

from myThesis.lrp.calc.cli import parse_args
from myThesis.lrp.calc.core_analysis import run_analysis


def main(
	images_dir: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images",
	layer_index: int = 3,
	feature_index: int = 214,
	target_norm: str = "sum1",
	lrp_epsilon: float = 1e-6,
	output_csv: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/lrp_result.csv",
	which_module: str = "encoder",
	method: str = "lrp",
    weights_path: str | None = None,
):
	"""Programmierbarer Einstiegspunkt mit denselben Parametern wie der CLI-Parser.

	Hinweise:
	- ``layer_index`` ist 1-basiert (wie zuvor in der CLI).
	- ``limit_images``: 0 oder negativ bedeutet alle Bilder (intern wird ``None`` übergeben).
	"""
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

	run_analysis(
		images_dir=images_dir,
		layer_index=layer_index,
		feature_index=feature_index,
		output_csv=output_csv,
		target_norm=target_norm,
		lrp_epsilon=lrp_epsilon,
		which_module=which_module,
		method=method,
		weights_path=weights_path,
	)


if __name__ == "__main__":
	# Rückwärtskompatibler CLI-Einstiegspunkt
	_args = parse_args()
	main(
		images_dir=_args.images_dir,
		layer_index=_args.layer_index,
		feature_index=_args.feature_index,
		target_norm=_args.target_norm,
		lrp_epsilon=_args.lrp_epsilon,
		output_csv=_args.output_csv,
		which_module=_args.which_module,
		method=_args.method,
		weights_path=getattr(_args, "weights_path", None),
	)
