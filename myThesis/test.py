# Warnungen unterdrücken BEVOR andere Module importiert werden
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os

from myThesis.lrp import calc_lrp
from myThesis.lrp import calculate_network as netcalc


# Eingaben/Wege
  # 1images hier Maß aller Dinge, aber test_single -> besser für schnellen Test
  # /Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images -> test_single
images_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/test_single"  # 1 Bild für Decoder-Test
weights_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth"
model = "car"
train_state = "finetune6"
basic_root = "/Users/nicklehmacher/Alles/MasterArbeit/"


def _ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


def _run_single(module_row: "netcalc.TopFeature") -> None:
	"""Führt calc_lrp für genau ein Modul/Feature aus (Encoder oder Decoder)."""
	# calc_lrp erwartet 1-basierte Layer-Indizes
	layer_1based = module_row.layer_idx + 1
	module = module_row.module  # "encoder" | "decoder"

	# Output-Datei (sicherstellen, dass es wirklich eine DATEI ist, nicht nur ein Ordner)
	out_dir = os.path.join(
		basic_root,
		f"myThesis/output/{model}/{train_state}/lrp/tests/{module}",
	)
	_ensure_dir(out_dir)
	out_csv = os.path.join(
		out_dir, f"layer{module_row.layer_idx}_feat{module_row.feature_idx}.csv"
	)

	print(
		f"Starte LRP: module={module}, layer={layer_1based} (0b={module_row.layer_idx}), feature={module_row.feature_idx} -> {out_csv}"
	)

	calc_lrp.main(
		images_dir=images_dir,
		layer_index=layer_1based,
		feature_index=module_row.feature_idx,
		which_module=module,
		output_csv=out_csv,
		weights_path=weights_path,
  # num_queries sollte generell entfernt werden -> besser für schnellen Test
		# num_queries=10,  # Nur 1 Query für Test (spart Memory bei Decoder)
	)


if __name__ == "__main__":
	# 1) Pfade wie in calculate_network ableiten und Top-Features je Layer ermitteln
	output_root = os.path.join(basic_root, f"myThesis/output/{model}/{train_state}")
	encoder_rot_dir = os.path.join(output_root, "encoder/rot")
	decoder_dir = os.path.join(output_root, "decoder")

	print(f"Suche Top-Features in\n  encoder_rot_dir={encoder_rot_dir}\n  decoder_dir={decoder_dir}")
	rows = netcalc.find_top_k_per_layer(
		k=1,
		encoder_rot_dir=encoder_rot_dir,
		decoder_dir=decoder_dir,
	)

	# 2) Je 2 Kandidaten auswählen (Encoder und Decoder)
	enc_rows = [r for r in rows if r.module == "encoder"][:2]
	dec_rows = [r for r in rows if r.module == "decoder"][:2]

	# 3) Falls nicht genug gefunden wurden, mit sinnvollen Defaults auffüllen
	if len(enc_rows) < 2:
		print(f"Nur {len(enc_rows)} Encoder-Top-Features gefunden – fülle mit Fallbacks auf.")
		fallback_enc = [
			netcalc.TopFeature(module="encoder", layer_idx=2, feature_idx=233, miou=0.0),
			netcalc.TopFeature(module="encoder", layer_idx=2, feature_idx=100, miou=0.0),
		]
		enc_rows.extend(fallback_enc[len(enc_rows):2])
	if len(dec_rows) < 2:
		print(f"Nur {len(dec_rows)} Decoder-Top-Features gefunden – fülle mit Fallbacks auf.")
		fallback_dec = [
			netcalc.TopFeature(module="decoder", layer_idx=2, feature_idx=278, miou=0.0),
			netcalc.TopFeature(module="decoder", layer_idx=2, feature_idx=150, miou=0.0),
		]
		dec_rows.extend(fallback_dec[len(dec_rows):2])

	# 4) Ausführen - 2 Encoder und 2 Decoder
	print(f"\n=== Starte LRP für {len(enc_rows)} Encoder-Features ===")
	for i, enc_row in enumerate(enc_rows, 1):
		print(f"\n--- Encoder {i}/{len(enc_rows)} ---")
		_run_single(enc_row)

	print(f"\n=== Starte LRP für {len(dec_rows)} Decoder-Features ===")
	for i, dec_row in enumerate(dec_rows, 1):
		print(f"\n--- Decoder {i}/{len(dec_rows)} ---")
		_run_single(dec_row)