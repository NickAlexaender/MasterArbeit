import os

from myThesis.lrp import calc_lrp
from myThesis.lrp import calculate_network as netcalc


# Eingaben/Wege
images_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images"
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

	# 2) Je einen Kandidaten auswählen
	enc_row = next((r for r in rows if r.module == "encoder"), None)
	dec_row = next((r for r in rows if r.module == "decoder"), None)

	# 3) Falls nichts gefunden wurde, mit sinnvollen Defaults weiterrechnen
	if enc_row is None:
		print("Kein Encoder-Top-Feature gefunden – verwende Fallback (layer=3, feature=233).")
		enc_row = netcalc.TopFeature(module="encoder", layer_idx=2, feature_idx=233, miou=0.0)
	if dec_row is None:
		print("Kein Decoder-Top-Feature gefunden – verwende Fallback (layer=3, feature=278).")
		dec_row = netcalc.TopFeature(module="decoder", layer_idx=2, feature_idx=278, miou=0.0)

	# 4) Ausführen
	_run_single(enc_row)
	_run_single(dec_row)