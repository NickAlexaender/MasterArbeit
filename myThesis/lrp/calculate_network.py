import os
import csv
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, Dict


DEFAULT_BASE_DIR = "/Users/nicklehmacher/Alles/MasterArbeit"
DEFAULT_OUTPUT_ROOT = os.path.join(
	DEFAULT_BASE_DIR, "myThesis/output/car/finetune6"
)
DEFAULT_ENCODER_ROT_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "encoder/rot")
DEFAULT_DECODER_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "decoder")
DEFAULT_LRP_OUT_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "lrp")
DEFAULT_LRP_ENCODER_DIR = os.path.join(DEFAULT_LRP_OUT_DIR, "encoder")
DEFAULT_LRP_DECODER_DIR = os.path.join(DEFAULT_LRP_OUT_DIR, "decoder")
DEFAULT_SUMMARY_CSV = os.path.join(DEFAULT_LRP_OUT_DIR, "top_features.csv")
DEFAULT_IMAGES_DIR = os.path.join(DEFAULT_BASE_DIR, "myThesis/image/1images")


@dataclass
class TopFeature:
	module: str  # "encoder" | "decoder"
	layer_idx: int  # 0-based layer index in results
	feature_idx: int  # encoder: feature_idx, decoder: query_idx
	miou: float

	def to_row(self) -> List[Any]:
		return [self.module, self.layer_idx, self.feature_idx, self.miou]

# Berechne effektive Pfade basierend auf übergebenen Parametern und Standardwerten

def _compute_paths(
	*,
	output_root: Optional[str] = None,
	images_dir: Optional[str] = None,
	encoder_rot_dir: Optional[str] = None,
	decoder_dir: Optional[str] = None,
	lrp_out_dir: Optional[str] = None,
	lrp_encoder_dir: Optional[str] = None,
	lrp_decoder_dir: Optional[str] = None,
	summary_csv: Optional[str] = None,
) -> Dict[str, str]:

	# Start from defaults
	eff_output_root = output_root or DEFAULT_OUTPUT_ROOT
	eff_images_dir = images_dir or DEFAULT_IMAGES_DIR

	# Derived from output_root if not explicitly provided
	eff_encoder_rot_dir = encoder_rot_dir or os.path.join(eff_output_root, "encoder/rot")
	eff_decoder_dir = decoder_dir or os.path.join(eff_output_root, "decoder")
	eff_lrp_out_dir = lrp_out_dir or os.path.join(eff_output_root, "lrp")
	eff_lrp_encoder_dir = lrp_encoder_dir or os.path.join(eff_lrp_out_dir, "encoder")
	eff_lrp_decoder_dir = lrp_decoder_dir or os.path.join(eff_lrp_out_dir, "decoder")
	eff_summary_csv = summary_csv or os.path.join(eff_lrp_out_dir, "top_features.csv")

	return {
		"output_root": eff_output_root,
		"images_dir": eff_images_dir,
		"encoder_rot_dir": eff_encoder_rot_dir,
		"decoder_dir": eff_decoder_dir,
		"lrp_out_dir": eff_lrp_out_dir,
		"lrp_encoder_dir": eff_lrp_encoder_dir,
		"lrp_decoder_dir": eff_lrp_decoder_dir,
		"summary_csv": eff_summary_csv,
	}


def _list_layer_dirs(parent: str) -> List[str]:
	if not os.path.isdir(parent):
		return []
	names = [n for n in os.listdir(parent) if n.startswith("layer")]
	# sort by numeric suffix
	def _key(n: str) -> int:
		try:
			return int(n.replace("layer", ""))
		except Exception:
			return 1_000_000

	names.sort(key=_key)
	return [os.path.join(parent, n) for n in names]


def _read_encoder_csv(csv_path: str) -> List[TopFeature]:
	tops: List[TopFeature] = []
	with open(csv_path, "r", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				layer_idx = int(row.get("layer_idx", "0"))
				feature_idx = int(row["feature_idx"])  # required
				# encoder file uses column name 'miou'
				miou = float(row.get("miou") or row.get("mean_iou") or row.get("mIoU") or 0.0)
				tops.append(TopFeature("encoder", layer_idx, feature_idx, miou))
			except Exception:
				continue
	return tops


def _read_decoder_csv(csv_path: str, layer_idx_from_dir: Optional[int]) -> List[TopFeature]:
	tops: List[TopFeature] = []
	with open(csv_path, "r", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				# decoder csv has no explicit layer column; take from folder name
				layer_idx = layer_idx_from_dir if layer_idx_from_dir is not None else 0
				feature_idx = int(row["query_idx"])  # queries act as feature indices
				miou = float(row.get("mean_iou") or row.get("miou") or row.get("mIoU") or 0.0)
				tops.append(TopFeature("decoder", layer_idx, feature_idx, miou))
			except Exception:
				continue
	return tops


def find_top_k_per_layer(
	k: int = 5,
	*,
	encoder_rot_dir: str = DEFAULT_ENCODER_ROT_DIR,
	decoder_dir: str = DEFAULT_DECODER_DIR,
) -> List[TopFeature]:
	results: List[TopFeature] = []

	# Encoder (rot variant)
	for layer_dir in _list_layer_dirs(encoder_rot_dir):
		layer_name = os.path.basename(layer_dir)
		try:
			layer_idx = int(layer_name.replace("layer", ""))
		except Exception:
			continue
		csv_path = os.path.join(layer_dir, "miou_network_dissection.csv")
		if not os.path.isfile(csv_path):
			continue
		rows = _read_encoder_csv(csv_path)
		# Ensure we only keep rows for this layer (robustness)
		rows = [r for r in rows if r.layer_idx == layer_idx]
		rows.sort(key=lambda r: r.miou, reverse=True)
		results.extend(rows[:k])

	# Decoder
	for layer_dir in _list_layer_dirs(decoder_dir):
		layer_name = os.path.basename(layer_dir)
		try:
			layer_idx = int(layer_name.replace("layer", ""))
		except Exception:
			continue
		csv_path = os.path.join(layer_dir, "mIoU_per_Query.csv")
		if not os.path.isfile(csv_path):
			continue
		rows = _read_decoder_csv(csv_path, layer_idx_from_dir=layer_idx)
		rows.sort(key=lambda r: r.miou, reverse=True)
		results.extend(rows[:k])

	return results


def save_summary(rows: List[TopFeature], out_csv: str = DEFAULT_SUMMARY_CSV) -> None:
	os.makedirs(os.path.dirname(out_csv), exist_ok=True)
	with open(out_csv, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["module", "layer_idx", "feature_idx", "miou"])  # header
		for r in rows:
			w.writerow(r.to_row())


def print_summary(rows: List[TopFeature]) -> None:
	print("Top-Features (pro Layer, je 5) – Modul | Layer | Feature | mIoU:")
	for r in rows:
		print(f"{r.module:7s} | L{r.layer_idx:02d} | F{r.feature_idx:03d} | {r.miou:.6f}")


def _get_existing_features(
	lrp_encoder_dir: str,
	lrp_decoder_dir: str,
) -> set:
	existing = set()
	
	# Check encoder directory
	if os.path.isdir(lrp_encoder_dir):
		for fname in os.listdir(lrp_encoder_dir):
			if fname.startswith("._"):  # Skip macOS metadata files
				continue
			if fname.endswith(".csv") and fname.startswith("layer"):
				try:
					# Parse "layer{layer_idx}_feat{feature_idx}.csv"
					parts = fname.replace(".csv", "").split("_")
					layer_idx = int(parts[0].replace("layer", ""))
					feature_idx = int(parts[1].replace("feat", ""))
					existing.add(("encoder", layer_idx, feature_idx))
				except Exception:
					continue
	
	# Check decoder directory
	if os.path.isdir(lrp_decoder_dir):
		for fname in os.listdir(lrp_decoder_dir):
			if fname.startswith("._"):  # Skip macOS metadata files
				continue
			if fname.endswith(".csv") and fname.startswith("layer"):
				try:
					# Parse "layer{layer_idx}_feat{feature_idx}.csv"
					parts = fname.replace(".csv", "").split("_")
					layer_idx = int(parts[0].replace("layer", ""))
					feature_idx = int(parts[1].replace("feat", ""))
					existing.add(("decoder", layer_idx, feature_idx))
				except Exception:
					continue
	
	return existing


def run_lrp_for(
	rows: List[TopFeature],
	*,
	images_dir: str = DEFAULT_IMAGES_DIR,
	lrp_encoder_dir: str = DEFAULT_LRP_ENCODER_DIR,
	lrp_decoder_dir: str = DEFAULT_LRP_DECODER_DIR,
	limit_images: Optional[int] = None,
	weights_path: Optional[str] = None,
	model_type: str = "car",
) -> None:
	# Import here to avoid heavy import time when only listing
	import gc
	import torch
	from myThesis.lrp import calc_lrp

	# Ensure output folders
	os.makedirs(lrp_encoder_dir, exist_ok=True)
	os.makedirs(lrp_decoder_dir, exist_ok=True)

	# Check which features already exist
	existing_features = _get_existing_features(lrp_encoder_dir, lrp_decoder_dir)
	to_process = [
		(r, i) for i, r in enumerate(rows)
		if (r.module, r.layer_idx, r.feature_idx) not in existing_features
	]
	
	if not to_process:
		print("✅ Alle Features sind bereits berechnet!")
		return
	
	skipped_count = len(rows) - len(to_process)
	if skipped_count > 0:
		print(f"⏭️  {skipped_count} Features überspringen (bereits berechnet)")
		print(f"📊 Verarbeite {len(to_process)} neue Features...\n")

	for i, (r, orig_idx) in enumerate(to_process):
		# calc_lrp.main expects 1-based layer index; our CSV layers are 0-based
		layer_1based = r.layer_idx + 1
		# CSV feature_idx is 1-based (1-256), but LRP expects 0-based indices (0-255)
		feature_0based = r.feature_idx - 1
		which_module = r.module
		if which_module == "encoder":
			out_csv = os.path.join(lrp_encoder_dir, f"layer{r.layer_idx}_feat{r.feature_idx}.csv")
		else:
			out_csv = os.path.join(lrp_decoder_dir, f"layer{r.layer_idx}_feat{r.feature_idx}.csv")

		# Run with cache disabled to free memory after each run
		calc_lrp.main(
			images_dir=images_dir,
			layer_index=layer_1based,
			feature_index=feature_0based,
			which_module=which_module,
			output_csv=out_csv,
			weights_path=weights_path,
			model_type=model_type,
			use_model_cache=False,
			# Keep the rest as defaults; expose limit_images to be optionally stricter
			#limit_images=(0 if limit_images is None else max(0, int(limit_images))),
		)
		
		# Explicit memory cleanup after each LRP run
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		# Progress indicator (show progress within to_process)
		total_new = len(to_process)
		print(f"✅ LRP {i+1}/{total_new} abgeschlossen: {r.module} L{r.layer_idx} F{feature_0based}")


def main(
	k: int = 5,
	do_lrp: bool = True,
	limit_images: Optional[int] = None,
	*,
	# High-level inputs/outputs you may want to override
	images_dir: Optional[str] = None,
	output_root: Optional[str] = None,
	# Fine-grained overrides (optional; otherwise derived from output_root)
	encoder_rot_dir: Optional[str] = None,
	decoder_dir: Optional[str] = None,
	lrp_out_dir: Optional[str] = None,
	lrp_encoder_dir: Optional[str] = None,
	lrp_decoder_dir: Optional[str] = None,
	summary_csv: Optional[str] = None,
	# Model weights override (optional)
	weights_path: Optional[str] = None,
	# Modelltyp ('car' oder 'butterfly')
	model_type: str = "car",
) -> None:
	# Resolve all effective paths based on overrides
	paths = _compute_paths(
		output_root=output_root,
		images_dir=images_dir,
		encoder_rot_dir=encoder_rot_dir,
		decoder_dir=decoder_dir,
		lrp_out_dir=lrp_out_dir,
		lrp_encoder_dir=lrp_encoder_dir,
		lrp_decoder_dir=lrp_decoder_dir,
		summary_csv=summary_csv,
	)

	rows = find_top_k_per_layer(
		k=k,
		encoder_rot_dir=paths["encoder_rot_dir"],
		decoder_dir=paths["decoder_dir"],
	)
	# Persist and print
	save_summary(rows, paths["summary_csv"])
	print_summary(rows)

	# Execute LRP for each selected feature
	if do_lrp:
		run_lrp_for(
			rows,
			images_dir=paths["images_dir"],
			lrp_encoder_dir=paths["lrp_encoder_dir"],
			lrp_decoder_dir=paths["lrp_decoder_dir"],
			limit_images=limit_images,
			weights_path=weights_path,
			model_type=model_type,
		)


if __name__ == "__main__":
	k_env = os.environ.get("TOP_K", "5")
	do_lrp_env = os.environ.get("DO_LRP", "1")
	limit_env = os.environ.get("LIMIT_IMAGES", "")
	images_dir_env = os.environ.get("IMAGES_DIR", "").strip() or None
	output_root_env = os.environ.get("OUTPUT_ROOT", "").strip() or None
	weights_path_env = os.environ.get("WEIGHTS_PATH", "").strip() or None
	encoder_rot_dir_env = os.environ.get("ENCODER_ROT_DIR", "").strip() or None
	decoder_dir_env = os.environ.get("DECODER_DIR", "").strip() or None
	lrp_out_dir_env = os.environ.get("LRP_OUT_DIR", "").strip() or None
	lrp_encoder_dir_env = os.environ.get("LRP_ENCODER_DIR", "").strip() or None
	lrp_decoder_dir_env = os.environ.get("LRP_DECODER_DIR", "").strip() or None
	summary_csv_env = os.environ.get("SUMMARY_CSV", "").strip() or None

	try:
		k_val = int(k_env)
	except Exception:
		k_val = 5
	do_lrp_flag = do_lrp_env.strip() not in ("0", "false", "False", "no", "NO")
	limit_val: Optional[int] = None
	if limit_env.strip():
		try:
			limit_val = int(limit_env)
		except Exception:
			limit_val = None

	main(
		k=k_val,
		do_lrp=do_lrp_flag,
		limit_images=limit_val,
		images_dir=images_dir_env,
		output_root=output_root_env,
		encoder_rot_dir=encoder_rot_dir_env,
		decoder_dir=decoder_dir_env,
		lrp_out_dir=lrp_out_dir_env,
		lrp_encoder_dir=lrp_encoder_dir_env,
		lrp_decoder_dir=lrp_decoder_dir_env,
		summary_csv=summary_csv_env,
		weights_path=weights_path_env,
	)

