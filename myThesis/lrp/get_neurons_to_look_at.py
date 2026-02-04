import argparse
import csv
import json
import os
import re
from typing import Dict, List, Tuple



REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENCODER_MIOU_BASE = os.path.join(
	REPO_ROOT, "output", "encoder", "network_dissection"
)
DECODER_MIOU_BASE = os.path.join(
	REPO_ROOT, "output", "decoder", "iou_results"
)
LRP_OUT_BASE = os.path.join(REPO_ROOT, "output", "lrp")


layer_dir_pattern = re.compile(r"^layer(\d+)$")


def _ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)

# Wir suchen nach Layer-Unterordnern und den entsprechenden CSV-Dateien

def _find_layer_files(base_dir: str, filename: str) -> List[Tuple[int, str]]:
	if not os.path.isdir(base_dir):
		return []

	entries = []
	for name in os.listdir(base_dir):
		m = layer_dir_pattern.match(name)
		if not m:
			continue
		layer_idx = int(m.group(1))
		fpath = os.path.join(base_dir, name, filename)
		if os.path.isfile(fpath):
			entries.append((layer_idx, fpath))

	return sorted(entries, key=lambda x: x[0])


def _read_csv(path: str) -> List[Dict[str, str]]:
	with open(path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		return [dict(row) for row in reader]


def _write_list_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
	_ensure_dir(os.path.dirname(path))
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _safe_float(v: str, default: float = 0.0) -> float:
	try:
		return float(v)
	except Exception:
		return default

# Pro Layer lesen wir aus, welche Features die mIoU-Schwelle überschreiten

def collect_encoder_features_above_threshold(
	threshold: float,
	base_dir: str = ENCODER_MIOU_BASE,
) -> Dict[int, List[Dict[str, object]]]:
	result: Dict[int, List[Dict[str, object]]] = {}

	for layer_idx, csv_path in _find_layer_files(base_dir, "miou_network_dissection.csv"):
		rows = _read_csv(csv_path)
		selected: List[Dict[str, object]] = []
		for r in rows:
			miou = _safe_float(r.get("miou", "0"))
			if miou >= threshold:
				selected.append(
					{
						"feature_idx": int(r.get("feature_idx", 0)),
						"miou": miou,
						"nd_threshold": _safe_float(r.get("nd_threshold", "nan"), default=float("nan")),
						"n_images": int(r.get("n_images", 0) or 0),
					}
				)
		if selected:
			# absteigend nach mIoU sortieren
			selected.sort(key=lambda d: float(d["miou"]), reverse=True)
			result[layer_idx] = selected

	return result


def save_encoder_selection(
	selection: Dict[int, List[Dict[str, object]]],
	threshold: float,
	out_base: str = LRP_OUT_BASE,
) -> None:
	out_dir = os.path.join(out_base, "encoder")
	_ensure_dir(out_dir)

	summary: Dict[str, object] = {"threshold": threshold, "layers": {}}

	for layer_idx, rows in selection.items():
		# CSV pro Layer
		csv_path = os.path.join(out_dir, f"layer{layer_idx}_selected_features.csv")
		_write_list_csv(
			csv_path,
			rows,
			fieldnames=["feature_idx", "miou", "nd_threshold", "n_images"],
		)

		# Nur Indizes als TXT
		txt_path = os.path.join(out_dir, f"layer{layer_idx}_selected_features.txt")
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write("\n".join(str(int(r["feature_idx"])) for r in rows))

		# In Summary
		summary["layers"][str(layer_idx)] = {
			"count": len(rows),
			"features": [int(r["feature_idx"]) for r in rows],
		}

	# JSON-Summary
	json_path = os.path.join(out_dir, "summary.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)


# Pro Layer lesen wir aus, welche Queries die mIoU-Schwelle überschreiten

def collect_decoder_queries_above_threshold(
	threshold: float,
	base_dir: str = DECODER_MIOU_BASE,
) -> Dict[int, List[Dict[str, object]]]:
	result: Dict[int, List[Dict[str, object]]] = {}

	for layer_idx, csv_path in _find_layer_files(base_dir, "mIoU_per_Query.csv"):
		rows = _read_csv(csv_path)
		selected: List[Dict[str, object]] = []
		for r in rows:
			miou = _safe_float(r.get("mean_iou", "0"))
			if miou >= threshold:
				selected.append(
					{
						"query_idx": int(r.get("query_idx", 0)),
						"mean_iou": miou,
						"num_images": int(r.get("num_images", 0) or 0),
					}
				)
		if selected:
			selected.sort(key=lambda d: float(d["mean_iou"]), reverse=True)
			result[layer_idx] = selected

	return result


def save_decoder_selection(
	selection: Dict[int, List[Dict[str, object]]],
	threshold: float,
	out_base: str = LRP_OUT_BASE,
) -> None:
	out_dir = os.path.join(out_base, "decoder")
	_ensure_dir(out_dir)

	summary: Dict[str, object] = {"threshold": threshold, "layers": {}}

	for layer_idx, rows in selection.items():
		# CSV pro Layer
		csv_path = os.path.join(out_dir, f"layer{layer_idx}_selected_queries.csv")
		_write_list_csv(
			csv_path,
			rows,
			fieldnames=["query_idx", "mean_iou", "num_images"],
		)

		# Nur Indizes als TXT
		txt_path = os.path.join(out_dir, f"layer{layer_idx}_selected_queries.txt")
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write("\n".join(str(int(r["query_idx"])) for r in rows))

		# In Summary
		summary["layers"][str(layer_idx)] = {
			"count": len(rows),
			"queries": [int(r["query_idx"]) for r in rows],
		}

	# JSON-Summary
	json_path = os.path.join(out_dir, "summary.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Liest pro Layer mIoU-CSV-Dateien von Encoder und Decoder und speichert "
			"die Indizes (Features/Queries), die oberhalb einer Schwelle liegen, in myThesis/output/lrp."
		)
	)
	parser.add_argument(
		"--encoder-threshold",
		type=float,
		default=0.20,
		help="Schwelle für Encoder-mIoU (miou >= threshold). Default: 0.20",
	)
	parser.add_argument(
		"--decoder-threshold",
		type=float,
		default=0.11,
		help="Schwelle für Decoder-mIoU (mean_iou >= threshold). Default: 0.11",
	)
	parser.add_argument(
		"--skip-encoder",
		action="store_true",
		help="Überspringt die Auswertung für den Encoder.",
	)
	parser.add_argument(
		"--skip-decoder",
		action="store_true",
		help="Überspringt die Auswertung für den Decoder.",
	)

	args = parser.parse_args()

	print("==> Starte Auswertung für LRP-Auswahl")
	# Kurze Info zu den erwarteten Verzeichnissen
	print(f"Encoder-CSV-Wurzel: {ENCODER_MIOU_BASE} (exists={os.path.isdir(ENCODER_MIOU_BASE)})")
	print(f"Decoder-CSV-Wurzel: {DECODER_MIOU_BASE} (exists={os.path.isdir(DECODER_MIOU_BASE)})")

	if not args.skip_encoder:
		print(f"\n[Encoder] Suche CSVs unter: {ENCODER_MIOU_BASE}")
		enc_sel = collect_encoder_features_above_threshold(args.encoder_threshold)
		if enc_sel:
			save_encoder_selection(enc_sel, args.encoder_threshold)
			total = sum(len(v) for v in enc_sel.values())
			print(f"[Encoder] Fertig. Layer: {len(enc_sel)} | Gesamt-Features: {total}")
		else:
			print("[Encoder] Keine passenden CSVs oder keine Features über Schwelle gefunden.")

	if not args.skip_decoder:
		print(f"\n[Decoder] Suche CSVs unter: {DECODER_MIOU_BASE}")
		dec_sel = collect_decoder_queries_above_threshold(args.decoder_threshold)
		if dec_sel:
			save_decoder_selection(dec_sel, args.decoder_threshold)
			total = sum(len(v) for v in dec_sel.values())
			print(f"[Decoder] Fertig. Layer: {len(dec_sel)} | Gesamt-Queries: {total}")
		else:
			print("[Decoder] Keine passenden CSVs oder keine Queries über Schwelle gefunden.")

	print(f"\nAusgabe-Verzeichnis: {LRP_OUT_BASE}")
	print("Done.")


if __name__ == "__main__":
	main()

