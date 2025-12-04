"""
Validierungsskript für LRP-Ergebnisse.

Prüft ob die gespeicherten CSV-Dateien sinnvolle Relevanz-Werte enthalten:
1. Normalisierung: sum(normalized_relevance) ≈ 1 oder sum(|normalized_relevance|) ≈ 1
2. Keine NaN/Inf Werte
3. Sinnvolle Verteilung (nicht alle Werte gleich, nicht alle null)
4. Konservierungsprüfung (falls möglich)

Verwendung:
    python -m myThesis.lrp.validate_lrp_results [--csv PATH] [--dir PATH]
    
Beispiele:
    # Einzelne CSV validieren
    python -m myThesis.lrp.validate_lrp_results --csv output/lrp_result.csv
    
    # Alle CSVs in einem Ordner validieren
    python -m myThesis.lrp.validate_lrp_results --dir output/car/finetune6/lrp/tests
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Logger einrichten
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("lrp.validate")


@dataclass
class ValidationResult:
    """Ergebnis der Validierung einer CSV-Datei."""
    
    filepath: str
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    
    def add_error(self, msg: str) -> None:
        """Fügt einen Fehler hinzu und markiert als ungültig."""
        self.errors.append(msg)
        self.is_valid = False
    
    def add_warning(self, msg: str) -> None:
        """Fügt eine Warnung hinzu."""
        self.warnings.append(msg)
    
    def summary(self) -> str:
        """Gibt eine Zusammenfassung des Validierungsergebnisses zurück."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines = [f"\n{'='*60}", f"Datei: {self.filepath}", f"Status: {status}"]
        
        if self.metrics:
            lines.append("\nMetriken:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.6f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        if self.errors:
            lines.append("\n❌ Fehler:")
            for err in self.errors:
                lines.append(f"  - {err}")
        
        if self.warnings:
            lines.append("\n⚠️  Warnungen:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        
        lines.append("="*60)
        return "\n".join(lines)


def validate_csv(filepath: str, verbose: bool = True) -> ValidationResult:
    """
    Validiert eine LRP-Ergebnis-CSV-Datei.
    
    Args:
        filepath: Pfad zur CSV-Datei
        verbose: Ausführliche Ausgabe
        
    Returns:
        ValidationResult mit Validierungsergebnissen
    """
    result = ValidationResult(filepath=filepath)
    
    # Datei laden
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        result.add_error(f"Konnte CSV nicht laden: {e}")
        return result
    
    # Grundlegende Struktur prüfen
    required_cols = ["relevance"]
    for col in required_cols:
        if col not in df.columns:
            result.add_error(f"Erforderliche Spalte '{col}' fehlt")
    
    if not result.is_valid:
        return result
    
    # Relevanz-Werte extrahieren
    relevance = df["relevance"].values
    
    # Metadaten extrahieren (falls vorhanden)
    module_role = df["module_role"].iloc[0] if "module_role" in df.columns else "Unknown"
    layer_name = df["layer_name"].iloc[0] if "layer_name" in df.columns else "Unknown"
    num_images = df["num_images"].iloc[0] if "num_images" in df.columns else "Unknown"
    
    result.metrics["module_role"] = module_role
    result.metrics["layer_name"] = layer_name
    result.metrics["num_entries"] = len(df)
    result.metrics["num_images"] = num_images
    
    # ==================================================================
    # Test 1: Keine NaN/Inf Werte
    # ==================================================================
    nan_count = np.isnan(relevance).sum()
    inf_count = np.isinf(relevance).sum()
    
    if nan_count > 0:
        result.add_error(f"{nan_count} NaN-Werte in 'relevance'")
    if inf_count > 0:
        result.add_error(f"{inf_count} Inf-Werte in 'relevance'")
    
    # Für weitere Tests nur endliche Werte verwenden
    finite_mask = np.isfinite(relevance)
    relevance_finite = relevance[finite_mask]
    
    if len(relevance_finite) == 0:
        result.add_error("Keine endlichen Relevanz-Werte vorhanden")
        return result
    
    # ==================================================================
    # Test 2: Normalisierung prüfen
    # ==================================================================
    # Bei sum1 Normalisierung sollte die Summe ≈ 1 sein (oder -1 bei negativer Relevanz)
    rel_sum = relevance_finite.sum()
    rel_abs_sum = np.abs(relevance_finite).sum()
    
    result.metrics["relevance_sum"] = rel_sum
    result.metrics["relevance_abs_sum"] = rel_abs_sum
    
    # normalized_relevance prüfen (falls vorhanden)
    if "normalized_relevance" in df.columns:
        norm_rel = df["normalized_relevance"].values
        norm_rel_finite = norm_rel[np.isfinite(norm_rel)]
        
        norm_abs_sum = np.abs(norm_rel_finite).sum()
        result.metrics["normalized_abs_sum"] = norm_abs_sum
        
        # Die Summe der absoluten normalisierten Relevanzen sollte ≈ 1 sein
        if abs(norm_abs_sum - 1.0) > 0.01:
            result.add_warning(
                f"Normalisierte Relevanz-Summe weicht von 1 ab: {norm_abs_sum:.6f}"
            )
    
    # ==================================================================
    # Test 3: Sinnvolle Verteilung
    # ==================================================================
    rel_mean = relevance_finite.mean()
    rel_std = relevance_finite.std()
    rel_min = relevance_finite.min()
    rel_max = relevance_finite.max()
    
    result.metrics["relevance_mean"] = rel_mean
    result.metrics["relevance_std"] = rel_std
    result.metrics["relevance_min"] = rel_min
    result.metrics["relevance_max"] = rel_max
    
    # Prüfe ob Werte nicht alle gleich sind
    if rel_std < 1e-12:
        result.add_error("Alle Relevanz-Werte sind identisch (std ≈ 0)")
    
    # Prüfe ob Werte nicht alle null sind
    if rel_abs_sum < 1e-12:
        result.add_error("Alle Relevanz-Werte sind null")
    
    # Prüfe ob es positive UND negative Werte gibt (üblich bei LRP)
    has_positive = (relevance_finite > 0).any()
    has_negative = (relevance_finite < 0).any()
    
    result.metrics["has_positive"] = has_positive
    result.metrics["has_negative"] = has_negative
    
    if not has_positive and not has_negative:
        result.add_warning("Keine positiven oder negativen Werte (alle ≈ 0)")
    
    # ==================================================================
    # Test 4: Verteilungsstatistiken
    # ==================================================================
    # Top-K Relevanz (wie viel % der Gesamtrelevanz haben die Top-10?)
    sorted_abs = np.sort(np.abs(relevance_finite))[::-1]
    top_10_ratio = sorted_abs[:10].sum() / (sorted_abs.sum() + 1e-12)
    top_20_ratio = sorted_abs[:20].sum() / (sorted_abs.sum() + 1e-12)
    
    result.metrics["top_10_concentration"] = top_10_ratio
    result.metrics["top_20_concentration"] = top_20_ratio
    
    # Bei guter LRP sollte die Relevanz nicht zu stark konzentriert sein
    # (außer es gibt wirklich nur wenige wichtige Features)
    if top_10_ratio > 0.95 and len(relevance_finite) > 50:
        result.add_warning(
            f"Relevanz stark konzentriert: Top-10 = {top_10_ratio*100:.1f}% der Gesamtrelevanz"
        )
    
    # ==================================================================
    # Test 5: Modul-spezifische Prüfungen
    # ==================================================================
    if module_role == "Encoder":
        # Encoder sollte 256 Features haben (hidden_dim)
        expected_features = 256
        if len(df) != expected_features:
            result.add_warning(
                f"Encoder: Erwartete {expected_features} Features, gefunden: {len(df)}"
            )
    
    elif module_role == "Decoder":
        # Decoder sollte 300 Queries haben
        expected_queries = 300
        if len(df) != expected_queries:
            result.add_warning(
                f"Decoder: Erwartete {expected_queries} Queries, gefunden: {len(df)}"
            )
    
    return result


def validate_directory(dirpath: str, recursive: bool = True, verbose: bool = True) -> List[ValidationResult]:
    """
    Validiert alle CSV-Dateien in einem Verzeichnis.
    
    Args:
        dirpath: Pfad zum Verzeichnis
        recursive: Auch Unterverzeichnisse durchsuchen
        verbose: Ausführliche Ausgabe
        
    Returns:
        Liste von ValidationResult für jede gefundene CSV
    """
    pattern = os.path.join(dirpath, "**/*.csv") if recursive else os.path.join(dirpath, "*.csv")
    csv_files = glob.glob(pattern, recursive=recursive)
    
    if not csv_files:
        logger.warning(f"Keine CSV-Dateien in {dirpath} gefunden")
        return []
    
    logger.info(f"Validiere {len(csv_files)} CSV-Dateien in {dirpath}")
    
    results = []
    for csv_file in sorted(csv_files):
        result = validate_csv(csv_file, verbose=verbose)
        results.append(result)
        
        if verbose:
            print(result.summary())
    
    return results


def print_summary(results: List[ValidationResult]) -> None:
    """Druckt eine Gesamtzusammenfassung aller Validierungen."""
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    invalid = total - valid
    warnings = sum(len(r.warnings) for r in results)
    
    print("\n" + "="*60)
    print("GESAMTZUSAMMENFASSUNG")
    print("="*60)
    print(f"Dateien validiert: {total}")
    print(f"  ✅ Gültig: {valid}")
    print(f"  ❌ Ungültig: {invalid}")
    print(f"  ⚠️  Warnungen gesamt: {warnings}")
    
    if invalid > 0:
        print("\nUngültige Dateien:")
        for r in results:
            if not r.is_valid:
                print(f"  - {r.filepath}")
                for err in r.errors:
                    print(f"      ❌ {err}")
    
    print("="*60)


def main():
    """Hauptfunktion für CLI-Aufruf."""
    parser = argparse.ArgumentParser(
        description="Validiert LRP-Ergebnis-CSV-Dateien",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Pfad zu einer einzelnen CSV-Datei",
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        help="Pfad zu einem Verzeichnis mit CSV-Dateien",
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Unterverzeichnisse nicht durchsuchen",
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Nur Zusammenfassung ausgeben",
    )
    
    args = parser.parse_args()
    
    if not args.csv and not args.dir:
        # Standard: Validiere bekannte Ausgabepfade
        default_paths = [
            "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/lrp_result.csv",
            "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/lrp/tests",
        ]
        
        results = []
        for path in default_paths:
            if os.path.isfile(path):
                r = validate_csv(path, verbose=not args.quiet)
                results.append(r)
                if not args.quiet:
                    print(r.summary())
            elif os.path.isdir(path):
                results.extend(validate_directory(
                    path, 
                    recursive=not args.no_recursive,
                    verbose=not args.quiet
                ))
        
        if results:
            print_summary(results)
        else:
            logger.error("Keine Dateien zum Validieren gefunden")
            sys.exit(1)
    
    elif args.csv:
        if not os.path.isfile(args.csv):
            logger.error(f"Datei nicht gefunden: {args.csv}")
            sys.exit(1)
        
        result = validate_csv(args.csv, verbose=not args.quiet)
        print(result.summary())
        
        if not result.is_valid:
            sys.exit(1)
    
    elif args.dir:
        if not os.path.isdir(args.dir):
            logger.error(f"Verzeichnis nicht gefunden: {args.dir}")
            sys.exit(1)
        
        results = validate_directory(
            args.dir,
            recursive=not args.no_recursive,
            verbose=not args.quiet,
        )
        
        if results:
            print_summary(results)
            if any(not r.is_valid for r in results):
                sys.exit(1)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
