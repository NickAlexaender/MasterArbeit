"""Streaming-Statistiken für Feature-Heatmaps (ohne Speicherung kompletter Karten).

Dieses Modul ersetzt den alten, speicherhungrigen Aggregator durch eine
leichtgewichtige Statistik-Komponente je Feature. Sie unterstützt zwei
Approximationen für die Perzentil-Bestimmung über alle Pixelwerte eines
Features:

- "reservoir": Einfache Reservoir-Samples mit fester Größe (ein Pass)
- "histogram": Zweipass-Histogramm mit fester Bin-Anzahl

Hinweis: Die IoU-Berechnung erfolgt nicht mehr hier, sondern direkt in der
Pipeline (Pass 2), indem Heatmaps on-the-fly erneut erzeugt werden.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .config import NETWORK_DISSECTION_PERCENTILE

logger = logging.getLogger(__name__)


class FeatureStats:
    """Streaming-Statistiken pro (layer_idx, feature_idx).

    Zwei Modi:
    - reservoir: behält bis zu ``reservoir_size`` Beispielwerte (float32)
    - histogram: benötigt zwei Scans; zuerst min/max sammeln, dann Bins füllen

    Wichtig: Diese Klasse speichert keine Heatmaps und keine Masken.
    """

    def __init__(
        self,
        layer_idx: int,
        feature_idx: int,
        method: str = "reservoir",
        reservoir_size: int = 200_000,
        num_bins: int = 1024,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.feature_idx = int(feature_idx)
        self.method = str(method)
        if self.method not in ("reservoir", "histogram"):
            raise ValueError("FeatureStats.method must be 'reservoir' or 'histogram'")

        # Reservoir-Daten
        self._reservoir_size = int(max(1, reservoir_size))
        self._reservoir: Optional[np.ndarray] = None  # float32
        self._seen: int = 0  # Anzahl gesehener Pixelwerte

        # Histogramm-Daten (min/max in Pass 1a; counts in Pass 1b)
        self._num_bins = int(max(2, num_bins))
        self._min: float = np.inf
        self._max: float = -np.inf
        self._counts: Optional[np.ndarray] = None  # int64, Länge num_bins
        self._total_count: int = 0

    # -----------------
    # Reservoir-Modus
    # -----------------
    def _ingest_reservoir(self, values: np.ndarray) -> None:
        """Fügt Werte einem Reservoir-Sample hinzu (Algorithmus R, batch-weise).

        Hinweis: Für Effizienz wird batch-weise gearbeitet. Für den initialen
        Füllstand werden die ersten K gesehenen Werte übernommen, danach wird
        eine Untermenge durch Zufall ersetzt. Der Algorithmus ist eine
        vektorisiert umgesetzte Variante von Vitter's R (annähernd äquivalent
        bei großen Batches)."""
        v = values.astype(np.float32, copy=False).ravel()
        n = v.size
        if n == 0:
            return

        # Initiale Befüllung
        if self._reservoir is None:
            take = min(self._reservoir_size, n)
            self._reservoir = np.empty((self._reservoir_size,), dtype=np.float32)
            self._reservoir[:take] = v[:take]
            self._seen = take
            # Restliche Batch-Kandidaten unterliegen bereits der Ersetzung
            start = take
        else:
            start = 0

        remaining = n - start
        if remaining <= 0:
            return

        # Für die verbleibenden 'remaining' Werte: Ersetzungswahrscheinlichkeit
        # j ~ U[0, self._seen + i), ersetze wenn j < K. Wir approximieren dies
        # batchweise: Ziehe Zufallsindices im Bereich [0, self._seen + remaining)
        # und behalte diejenigen < K. Diese Näherung ist in der Praxis hinreichend.
        K = self._reservoir_size
        seen_before = self._seen
        # Zufallsindices für alle Kandidaten der Batch im großen Bereich
        big_range = seen_before + np.arange(1, remaining + 1, dtype=np.int64)
        # Uniforme Zufallszahlen im jeweiligen Bereich simulieren, indem wir
        # relative Uniforms ziehen und skalieren.
        # Achtung: Verwende float32 RNG, dann runden.
        rng = np.random.random_sample(remaining).astype(np.float32)
        j = (rng * big_range).astype(np.int64)
        # Kandidaten mit j < K werden in das Reservoir aufgenommen
        mask = j < K
        idxs = j[mask]
        vals = v[start:][mask]
        if idxs.size:
            self._reservoir[idxs] = vals
        self._seen = int(seen_before + remaining)

    # -----------------
    # Histogramm-Modus
    # -----------------
    def observe_minmax(self, values: np.ndarray) -> None:
        v = values.astype(np.float32, copy=False)
        if v.size == 0:
            return
        mn = float(np.min(v))
        mx = float(np.max(v))
        if mn < self._min:
            self._min = mn
        if mx > self._max:
            self._max = mx

    def prepare_hist(self) -> None:
        if not np.isfinite(self._min) or not np.isfinite(self._max):
            # Kein Datenpunkt gesehen
            self._min = 0.0
            self._max = 0.0
        if self._max <= self._min:
            # Verhindere Division-by-zero; erzeuge minimale Spannweite
            eps = 1e-6
            self._max = self._min + eps
        self._counts = np.zeros((self._num_bins,), dtype=np.int64)
        self._total_count = 0

    def ingest_hist(self, values: np.ndarray) -> None:
        if self._counts is None:
            raise RuntimeError("Histogram not prepared. Call prepare_hist() first.")
        v = values.astype(np.float32, copy=False)
        if v.size == 0:
            return
        # np.histogram ist in C implementiert und effizient
        cnts, _ = np.histogram(v, bins=self._num_bins, range=(self._min, self._max))
        self._counts += cnts.astype(np.int64, copy=False)
        self._total_count += int(v.size)

    # -----------------
    # Gemeinsame API
    # -----------------
    def ingest_heatmap(self, heatmap: np.ndarray, stage: str = "auto") -> None:
        """Streamt eine Heatmap in die Statistiken.

        Args:
            heatmap: 2D float32 Array
            stage:   Für 'histogram': 'minmax' oder 'hist'. Für 'reservoir': ignoriert.
        """
        flat = np.asarray(heatmap, dtype=np.float32).ravel()
        if self.method == "reservoir":
            self._ingest_reservoir(flat)
        else:
            if stage == "minmax" or stage == "auto":
                # In 'auto' gehen wir davon aus, dass Pass 1a läuft
                self.observe_minmax(flat)
            elif stage == "hist":
                self.ingest_hist(flat)
            else:
                raise ValueError("Unknown stage for histogram mode: expected 'minmax' or 'hist'")

    def compute_threshold(self, percentile: Optional[float]) -> float:
        if percentile is None:
            percentile = NETWORK_DISSECTION_PERCENTILE
        p = float(percentile)
        p = max(0.0, min(100.0, p))

        if self.method == "reservoir":
            if self._reservoir is None or self._seen == 0:
                return 0.0
            return float(np.percentile(self._reservoir[: self._reservoir_size], p))
        # histogram
        if self._counts is None or self._total_count == 0:
            return 0.0
        target_rank = int(np.ceil((p / 100.0) * float(self._total_count)))
        target_rank = max(1, target_rank)
        cumsum = np.cumsum(self._counts, dtype=np.int64)
        bin_idx = int(np.searchsorted(cumsum, target_rank, side="left"))
        bin_idx = min(max(bin_idx, 0), self._num_bins - 1)
        # Schwellenwert als obere Bin-Grenze (konservativ)
        bin_width = (self._max - self._min) / float(self._num_bins)
        thr = self._min + (bin_idx + 1) * bin_width
        return float(thr)


__all__ = [
    "FeatureStats",
]
