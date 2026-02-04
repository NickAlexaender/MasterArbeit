from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .config import NETWORK_DISSECTION_PERCENTILE

logger = logging.getLogger(__name__)

# Diese Klasse sammelt Statistiken laufend, also während Daten vorbeikommen, und zwar für jede Kombination aus

class FeatureStats:

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

        self._reservoir_size = int(max(1, reservoir_size))
        self._reservoir: Optional[np.ndarray] = None
        self._seen: int = 0 

        self._num_bins = int(max(2, num_bins))
        self._min: float = np.inf
        self._max: float = -np.inf
        self._counts: Optional[np.ndarray] = None
        self._total_count: int = 0

# im ersten Ansatz merkt sich das Programm zufällige Werte aus dem großen Datenstrom ohne alles zu speichern.

    def _ingest_reservoir(self, values: np.ndarray) -> None:
        v = values.astype(np.float32, copy=False).ravel()
        n = v.size
        if n == 0:
            return

        if self._reservoir is None:
            take = min(self._reservoir_size, n)
            self._reservoir = np.empty((self._reservoir_size,), dtype=np.float32)
            self._reservoir[:take] = v[:take]
            self._seen = take
            start = take
        else:
            start = 0

        remaining = n - start
        if remaining <= 0:
            return

        K = self._reservoir_size
        seen_before = self._seen
        big_range = seen_before + np.arange(1, remaining + 1, dtype=np.int64)
        rng = np.random.random_sample(remaining).astype(np.float32)
        j = (rng * big_range).astype(np.int64)
        mask = j < K
        idxs = j[mask]
        vals = v[start:][mask]
        if idxs.size:
            self._reservoir[idxs] = vals
        self._seen = int(seen_before + remaining)

# Als zweiter Ansatz sammeln wir ein Histogramm in Bins

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
            self._min = 0.0
            self._max = 0.0
        if self._max <= self._min:
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
        cnts, _ = np.histogram(v, bins=self._num_bins, range=(self._min, self._max))
        self._counts += cnts.astype(np.int64, copy=False)
        self._total_count += int(v.size)

# Nun nehmen wir die Heatmap und führen die Zahlen nach und nach in unsere statistische Funktion -> nur die Zahlenwerte 

    def ingest_heatmap(self, heatmap: np.ndarray, stage: str = "auto") -> None:
        flat = np.asarray(heatmap, dtype=np.float32).ravel()
        if self.method == "reservoir":
            self._ingest_reservoir(flat)
        else:
            if stage == "minmax" or stage == "auto":
                self.observe_minmax(flat)
            elif stage == "hist":
                self.ingest_hist(flat)
            else:
                raise ValueError("Unknown stage for histogram mode: expected 'minmax' or 'hist'")

# Aus den gesammelten Daten können wir dann den Schwellwert für die Binarisierung berechnen.
# Eine Berechnung auf diese Weise spart Speicherplatz und Rechenzeit

    def compute_threshold(self, percentile: Optional[float]) -> float:
        if percentile is None:
            percentile = NETWORK_DISSECTION_PERCENTILE
        p = float(percentile)
        p = max(0.0, min(100.0, p))

        if self.method == "reservoir":
            if self._reservoir is None or self._seen == 0:
                return 0.0
            return float(np.percentile(self._reservoir[: self._reservoir_size], p))
        if self._counts is None or self._total_count == 0:
            return 0.0
        target_rank = int(np.ceil((p / 100.0) * float(self._total_count)))
        target_rank = max(1, target_rank)
        cumsum = np.cumsum(self._counts, dtype=np.int64)
        bin_idx = int(np.searchsorted(cumsum, target_rank, side="left"))
        bin_idx = min(max(bin_idx, 0), self._num_bins - 1)
        bin_width = (self._max - self._min) / float(self._num_bins)
        thr = self._min + (bin_idx + 1) * bin_width
        return float(thr)


__all__ = [
    "FeatureStats",
]
