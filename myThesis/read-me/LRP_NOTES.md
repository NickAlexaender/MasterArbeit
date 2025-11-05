# LRP Notes (Transformer/MaskDINO)

- Cross-Attention Attribution: Der Value-Pfad attribuiert bewusst auf die Quelle (Encoder-Memory, Länge S). Für Decoder-Cross-Attention entsteht daher eine Relevanzkarte der Form (B, S, C). Dieses Verhalten ist methodisch korrekt, da die Query-Relevanz aus den Quellen gezogen wird.

- Konservativer Residual-Split: In `LRPEngine.run_local(...)` kann optional strikte Konservativität über den Flag `conservative_residual=True` aktiviert werden. Standard ist `False` (kompatibel zum bisherigen Verhalten), wodurch der Skip-Pfad im only_transform-Modus (Attn/FFN) auf 0 gesetzt wird.

- Normalisierung: `norm="sum1"` nutzt die signierte Summe (Summe ≈ 1.0), `norm="sumAbs1"` nutzt die L1-Summe (Summe der Absolutwerte ≈ 1.0).

- Zielrelevanz (token_reduce): `token_reduce="max"` wählt pro Batch den Token mit maximaler Aktivierung (One-Hot); `mean` verteilt gleichmäßig.

- Tensor-Formate: `_to_BTC` ist zentralisiert und nimmt 3D-Inputs unverändert als (B,T,C) an. Bei abweichenden Layouts (z. B. (T,B,C)) bitte vor dem Aufruf explizit transponieren.
