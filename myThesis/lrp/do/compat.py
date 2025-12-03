"""
Compatibility/monkey patches for Pillow and NumPy.
Must be imported before any other heavy imports.
Note: Do not import torch or any internal modules here.
"""

# PIL/Pillow compatibility fixes
try:
    import PIL.Image  # type: ignore
    # Fix for newer Pillow versions (Resampling enum)
    if not hasattr(PIL.Image, 'LINEAR'):
        PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR
    if not hasattr(PIL.Image, 'CUBIC'):
        PIL.Image.CUBIC = PIL.Image.Resampling.BICUBIC
    if not hasattr(PIL.Image, 'LANCZOS'):
        PIL.Image.LANCZOS = PIL.Image.Resampling.LANCZOS
    if not hasattr(PIL.Image, 'NEAREST'):
        PIL.Image.NEAREST = PIL.Image.Resampling.NEAREST
except Exception as e:  # pragma: no cover - defensive
    print(f"⚠️ PIL fix failed, but continuing: {e}")

# NumPy compatibility fixes
try:
    import numpy as np  # type: ignore
    if not hasattr(np, 'bool'):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, 'int'):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, 'float'):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, 'complex'):
        np.complex = complex  # type: ignore[attr-defined]
except Exception as e:  # pragma: no cover - defensive
    print(f"⚠️ NumPy fix failed, but continuing: {e}")
