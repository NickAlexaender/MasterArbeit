"""
I/O helpers for images and results.
"""
from __future__ import annotations

import glob
import os
from typing import List


def collect_images(images_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)
