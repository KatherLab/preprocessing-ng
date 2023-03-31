#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from tile import get_scaled_thumb, get_mask_from_thumb, handle_missing_mpp


def test_get_mask_from_thumb():
    thumb = Image.new("L", (10, 10), 128)
    threshold = 100
    mask = get_mask_from_thumb(thumb, threshold)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (10, 10)

if __name__ == "__main__":
    test_get_mask_from_thumb()
    print("All tests passed.")
