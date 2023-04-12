#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
import openslide
from pathlib import Path
slide = openslide.OpenSlide(Path('/mnt/sda1/vsi/all/22H02832_2_HES.vsi'))
print(slide.properties)