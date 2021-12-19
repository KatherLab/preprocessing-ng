#!/usr/bin/env python3

__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2021, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'markovantreeck@gmail.com'

import os
import fire
import numpy as np
from openslide import OpenSlide, PROPERTY_NAME_MPP_X
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from common import supported_extensions


def main(
        cohort_path: os.PathLike, outpath: os.PathLike,
        thumbnail_mpp: float = 32., force: bool = False) -> None:
    """Extracts thumbnails from whole slide images.

    Args:
        thumbnail_mpp:  Micrometers per tile for the thumbnails.
        force:  Overwrite already existing thumbnails.
    """
    cohort_path, outpath = Path(cohort_path), Path(outpath)
    slides = sum((list(cohort_path.glob(f'**/*.{ext}'))
                  for ext in supported_extensions),
                 start=[])

    outpath.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor() as e:
        futures = {
            e.submit(extract_thumbnail, slide_path, outpath, thumbnail_mpp, force)
            for slide_path in slides}
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass


def extract_thumbnail(
        slide_path: Path, outpath: Path,
        thumbnail_mpp: float, force: bool) -> None:

    fn = outpath/f'{slide_path.stem}.jpg'
    if fn.exists() and not force:
        return

    slide = OpenSlide(str(slide_path))
    slide \
        .get_thumbnail(np.array(slide.dimensions) *
                       float(slide.properties[PROPERTY_NAME_MPP_X]) / thumbnail_mpp) \
        .convert('RGB') \
        .save(fn)


if __name__ == '__main__':
    fire.Fire(main)
