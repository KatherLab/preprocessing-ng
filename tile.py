#!/usr/bin/env python3

__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2021, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'markovantreeck@gmail.com'

import os
import shutil
import tempfile
import fire
import numpy as np
from openslide import OpenSlide, PROPERTY_NAME_MPP_X
from concurrent import futures
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from typing import Tuple
from common import supported_extensions
import time
import logging
import queue


def main(
        cohort_path: os.PathLike, outpath: os.PathLike,
        tile_size: int = 224, um_per_tile: float = 256.,
        threshold: int = 224, force: bool = False) -> None:
    """Extracts tiles from whole slide images.

    Args:
        cohort_path:  A folder containing whole slide images.
        outpath:  The output folder.
        tile_size:  The size of the output tiles in pixels.
        um_per_tile:  Size each tile spans in µm.
        force:  Overwrite existing tiles.
    """
    cohort_path, outpath = Path(cohort_path), Path(outpath)
    logging.basicConfig(filename=outpath/'logfile', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    slides = sum((list(cohort_path.glob(f'**/*.{ext}'))
                  for ext in supported_extensions),
                 start=[])

    tmpdir = tempfile.mkdtemp(prefix='tile')
    submitted_jobs = {}
    with futures.ThreadPoolExecutor(1) as executor:
        for i, slide_path in enumerate((progress := tqdm(slides))):
            try:
                progress.set_description(slide_path.stem)
                tmp_slide_path = Path(tmpdir)/slide_path.name
                shutil.copy(slide_path, tmp_slide_path)
                future = executor.submit(
                    extract_tiles,
                    tmp_slide_path,
                    outpath/slide_path.relative_to(cohort_path).parent/slide_path.stem,
                    tile_size=tile_size,
                    um_per_tile=um_per_tile,
                    threshold=threshold,
                    force=force)
                submitted_jobs[future] = tmp_slide_path
                if len(submitted_jobs) > 2:
                    done, _ = futures.wait(submitted_jobs, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        submitted_jobs[future].unlink()
                        del submitted_jobs[future]
            except Exception as e:
                logging.exception(f'{tmp_slide_path}: {e}')


def extract_tiles(
        slide_path: Path, outdir: Path,
        *,
        tile_size: int, um_per_tile: float,
        threshold: int, force: bool
) -> None:
    slide = OpenSlide(str(slide_path))

    tile_size_px, thumb = get_scaled_thumb(slide, um_per_tile)
    mask = get_mask_from_thumb(thumb, threshold)
    coords = np.flip(np.transpose(mask.nonzero()), 1) * tile_size_px

    outdir.mkdir(exist_ok=True, parents=True)
    with futures.ThreadPoolExecutor(os.cpu_count()) as e:
        for c in coords:
            c = c.astype(int)
            fn = outdir/f'{outdir.stem}_({c[0]},{c[1]}).jpg'
            if fn.exists() and not force:
                continue
            e.submit(read_and_save_tile, slide, fn, c, tile_size_px, tile_size)


def get_scaled_thumb(slide: OpenSlide, um_per_tile: float) -> Tuple[float, Image.Image]:
    # TODO handle missing mpp
    tile_size_px = um_per_tile/float(slide.properties[PROPERTY_NAME_MPP_X])

    thumb_size = (np.array(slide.dimensions)/tile_size_px).astype(int)
    return tile_size_px, slide.get_thumbnail(thumb_size)


def get_mask_from_thumb(thumb, threshold: int) -> np.ndarray:
    thumb = thumb.convert('L')
    return np.array(thumb) < threshold


def read_and_save_tile(slide, outpath, coords, tile_size_px, tile_size_out):
    tile = slide.read_region(coords, 0, (int(tile_size_px),)*2)
    tile = tile.convert('RGB').resize((tile_size_out,)*2)
    tile.save(outpath)


if __name__ == '__main__':
    fire.Fire(main)
