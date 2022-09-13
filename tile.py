#!/usr/bin/env python3

__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.3.0'
__maintainer__ = ['Marko van Treeck', 'Omar El Nahhas']
__email__ = 'markovantreeck@gmail.com'


'''
Version 0.2.0 from 29-08-2022, added Canny edge detector to remove blurry/white
tiles which are not useful. The prior method of filtering everything > 224
pixels (i.e., white slides) did not account for blur or black tiles, or mostly
white tiles.  The edge cut-off (>2) has been hard-coded and adapted from the
Normalisation script in the old-gen pre-processing script. Note that Canny
itself has hard-coded thresholds as well (40, 100).

Version 0.2.1 from 30-08-2022 added canny as optional input to run.

Version 0.3.0 from 06-09-2022 added proper argparser.
'''

import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract tiles from WSIs.')
    parser.add_argument('cohort_path', type=Path)
    parser.add_argument('-o', '--outdir', type=Path)
    parser.add_argument(
        '--tile-size', type=int, default=224,
        help='Size of output tiles.')
    parser.add_argument(
        '--um-per-tile', type=float, default=256.,
        help='Microns covered by each tile.')
    parser.add_argument(
        '--brightness-cutoff', type=int, default=224,
        help='Brightness past which tiles are rejected as background.')
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Overwrite existing tile.')
    parser.add_argument(
        '--no-canny', dest='use_canny', action='store_false',
        help='Disable rejection of edge tiles. Useful for TMAs / sparse slides.')
    args = parser.parse_args()

from contextlib import contextmanager
import os
import shutil
import tempfile
import fire
import numpy as np
from openslide import OpenSlide, PROPERTY_NAME_MPP_X
from concurrent import futures
from tqdm import tqdm
from PIL import Image
from typing import Tuple
from common import supported_extensions
import logging
import cv2


def main(
        cohort_path: Path, outdir: Path,
        tile_size: int = 224, um_per_tile: float = 256.,
        brightness_cutoff: int = 224, force: bool = False, use_canny: bool = True
) -> None:
    """Extracts tiles from whole slide images.

    Args:
        cohort_path:  A folder containing whole slide images.
        outpath:  The output folder.
        tile_size:  The size of the output tiles in pixels.
        um_per_tile:  Size each tile spans in µm.
        force:  Overwrite existing tiles.
    """
    outdir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=outdir/'logfile', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    slides = sum((list(cohort_path.glob(f'**/*.{ext}'))
                  for ext in supported_extensions),
                 start=[])

    submitted_jobs = {}
    with (futures.ThreadPoolExecutor(1) as executor,
          tempdir(prefix='tile-') as tmpdir):
        for i, slide_path in enumerate((progress := tqdm(slides))):
            progress.set_description(slide_path.stem)
            tmp_slide_path = tmpdir/slide_path.name
            shutil.copy(slide_path, tmp_slide_path)

            future = executor.submit(
                extract_tiles,
                slide_path=tmp_slide_path,
                outdir=outdir /
                slide_path.relative_to(cohort_path).parent/slide_path.stem,
                tile_size=tile_size,
                um_per_tile=um_per_tile,
                threshold=brightness_cutoff,
                force=force,
                canny=use_canny)
            submitted_jobs[future] = tmp_slide_path     # to delete later

            while len(submitted_jobs) > 2 or (submitted_jobs and i == len(slides) - 1):
                done, _ = futures.wait(
                    submitted_jobs, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    # delete temporary slide copy
                    submitted_jobs[future].unlink()
                    try:
                        future.result()     # force result to get eventual exceptions
                    except Exception as e:
                        logging.exception(f'{slide_path}: {e}')
                    del submitted_jobs[future]


@contextmanager
def tempdir(*args, **kwargs):
    """A context manager to (hopefully) clean up our tmpdir on a crash."""
    path = Path(tempfile.mkdtemp(*args, **kwargs))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def extract_tiles(
        *,
        slide_path: Path, outdir: Path,
        tile_size: int, um_per_tile: float,
        threshold: int, force: bool, canny: bool
) -> None:
    slide = OpenSlide(str(slide_path))

    tile_size_px, thumb = get_scaled_thumb(slide, um_per_tile)

    # the mask contains the tiles which have survived the threshold criteria of being < 224 pixels
    mask = get_mask_from_thumb(thumb, threshold)

    # coords has dimension of: amount of tiles * (x,y) coords of tiles
    coords = np.flip(np.transpose(mask.nonzero()), 1) * tile_size_px

    outdir.mkdir(exist_ok=True, parents=True)
    with futures.ThreadPoolExecutor(os.cpu_count()) as executor:
        jobs = []
        for c in coords:
            c = c.astype(int)
            fn = outdir/f'{outdir.stem}_({c[0]},{c[1]}).jpg'
            if fn.exists() and not force:
                continue
            future = executor.submit(
                read_and_save_tile,
                slide=slide, outpath=fn, coords=c,
                tile_size_px=tile_size_px, tile_size_out=tile_size,
                use_canny=canny)
            jobs.append(future)

        for future in tqdm(futures.as_completed(jobs), total=len(jobs), leave=False):
            try:
                future.result()
            except Exception as e:
                logging.exception(f'{slide_path}: {e}')


def get_scaled_thumb(slide: OpenSlide, um_per_tile: float) -> Tuple[float, Image.Image]:
    # TODO handle missing mpp
    tile_size_px = um_per_tile/float(slide.properties[PROPERTY_NAME_MPP_X])

    thumb_size = (np.array(slide.dimensions)/tile_size_px).astype(int)
    return tile_size_px, slide.get_thumbnail(thumb_size)


def get_mask_from_thumb(thumb, threshold: int) -> np.ndarray:
    thumb = thumb.convert('L')
    return np.array(thumb) < threshold


def read_and_save_tile(*, slide, outpath, coords, tile_size_px, tile_size_out, use_canny):
    tile = slide.read_region(coords, 0, (int(tile_size_px),)*2)

    # True by default, which runs Canny edge detection
    if use_canny:
        # Below was added in version 0.2.0, using Canny as extra filtering method
        tile_to_greyscale = tile.convert('L')
        # tile_to_greyscale is an PIL.Image.Image with image mode L
        # Note: If you have an L mode image, that means it is
        # a single channel image - normally interpreted as greyscale.
        # The L means that is just stores the Luminance.
        # It is very compact, but only stores a greyscale, not colour.

        tile2array = np.array(tile_to_greyscale)

        # hardcoded thresholds
        edge = cv2.Canny(tile2array, 40, 100)

        # avoid dividing by zero
        edge = (edge / np.max(edge) if np.max(edge) != 0 else 0)
        edge = (((np.sum(np.sum(edge)) / (tile2array.shape[0]*tile2array.shape[1])) * 100)
                if (tile2array.shape[0]*tile2array.shape[1]) != 0 else 0)

        # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
        if(edge < 2.):
            logging.info(
                f'Tile rejected, found 2 or less edges. Tile: {outpath}')
            return

    tile = tile.convert('RGB').resize((tile_size_out,)*2)
    tile.save(outpath)


if __name__ == '__main__':
    main(**vars(args))
