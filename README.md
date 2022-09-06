
# Preprocessing
This repository contains two main functions to preprocess Whole Slide Images(WSI). 

    1. Thumbnail  Extract a thumbnail from each WSI
    2. Tiling     Extract tiles from each WSI

## Tiling 
For each WSI within the given cohort path, a folder containing the extracted tiles from the WSI will be stored in the output path.

Further: Canny Edge Detector is applied to filter out tiles that mainly contain background or are blurred. 

### Command
```
python -m tile.py COHORT_PATH OUTPATH <flags>
```

COHORT_PATH
* Type: PathLike
* A folder containing whole slide images

OUTPATH
* Type: PathLike
* The output folder.

FLAGS(Optional) 
* --tile_size=TILE_SIZE
    * Type: int
    * Default: 224
    * The size of the output tiles in pixels.
* --um_per_tile=UM_PER_TILE
    * Type: float
    * Default: 256.0
    * Size each tile spans in µm.
* --threshold=THRESHOLD
    * Type: int
    * Default: 224
* --force=FORCE
    * Type: bool
    * Default: False
    * Overwrite existing tiles.
* --canny
    * Type: bool
    * Default: True
    * Use Canny edge detector

## Thumbnails ##

For each WSI within the given cohort path, a thumbnail of the WSI will be stored in the output path.

### Command
```
python -m thumbnail.py <COHORT_PATH> <OUTPATH> <FLAGS>
```
COHORT_PATH
* Type: PathLike
* A folder containing whole slide images.

OUTPATH
* Type: PathLike
* The output folder.

FLAGS(Opitional)
* --thumbnail_mpp=THUMBNAIL_MPP
    * Type: float
    * Default: 32.0
    * Micrometers per tile for the thumbnails.
* --force=FORCE
    * Type: bool
    * Default: False
    * Overwrite already existing thumbnails.
