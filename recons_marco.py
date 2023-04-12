#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os
import argparse
import numpy as np
from PIL import Image, ImageFont
from openslide import OpenSlide
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def read_coordinates(file):
    string = file.split('.')[0].split('_')[1]
    x = string.split(',')[0][1:]
    y = string.split(',')[1][:-1]
    return int((int(x) - 1) / int(tile_size_px)), int((int(y) - 1) / int(tile_size_px))

if __name__ == '__main__':
    # parse the arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', type=dir_path)
    parser.add_argument('--feature_path', type=dir_path)
    parser.add_argument('--output', type=dir_path)

    # access the arguments

    # parse the arguments with input and output path

    args = parser.parse_args()

    slide_path = '/home/swarm/Downloads/test_cohort/1031892.svs'
    dirpath = 'output_marco/1031892'
    output_path = 'output_marco/1031892.jpg'

    #slide_path = args.slide_path
    slide = OpenSlide(str(slide_path))
    PROPERTY_NAME_MPP_X = 'openslide.mpp-x'

    print(slide.properties[PROPERTY_NAME_MPP_X])
    um_per_tile = 256
    tile_size_px = um_per_tile / float(slide.properties[PROPERTY_NAME_MPP_X])
    print(tile_size_px)
    thumb_size = (np.array(slide.dimensions) / tile_size_px).astype(int)
    print(thumb_size)
    thumb = slide.get_thumbnail(thumb_size)
    thumb.show()
    array = np.array(thumb)
    coords = np.flip(np.transpose(array.nonzero()), 1) * tile_size_px
    # print(coords[:10])

    for c in coords[:10]:
        c = c.astype(int)
        print(c)

    # get all the files in the directory
    files = os.listdir(dirpath)
    # sort the files
    files.sort()
    coords = []

    for file in files:
        # get the coordinates
        x, y = read_coordinates(file)

        '''
        x = int((x-1)/int(tile_size_px))
        y = int(((y-1)/int(tile_size_px)))
        #print(x, y)
        '''
        coords.append([x, y])

    # get the largest x and y
    max_x = max([c[0] for c in coords])
    max_y = max([c[1] for c in coords])
    print(max_x, max_y)

    img_big = np.zeros(((max_y + 1) * 224, (max_x + 1) * 224, 3))
    print(img_big.shape)

    for file in files:
        # get the coordinates
        x, y = read_coordinates(file)
        # print(x, y)
        # open tile_path as image
        tile = Image.open(dirpath + '/' + file)
        # norm_tile = Image.fromarray(np.array(file).astype(np.uint8))
        # tile = tile.convert('RGB')
        # convert to numpy array
        tile_array = np.array(tile)
        # print(tile_array.shape)
        x = x * 224
        y = y * 224
        img_big[y:y + 224, x:x + 224, ] = np.array(tile)
    img_big = img_big.astype(np.uint8)
    # convert to image
    img = Image.fromarray(img_big)
    # save image
    img.save(output_path)