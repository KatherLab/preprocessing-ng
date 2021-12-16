#!/usr/bin/env python3

from tile import main as tile
from thumbnail import main as thumbnail
import fire

if __name__ == '__main__':
    fire.Fire({'tile': tile, 'thumbnail': thumbnail})