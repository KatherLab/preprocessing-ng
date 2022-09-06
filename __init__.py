
__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.2.1'
__maintainer__ = ['Marko van Treeck', 'Omar El Nahhas']
__email__ = 'markovantreeck@gmail.com'


'''
Version 0.2.0 from 29-08-2022, added Canny edge detector to remove blurry/white
tiles which are not useful. The prior method of filtering everything > 224 pixels
(i.e., white slides) did not account for blur or black tiles, or mostly white tiles.
The edge cut-off (>2) has been hard-coded and adapted from the Normalisation script 
in the old-gen pre-processing script. Note that Canny itself has hard-coded thresholds
as well (40, 100).

Version 0.2.1 from 30-08-2022 added canny as optional input to run
'''
