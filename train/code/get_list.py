#!/usr/bin/env python

"""
    get_list.py
    
    Example:
        ./code/get_list.py ./data/casia/ > ./data/train.txt
"""

from __future__ import print_function

import os
import sys
import argparse
from glob import glob

if __name__ == "__main__":
    inpath = sys.argv[1]
    
    paths = glob(os.path.join(inpath, '*/*'))
    paths = filter(os.path.isfile, paths)
    paths = map(os.path.abspath, paths)
    paths = sorted(paths)
    
    classes = map(lambda x: x.split('/')[-2], paths)
    uclasses = sorted(list(set(classes)))
    lookup = dict(zip(uclasses, range(len(uclasses))))
    
    for p,c in zip(paths, classes):
        print(p, lookup[c])

