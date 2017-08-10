#!/usr/bin/env python

import sys
from rsub import *
from matplotlib import pyplot as plt

def smart_float(x):
    try:
        return float(x)
    except:
        pass

x = filter(None, map(smart_float, sys.stdin))
_ = plt.plot(x)
show_plot()
