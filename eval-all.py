import os
import sys
import numpy as np
from tqdm import tqdm
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('./tools/caffe-sphereface//python')

import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

from utils import run_lfw, distfuns

# --
# Helpers

def load_image(path, hflip=False, bgr=True):
    img = caffe.io.load_image(path)
    img *= 255
    img = (img - 127.5) / 128
    img = img.transpose((2, 0, 1))
    if hflip:
        img = img[:,:,::-1]
    
    if bgr:
        img = img[::-1,:,:]
    
    return img

# --
# Load model

all_results = []

model = './train/code/sphereface/sphereface_deploy.prototxt'
fs = glob('/home/bjohnson/data/lfw/face_chips/112-25/*/*')
labs = np.array([os.path.basename(f).split('.')[0] for f in fs])

from glob import glob

all_weights = glob('./train/result/sphereface/*.caffemodel')
all_weights = sorted(all_weights)

for weights in all_weights:
    net = caffe.Net(model, weights, caffe.TEST)
    
    all_feats = []
    for chunk in tqdm(np.array_split(fs, 250)):
        imgs = map(load_image, chunk)
        all_feats.append(net.forward_all(data=np.array(imgs))['fc5'])
        
    all_feats = np.vstack(all_feats)
    
    results = run_lfw(all_feats, labs, distfuns)
    results.update({'weights' : weights})
    print results
    all_results.append(results)
