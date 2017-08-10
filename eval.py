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

model = './train/code/sphereface/sphereface_deploy.prototxt'
weights = './train/result/sphereface/sphereface_model-2_iter_40000.caffemodel'
net = caffe.Net(model, weights, caffe.TEST)

# --
# Featurize LFW

# fs = glob('/home/bjohnson/software/facenet/data/lfw/mtcnnpy_112/*/*png')
fs = glob('/home/bjohnson/data/lfw/face_chips/112-25/*/*')

all_feats = []
for chunk in tqdm(np.array_split(fs, 200)):
    imgs = map(load_image, chunk)
    all_feats.append(net.forward_all(data=np.array(imgs))['fc5'])

all_feats = np.vstack(all_feats)


rev_feats = []
for chunk in tqdm(np.array_split(fs, 200)):
    imgs = map(lambda x: load_image(x, hflip=True), chunk)
    rev_feats.append(net.forward_all(data=np.array(imgs))['fc5'])

rev_feats = np.vstack(rev_feats)


fs = np.array(fs)

np.save('./labs', fs)
np.save('./all_feats-bgr', all_feats)
np.save('./rev_feats-bgr', rev_feats)

# --
# Run LFW

labs = np.array([os.path.basename(f).split('.')[0] for f in fs])
run_lfw(all_feats, labs, distfuns)
