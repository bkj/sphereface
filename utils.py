#!/usr/bin/env python

"""
    utils.py
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold

def load_pairs():
    """ load canonical LFW pairs """
    pair_path = '/home/bjohnson/data/lfw/sets/pairs.txt'
    pairs = map(lambda x: x.strip().split('\t'), open(pair_path).read().splitlines())[1:]
    
    for i,p in enumerate(pairs):
        if len(p) == 3:
            pairs[i] = (
                '{}_{:04}'.format(p[0], int(p[1])), 
                '{}_{:04}'.format(p[0], int(p[2])), 
                True,
            )
        else:
            pairs[i] = (
                '{}_{:04}'.format(p[0], int(p[1])), 
                '{}_{:04}'.format(p[2], int(p[3])), 
                False,
            )
        
    return pd.DataFrame(pairs, columns=('a', 'b', 'lab'))


def rand_pairs(seed=987):
    """ randomly generate LFW pairs """
    # path = '/home/bjohnson/software/facenet/data/lfw/mtcnnpy_160'
    path = '/home/bjohnson/software/facenet/data/lfw/mtcnnpy_182'
    
    fs = sorted(glob(os.path.join(path, '*', '*')))
    fs = map(lambda x: os.path.basename(x).split('.')[0], fs)
    
    tmp = pd.DataFrame({'a' : fs})
    tmp['id'] = tmp.a.str.replace('_[0-9]+$', '')
    
    # Positive examples -- sample by id
    pos = tmp.groupby('id').a.apply(lambda x: x.sample(x.shape[0], replace=False, random_state=seed)).reset_index(drop=True)
    pos = pd.DataFrame({'a' : fs, 'b' : pos, 'lab' : True})
    pos = pos[pos.a.str.replace('.*_', '') != pos.b.str.replace('.*_', '')]
    
    # Negative examples -- sample randomly
    neg = tmp.a.sample(tmp.shape[0], replace=False, random_state=seed)
    neg = pd.DataFrame({'a' : fs, 'b' : neg, 'lab' : False})
    neg = neg[neg.a.str.replace('_[0-9]+$', '') != neg.b.str.replace('_[0-9]+$', '')]
    
    return pd.concat([pos, neg], 0).reset_index(drop=True)


def tuned_accuracy(pairs, dist, k=5, seed=123):
    res = np.zeros(pairs.shape[0]) - 1
    ts = np.linspace(*np.percentile(dist, (25, 75)), num=100)
    for train, test in KFold(k, random_state=seed).split(range(pairs.shape[0])):
        accs = [(pairs.lab[train] == (dist[train] < t)).mean() for t in ts]
        t = ts[np.argmax(accs)]
        res[test] = pairs.lab[test] == (dist[test] < t)
        
    return res.mean()


def run_lfw(feats, labs, distfuns, default=True, seed=987):
    if default:
        pairs = load_pairs()
    else:
        pairs = rand_pairs(seed=seed)
    
    labs = pd.Series(np.arange(len(labs)), index=labs)
    
    pairs = pairs[pairs.a.isin(labs.index) & pairs.b.isin(labs.index)]
    
    feats_a = np.array(feats[labs.loc[pairs.a]])
    feats_b = np.array(feats[labs.loc[pairs.b]])
    
    out = {}
    for distname, distfun in distfuns.items():
        dist = distfun(feats_a, feats_b)
        
        auc = metrics.roc_auc_score(np.array(pairs.lab), - dist)
        acc = tuned_accuracy(pairs, dist)
        
        out.update({"%s_acc" % distname : acc, "%s_auc" % distname : auc})
    
    return out


# --
# Distance functions

def sq_euclidean_normed(a, b):
    return ((normalize(a) - normalize(b)) ** 2).sum(axis=1)

def sq_euclidean(a, b):
    return ((a - b) ** 2).sum(axis=1)

distfuns = {
    "sq_euclidean_normed" : sq_euclidean_normed,
    "sq_euclidean" : sq_euclidean
}
