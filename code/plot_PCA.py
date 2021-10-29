import torch
import clip
import argparse
from glob import glob
import os
import os.path as op
import numpy as np
from ipdb import set_trace
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
plt.ion()

from utils import *

parser = argparse.ArgumentParser(description='Load embeddings and plot PCs')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--emb-folder', default='hug_v3', help='stimuli folder')
parser.add_argument('-i', '--emb-file', default='modifiers_color_clip_vit_base_patch32.npy', help='input file')
parser.add_argument('-s', '--sent-file', default='modifiers_color.txt', help='input file')
parser.add_argument('-m', '--method', default='PCA', help='Dimensionality reduction method, PCA or MDS')
parser.add_argument('-d', '--dims', default=2, type=int, help='Number of dimensions to keep and plot')
# parser.add_argument('-l', '--layer', default=12, help='layer number to consider')
# parser.add_argument('--cat-or-last', default='last', help='whether to do PCA on all sequence element or just the last [CLS] token')
args = parser.parse_args()


## Load sentences
with open(f"{args.root_path}/stimuli/text/{args.sent_file}", "r") as f:
    sentences = f.readlines()
sentences = [s.rstrip().lower() for s in sentences]
# df = fns2pandas(all_img_fns)

## Load data
X = np.load(f"{args.root_path}/embeddings/{args.emb_folder}/{args.emb_file}")
n_trials, n_units = X.shape

## Get output filename
out_fn = f"{args.root_path}/results/dim_reduc/{args.emb_folder}/{args.method}_{args.emb_file}"
print(out_fn)
os.makedirs(op.dirname(out_fn), exist_ok=True)

## Actual PCA
method = PCA(args.dims) if args.method=="PCA" else MDS(args.dims, dissimilarity='precomputed')
if args.method == "MDS":
    X = euclidean_distances(X.astype(np.float64))
method.fit(X)
reduc_X = method.transform(X)
# n_trials * args.dims


## Plots
word2marker = {"circle": 'o', "triangle": '^', "square": 's'}
word2color = {"red":'r',"green":'g',"blue":'b'}

coloration = "color" in args.emb_file
sization = "size" in args.emb_file

fig = plt.subplots()
if coloration: # colored objects
    # displacement for plotting 2 markers side-by-side
    disp = np.sum((np.max(reduc_X[:,0]), np.min(reduc_X[:,0]))) / 25
    for i, sent in enumerate(sentences):
        objects = [sent.split()[2], sent.split()[-1]]
        colors = [sent.split()[1], sent.split()[-2]]
        plt.plot(reduc_X[i,0]-disp, reduc_X[i,1], marker=word2marker[objects[0]], color=word2color[colors[0]], markersize=2)
        plt.plot(reduc_X[i,0]+disp, reduc_X[i,1], marker=word2marker[objects[1]], color=word2color[colors[1]], markersize=2)

else: # no color, just objects
    # displacement for plotting 2 markers side-by-side
    disp = np.sum((np.max(reduc_X[:,0]), np.min(reduc_X[:,0]))) / 10
    for i, sent in enumerate(sentences):
        objects = [sent.split()[1], sent.split()[-1]]
        plt.plot(reduc_X[i,0]-disp, reduc_X[i,1], marker=word2marker[objects[0]], color="k")
        plt.plot(reduc_X[i,0]+disp, reduc_X[i,1], marker=word2marker[objects[1]], color="k")
plt.savefig(f"{out_fn}.png", dpi=400)
print(f"Saved figure at {out_fn}.png")

