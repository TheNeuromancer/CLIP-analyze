import torch
import clip
from PIL import Image
import argparse
from glob import glob
import os
import os.path as op
import numpy as np
from ipdb import set_trace
import pickle
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise, img_as_ubyte
from skimage.filters import gaussian
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
plt.ion()

from utils import *

parser = argparse.ArgumentParser(description='Load text embeddings and look at trajectories')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--folder', default='embeddings/', help='stimuli folder')
parser.add_argument('-v', '--version', default='hug_v2/', help='script version')
parser.add_argument('-i', '--in-file', default='all_txt_embs', help='input file')
parser.add_argument('-m', '--method', default='PCA', help='Dimensionality reduction method, PCA or MDS')
parser.add_argument('-d', '--dims', default=16, type=int, help='Number of dimensions to keep and plot')
parser.add_argument('-n', '--ncolors', default=10, type=int, help='total number of colors')
parser.add_argument('-l', '--layer', default=12, help='layer number to consider')
parser.add_argument('-u', '--use-all', action='store_true', help='use augmented data')
parser.add_argument('--cat-or-last', default='last', help='whether to do PCA on all sequence element or just the last [CLS] token')
args = parser.parse_args()

all_img_fns = glob(f"{args.root_path}/original_images/scene/*")
all_img_fns = [f"{op.basename(fn)}" for fn in all_img_fns]
all_img_fns = augment_text_with_colors(all_img_fns, args.ncolors)
all_captions = np.array([fn[:-4].lower() for fn in all_img_fns]) # remove first 2 char (= 'a ')
print(np.random.choice(all_captions))

aug_str = "_augmented" if args.use_all else ""
token_str = "_all_tokens" if args.cat_or_last=='cat' else "_last_token"
out_fn = f"{args.root_path}/results/trajectory/{args.version}/{args.method}_traj_{args.in_file}{aug_str}{token_str}"
print(out_fn)
if not op.exists(op.dirname(out_fn)):
    os.makedirs(op.dirname(out_fn))

X = load_embeddings(args)
n_trials, seq_length, n_units = X.shape

if args.cat_or_last == "cat":
    X = X.reshape((-1, n_units)) ## concatenate all the sequence elements
    X_for_pca = X
elif args.cat_or_last == "last":
    X_for_pca = X[:, -1, :] ## keep only the last sequence element to fit the PCA
    X = X.reshape((-1, n_units))


method = PCA(args.dims) if args.method=="PCA" else MDS(args.dims, dissimilarity='precomputed')
# scaler = RobustScaler()
# X = scaler.fit_transform(X)
X_for_pca = scaler.fit_transform(X_for_pca)

df = fns2pandas(all_img_fns)

if args.method == "MDS": 
    # print(X.shape)
    X = euclidean_distances(X.astype(np.float64))
# print(X.shape)

method.fit(X_for_pca)
reduc_X = method.transform(X)
reduc_X = reduc_X.reshape((-1, seq_length, args.dims))
# n_trials * seq_legnth * n_PCs

# additional_colors = ["red", "green", "blue", "yellow", "brown", "purple", "orange", "pink", "gold", "gray"]


## all PCs plots
# Mirror images
# exs = ['red circle to the left of a green triangle', 'red circle to the right of a green triangle',
#        'green triangle to the right of a red circle', 'green triangle to the left of a red circle']
ctr = 1
for shape1, shape2 in zip(["square", "square", "square"], ["square", "circle", "triangle"]):
    for color1, color2 in zip(["red", "blue"], ["red", "green"]):
        exs = [f"a {color1} {shape1} to the left of a {color2} {shape2}", f"a {color1} {shape1} to the right of a {color2} {shape2}",
               f"a {color2} {shape2} to the right of a {color1} {shape1}", f"a {color2} {shape2} to the left of a {color1} {shape1}"]
        indices = []
        for ex in exs:
            print(ex)
            indices.append(np.where(all_captions==ex)[0][0])
        colors = ["lightgreen", "firebrick", "green", "red"]
        plot_all_pca_trajectory(trajs=reduc_X[indices], labels=exs, out_fn=f"{out_fn}_all_pcs_mirror_{ctr}.png", colors=colors)
        ctr += 1


exit()

## Mirror images
exs = ['red circle to the left of a green triangle', 'red circle to the right of a green triangle',
       'green triangle to the right of a red circle', 'green triangle to the left of a red circle']
indices = []
for ex in exs:
    indices.append(np.where(all_captions==ex)[0][0])
fig, ax = plt.subplots(dpi=200)
for idx in indices:
    plt.plot(reduc_X[idx, :, 0], reduc_X[idx, :, 1])
## Put a star at the beginning of the sequence
ax.scatter(reduc_X[indices[0], 0, 0], reduc_X[indices[0], 0, 1], marker='*', color='k', zorder=10)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
ax.legend(exs, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True, prop={'size': 9})
plt.tight_layout()
plt.savefig(f"{out_fn}_results_mirror.png")



## same / different shape color 
exs = ['red circle to the left of a blue square', 'red square to the right of a blue circle',
       'blue circle to the right of a red square', 'blue square to the left of a red circle']
indices = []
for ex in exs:
    indices.append(np.where(all_captions==ex)[0][0])
fig, ax = plt.subplots(dpi=200)
for idx in indices:
    plt.plot(reduc_X[idx, :, 0], reduc_X[idx, :, 1])
## Put a star at the beginning of the sequence
ax.scatter(reduc_X[indices[0], 0, 0], reduc_X[indices[0], 0, 1], marker='*', color='k', zorder=10)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
ax.legend(exs, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True, prop={'size': 9})
plt.tight_layout()
plt.savefig(f"{out_fn}_results_samediff.png")


## all first objects, with lil' markers
word2marker = {"circle": 'o', "triangle": '^', "square": 's', "red": '*', "green": '*', "blue": '*', \
        "yellow": '*', "brown": '*', "purple": '*', "orange": '*', "pink": '*', "cyan": '*', "gray": '*'}
word2color = {"circle": 'k', "triangle": 'k', "square": 'k', "red": 'r', "green": 'g', "blue": 'b', \
        "yellow": 'yellow', "brown": 'brown', "purple": 'purple', "orange": 'orange', "pink": 'pink', "cyan": 'cyan', "gray": 'gray'}
# dones = []
fig, ax = plt.subplots(dpi=200)
for idx in range(len(all_captions)):
    words = all_captions[idx].split()[0:3]
    plt.plot(reduc_X[idx, :, 0], reduc_X[idx, :, 1], lw=.1, alpha=.3, c='grey')
    for i_w, w in enumerate(words):
        if w in word2color.keys(): # and (reduc_X[idx, i_w, 0], reduc_X[idx, i_w, 1]) not in dones: # words of interest
                color = word2color[words[i_w]] if w in additional_colors else word2color[words[i_w-1]] # get correct color
                plt.scatter(reduc_X[idx, i_w+1, 0], reduc_X[idx, i_w+1, 1], marker=word2marker[w], color=color, zorder=10, s=12)
            # dones.append((reduc_X[idx, i_w, 0], reduc_X[idx, i_w, 1]))
## Put a star at the beginning of the sequence
ax.scatter(reduc_X[indices[0], 0, 0], reduc_X[indices[0], 0, 1], marker='X', color='k', zorder=10)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
## Make legend
handles = []
handles.append(mlines.Line2D([], [], color='k', marker='X', linestyle='None', markersize=6, label='Start'))
handles.append(mlines.Line2D([], [], color='grey', marker='*', linestyle='None', markersize=6, label='First color'))
plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True, prop={'size': 6})
plt.tight_layout()
plt.savefig(f"{out_fn}_results_all_firstobj.png")


## all second objects, with lil' markers
word2marker = {"circle": 'o', "triangle": '^', "square": 's'}
# word2color = {"red": 'r', "green": 'g', "blue": 'b'} #, "right": 'k', "left": 'k'}
word2color = {"red":'r',"green":'g',"blue":'b',"yellow":'yellow',"brown":'brown',"purple":'purple',"orange":'orange',"pink":'pink',"cyan": 'cyan', "gray": 'gray'}
dones = []
fig, ax = plt.subplots(dpi=200)
for idx in range(len(all_captions)):
    words = all_captions[idx].split()[4::]
    plt.plot(reduc_X[idx, :, 0], reduc_X[idx, :, 1], lw=.1, alpha=.3, c='grey')
    for i_w, w in enumerate(words):
        if w in word2marker.keys() and (reduc_X[idx, 4+i_w, 0], reduc_X[idx, 4+i_w, 1]) not in dones: # words of interest
            color = word2color[words[i_w-1]] # get correct color
            plt.scatter(reduc_X[idx, 4+i_w+1, 0], reduc_X[idx, 4+i_w+1, 1], marker=word2marker[w], color=color, zorder=10, s=12)
            dones.append((reduc_X[idx, 4+i_w, 0], reduc_X[idx, 4+i_w, 1]))
## Put a star at the beginning of the sequence
ax.scatter(reduc_X[indices[0], 0, 0], reduc_X[indices[0], 0, 1], marker='X', color='k', zorder=10)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.savefig(f"{out_fn}_results_all_seoncdobj.png")


## all of the same shape
shapesOfInterest = ["triangle", "circle", "square"]
# word2color = {"red": 'r', "green": 'g', "blue": 'b'} #, "right": 'k', "left": 'k'}
word2color = {"red":'r',"green":'g',"blue":'b',"yellow":'yellow',"brown":'brown',"purple":'purple',"orange":'orange',"pink":'pink',"cyan": 'cyan', "gray": 'gray'}
word2marker = {"circle": 'o', "triangle": '^', "square": 's'}
for shape in shapesOfInterest:
    fig, ax = plt.subplots(dpi=200)
    for idx in range(len(all_captions)):
        words = all_captions[idx].split()
        plt.plot(reduc_X[idx, :, 0], reduc_X[idx, :, 1], lw=.1, alpha=.3, c='grey')
        for i_w, w in enumerate(words):
            if w == shape:
                color = word2color[words[i_w-1]]
                marker = word2marker[w]
                edgecolor = 'none' if i_w > 3 else 'k' # black border for the second object
                if shape == "triangle" and i_w > 3: # side triangle reflecting the position on the screen
                    if len([w for w in words if w=="triangle"]) == 2:
                        marker = "d"
                    else:
                        marker = ">" if 'left' in words else "<"
                if idx == 160: print(idx, w, color, edgecolor)
                plt.scatter(reduc_X[idx, i_w+1, 0], reduc_X[idx, i_w+1, 1], marker=marker, color=color, zorder=10, s=12, edgecolor=edgecolor, lw=1)
    ## Put a colored star for each color (first word)
    for c in word2color.keys():
        for idx in range(len(all_captions)):
            words = all_captions[idx].split()
            if words[0] == c:
                ax.scatter(reduc_X[idx, 1, 0], reduc_X[idx, 1, 1], marker='*', color=c, zorder=10)
    ## Put a cross at the beginning of the sequence
    ax.scatter(reduc_X[indices[0], 0, 0], reduc_X[indices[0], 0, 1], marker='X', color='k', zorder=10)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    ## Make legend
    handles = []
    handles.append(mlines.Line2D([], [], color='k', marker='X', linestyle='None', markersize=6, label='Start'))
    handles.append(mlines.Line2D([], [], color='grey', marker='*', linestyle='None', markersize=6, label='First color'))
    if shape == "triangle":
        handles.append(mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=6, label='Both shapes are triangles'))
        handles.append(mlines.Line2D([], [], color='grey', marker='<', linestyle='None', markersize=6, label='Color of the 2nd shape (shape is on the left)'))
        handles.append(mlines.Line2D([], [], color='grey', marker='>', linestyle='None', markersize=6, label='Color of the 2nd shape (shape is on the right)'))
    else:
        handles.append(mlines.Line2D([], [], color='grey', marker=word2marker[shape], markeredgecolor='k', markeredgewidth=1, linestyle='None', markersize=6, label='Color of the 1st shape'))
        handles.append(mlines.Line2D([], [], color='grey', marker=word2marker[shape], linestyle='None', markersize=6, label='Color of the 2nd shape'))

    plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True, prop={'size': 6})

    # ax.legend(exs, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True, prop={'size': 9})
    plt.tight_layout()
    plt.savefig(f"{out_fn}_results_all_{shape}s.png")



        # if w in word2marker.keys(): marker = word2marker[w]
        # if w in word2color.keys(): color = word2color[w]
        # if marker is not None and w in (word2marker.keys() or word2color.keys()) and (reduc_X[idx, i_w, 0], reduc_X[idx, i_w, 1]) not in dones: 
            # words of interest are: triangle, and colors when they immediately follow a tria