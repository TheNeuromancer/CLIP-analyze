import torch
import clip
from PIL import Image
import argparse
from glob import glob
import os.path as op
import numpy as np
from ipdb import set_trace
import pickle
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise, img_as_ubyte
from skimage.filters import gaussian
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt 
plt.ion()

from utils import *

parser = argparse.ArgumentParser(description='Decoding from CLIP embeddings')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--emb-folder', default='embeddings/', help='input folder')
parser.add_argument('-v', '--version', default='hug_v2/', help='script version')
parser.add_argument('-i', '--in-file', default='all_txt_embs', help='input file')
parser.add_argument('-q', '--query', default='all', help='pandas query to decode, eg left_shape=="square" and right_color=="blue" of "all"')
parser.add_argument('-u', '--use-all', action='store_true', help='use augmented data')
parser.add_argument('-n', '--ncolors', default=10, type=int, help='total number of colors')
parser.add_argument('-l', '--layer', default=12, help='layer number to consider')
parser.add_argument('--cat-or-last', default='last', help='whether to use all sequence element or just the last [CLS] token')
args = parser.parse_args()

all_img_fns = glob(f"{args.root_path}/{args.emb_folder}/original_images/scene/*")
all_img_fns = [f"{op.basename(fn)}" for fn in all_img_fns]
all_img_fns = augment_text_with_colors(all_img_fns, args.ncolors)
all_captions = [fn[2:-4].lower() for fn in all_img_fns] # remove first 2 char (= 'a ')
print(np.random.choice(all_captions))

aug_str = "_augmented" if args.use_all else ""
token_str = "_all_tokens" if args.cat_or_last=='cat' else "_last_token"
out_fn = f"{args.root_path}/results/decoding/AUC_{args.in_file}{aug_str}{token_str}"
print(out_fn)
if not op.exists(op.dirname(out_fn)):
    os.makedirs(op.dirname(out_fn))

all_colors = ["red", "green", "blue", "yellow", "brown", "purple", "orange", "pink", "cyan", "gray"]

X = load_embeddings(args)
n_trials, seq_length, n_units = X.shape

if args.cat_or_last == "cat":
    X = X.reshape((n_trials, -1)) ## concatenate all the sequence elements
elif args.cat_or_last == "last":
    X = X[:, -1, :] ## keep only the last sequence element

# clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=10000, verbose=False)
clf = LinearRegression()
pipeline = make_pipeline(RobustScaler(), clf)

df = fns2pandas(all_img_fns)

if args.query == "all":
    queries, query_groups = get_queries(all_colors)
    print(query_groups)
else:
    queries = [args.query]

results = {}
grouped_results = {}
for query, group in zip(queries, query_groups):
    y = np.zeros(n_trials)
    y[df.query(query).index.to_numpy()] = 1
    if args.use_all: y = np.tile(y, n_reps)
    # if y.sum() < n_folds:
    #     print(f"found only {y.sum()} positive trials for query: {query}, moving on")
    #     continue
    n_folds = np.min([30, int(y.sum())]) # max 20 folds, but use less if we do not have enough positive trials
    if not n_folds: set_trace()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=False)

    AUC, acc = decode(X, y, pipeline, cv)
    print(f"Average AUC for query {query} is: {AUC:.2f}")
    # print(f"Average accuracy for query {query} is: {acc:.2f}")

    results[query] = AUC
    if group not in grouped_results.keys(): 
        grouped_results[group] = [AUC]
    else:
        grouped_results[group].append(AUC)

pickle.dump(results, open(f"{out_fn}.p", "wb"))
pickle.dump(grouped_results, open(f"{out_fn}_grouped.p", "wb"))

## PLOTTING
all_bars = {}
all_std = {}
queries = grouped_results.keys()
queries = queries[0:9] + queries[11:13] + [queries[-1]] + [queries[-2]]
for query in queries:
    all_bars[query] = np.mean(grouped_results[query])
    all_std[query] = np.std(grouped_results[query])

fig, ax = plt.subplots(dpi=200)
colors = ['r'] * 4 + ['b'] * 4 + ['g'] * 4 + ['y']
plt.bar(range(len(all_bars)), all_bars.values(), yerr=all_std.values(), color=colors)
ax.set_xticks(range(len(all_bars)))
ax.set_xticklabels(grouped_results.keys(), rotation=40, ha='right')
plt.hlines(0.5, -.5, len(all_bars)-0.5, lw=1, color='black', zorder =-1, linestyle="--")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig(f"{out_fn}_results.png")


set_trace()


# y = np.array([decode in cap for cap in all_captions])
# all_AUC, all_acc = [], []
# for group_decode in [["square", "triangle", "circle"], ["blue", "red", "green"],
#                      ["blue square", "blue triangle", "blue circle", "red square", "red triangle", "red circle", "red square", "red triangle", "red circle"]]:
#     print(f"decoding: {group_decode}")
#     for decode in group_decode:
#         y = np.array([decode in cap for cap in all_captions])
#         # print(y)

#         AUC, acc = 0, 0
#         for train, test in cv.split(X, y):
#             pipeline.fit(X[train, :], y[train])
#             pred = pipeline.predict(X[test, :])

#             AUC += roc_auc_score(y_true=y[test], y_score=pred) / n_folds
#             acc += accuracy_score(y[test], pred>0.5) / n_folds

#         all_AUC.append(AUC)
#         all_acc.append(acc)

#     AUC = np.mean(all_AUC)
#     acc = np.mean(all_acc)
#     print(f"Average AUC is: {AUC:.2f}")
#     print(f"Average accuracy is: {acc:.2f}")

