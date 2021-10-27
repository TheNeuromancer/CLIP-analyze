import torch
import clip
from PIL import Image
import argparse
from glob import glob
import os.path as op
import os
import numpy as np
from ipdb import set_trace
import pickle
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise, img_as_ubyte
from skimage.filters import gaussian
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt 
plt.ion()

from utils import *

parser = argparse.ArgumentParser(description='Encoding on CLIP embeddings')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--folder', default='embeddings/', help='stimuli folder')
parser.add_argument('-v', '--version', default='hug_v2/', help='script version')
parser.add_argument('-i', '--in-file', default='all_txt_embs', help='input file: txt or img')
parser.add_argument('-n', '--ncolors', default=3, type=int, help='total number of colors')
parser.add_argument('-l', '--layer', default=12, help='layer number to consider')
parser.add_argument('-u', '--use-all', action='store_true', help='use augmented data')
parser.add_argument('--cat-or-last', default='last', help='whether to use all sequence element or just the last [CLS] token')
args = parser.parse_args()


all_img_fns = glob(f"{args.root_path}/original_images/scene/*")
all_sents = [f"{op.basename(fn)}" for fn in all_img_fns]
if "img" in args.in_file and args.ncolors > 3:
    print("Cannot add new colors to images ... exiting")
    exit()
all_sents = augment_text_with_colors(all_sents, args.ncolors)
# print(all_sents)

aug_str = "_augmented" if args.use_all else ""
out_fn = f"{args.root_path}/results/encoding/{args.version}/{args.in_file}{aug_str}_{args.ncolors}colors"
print(out_fn)
if not op.exists(op.dirname(out_fn)):
    os.makedirs(op.dirname(out_fn))
make_legend_encoding(out_fn)

y = load_embeddings(args)
n_trials, seq_length, n_units = y.shape

if args.cat_or_last == "cat":
    y = y.reshape((-1, n_units)) ## concatenate all the sequence elements
elif args.cat_or_last == "last":
    y = y[:, -1, :] ## keep only the last sequence element


encoder = RidgeCV()
pipeline = make_pipeline(RobustScaler(), encoder)
weight_scaler = StandardScaler()

n_folds = 10
cv = KFold(n_splits=n_folds, shuffle=True)


all_Rs, all_R2s, all_weights = {}, {}, {}
for feats in [["properties"], ["properties", "objects"], ["objects"], \
              ["properties", "objects", "sentences"], \
              ["sentences"], ["sentences", "objects"], \
              ["sentOrder"], ["sentOrder", "sentences"]]:
    X, feat_names = get_encoding_features(fns2pandas(all_sents), features=feats, ncolors=args.ncolors)
    if args.use_all: X = np.tile(X, (n_reps, 1))
    feat_label = "_".join(feats)
    print(f"Doing features {feat_label}, got X of shape {X.shape}")
    print(X.shape, y.shape)
    Rs, R2s, weights, r2 = encode(X, y, pipeline, cv)
    all_Rs[feat_label] = Rs
    all_R2s[feat_label] = R2s
    all_weights[feat_label] = weights
    print(f"mean r2: {r2:.2f}")

    weights = weight_scaler.fit_transform(weights.T).T
    indices_of_interest = weights.argmax(0)
    if len(feat_names) > 9:  # do multiple plots
        for i in range(len(feat_names) // 9):
            # print(indices_of_interest[i*9:(i+1)*9], feat_names[i*9:(i+1)*9])
            plot_hist_weights(weights[:,i*9:(i+1)*9], indices_of_interest[i*9:(i+1)*9], feat_names[i*9:(i+1)*9], f"{out_fn}_{feat_label}_units_{i}.png")
    else: # classic   
        plot_hist_weights(weights, indices_of_interest, feat_names, f"{out_fn}_{feat_label}_units.png")


exit()

pickle.dump(all_Rs, open(f"{out_fn}_all_Rs.p", "wb"))
pickle.dump(all_R2s, open(f"{out_fn}_all_R2s.p", "wb"))
pickle.dump(all_weights, open(f"{out_fn}_all_weights.p", "wb"))

# order vs side
plot_encoding_hierachy(all_Rs["sentences"], all_Rs["sentOrder_sentences"], all_Rs["sentOrder_sentences"], f"{out_fn}_sideVSorder", sort_with='R1', colors=['b','y', 'k'])
plot_encoding_hierachy(all_Rs["sentences"], all_Rs["sentOrder_sentences"], all_Rs["sentOrder_sentences"], f"{out_fn}sideVSorder", sort_with='R2', colors=['b','y', 'k'])
plot_encoding_hierachy(all_Rs["sentOrder"], all_Rs["sentOrder_sentences"], all_Rs["sentOrder_sentences"], f"{out_fn}_orderVSside", sort_with='R1', colors=['y','b', 'k'])
plot_encoding_hierachy(all_Rs["sentOrder"], all_Rs["sentOrder_sentences"], all_Rs["sentOrder_sentences"], f"{out_fn}orderVSside", sort_with='R2', colors=['y','b', 'k'])

# classical
plot_encoding_hierachy(all_Rs["properties"], all_Rs["properties_objects"], all_Rs["properties_objects_sentences"], out_fn, sort_with='R1')
plot_encoding_hierachy(all_Rs["properties"], all_Rs["properties_objects"], all_Rs["properties_objects_sentences"], out_fn, sort_with='R2')
plot_encoding_hierachy(all_Rs["properties"], all_Rs["properties_objects"], all_Rs["properties_objects_sentences"], out_fn, sort_with='R3')
plot_encoding_hierachy(all_Rs["properties"], all_Rs["properties_objects"], all_Rs["properties_objects_sentences"], out_fn, sort_with='R2-R1')
plot_encoding_hierachy(all_Rs["properties"], all_Rs["properties_objects"], all_Rs["properties_objects_sentences"], out_fn, sort_with='R3-R2')

# sentence first
plot_encoding_hierachy(all_Rs["sentences"], all_Rs["sentences_objects"], all_Rs["properties_objects_sentences"], f"{out_fn}_sentfirst", sort_with='R1', colors=['b','g', 'r'])
plot_encoding_hierachy(all_Rs["sentences"], all_Rs["sentences_objects"], all_Rs["properties_objects_sentences"], f"{out_fn}_sentfirst", sort_with='R2', colors=['b','g', 'r'])
plot_encoding_hierachy(all_Rs["sentences"], all_Rs["sentences_objects"], all_Rs["properties_objects_sentences"], f"{out_fn}_sentfirst", sort_with='R3', colors=['b','g', 'r'])
plot_encoding_hierachy(all_Rs["sentences"], all_Rs["sentences_objects"], all_Rs["properties_objects_sentences"], f"{out_fn}_sentfirst", sort_with='R2-R1', colors=['b','g', 'r'])
plot_encoding_hierachy(all_Rs["sentences"], all_Rs["sentences_objects"], all_Rs["properties_objects_sentences"], f"{out_fn}_sentfirst", sort_with='R3-R2', colors=['b','g', 'r'])


# df = fns2pandas(all_img_fns)
# df = df.drop('sentence', axis=1)
# X = pd.get_dummies(df).to_numpy()
# print(X.shape, y.shape)
# if args.use_all: X = np.tile(X, (n_reps, 1))
# all_Rs, all_R2s, all_weights, r2 = encode(X, y, pipeline, cv)
# print(f"Average r2 is: {r2:.2f}")
# # print(f"All Rs: {all_Rs}")

# print(f"Adding object features")
# df = add_objects_to_df(df)
# X = pd.get_dummies(df).to_numpy()
# if args.use_all: X = np.tile(X, (n_reps, 1))
# all_Rs_, all_R2s_, all_weights_, r2_ = encode(X, y, pipeline, cv)
# print(f"Average r2 is: {r2_:.2f}")
# # print(f"All Rs: {all_Rs_}")
# # print(f"Difference in Rs: {all_Rs_ - all_Rs}")
# print(f"Biggest difference in Rs: {np.max(all_Rs_ - all_Rs):.2f}")

# print(f"Adding sentence features")
# df = fns2pandas(all_img_fns)
# df = add_objects_to_df(df)
# X = pd.get_dummies(df).to_numpy()
# if args.use_all: X = np.tile(X, (n_reps, 1))
# all_Rs__, all_R2s__, all_weights__, r2__ = encode(X, y, pipeline, cv)
# print(f"Average r2 is: {r2__:.2f}")
# # print(f"All Rs: {all_Rs__}")
# # print(f"Difference in Rs: {all_Rs__ - all_Rs_}")
# print(f"Biggest difference in Rs: {np.max(all_Rs__ - all_Rs_):.2f}")

# plot_encoding_hierachy(all_Rs, all_R2s_, all_Rs__, out_fn, sort_with='R1')
# plot_encoding_hierachy(all_Rs, all_R2s_, all_Rs__, out_fn, sort_with='R2')
# plot_encoding_hierachy(all_Rs, all_R2s_, all_Rs__, out_fn, sort_with='R3')
# plot_encoding_hierachy(all_Rs, all_R2s_, all_Rs__, out_fn, sort_with='R2-R1')
# plot_encoding_hierachy(all_Rs, all_R2s_, all_Rs__, out_fn, sort_with='R3-R2')

#   set_trace()

# ## PLOTTING
# all_bars = {}
# all_std = {}
# queries = grouped_results.keys()
# queries = queries[0:9] + queries[11:13] + [queries[-1]] + [queries[-2]]
# for query in queries:
#     all_bars[query] = np.mean(grouped_results[query])
#     all_std[query] = np.std(grouped_results[query])

# fig, ax = plt.subplots(dpi=200)
# colors = ['r'] * 4 + ['b'] * 4 + ['g'] * 4 + ['y']
# plt.bar(range(len(all_bars)), all_bars.values(), yerr=all_std.values(), color=colors)
# ax.set_xticks(range(len(all_bars)))
# ax.set_xticklabels(grouped_results.keys(), rotation=40, ha='right')
# plt.hlines(0.5, -.5, len(all_bars)-0.5, lw=1, color='black', zorder =-1, linestyle="--")
# plt.ylabel("AUC")
# plt.tight_layout()
# plt.savefig(f"{out_fn}_results.png")



