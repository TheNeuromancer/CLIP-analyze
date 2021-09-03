import torch
import clip
from PIL import Image
import argparse
from glob import glob
import os.path as op
import numpy as np
from ipdb import set_trace
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise, img_as_ubyte
from skimage.filters import gaussian
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt 
plt.ion()


parser = argparse.ArgumentParser(description='Ranking images and desriptions with CLIP')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/clip/stims/', help='root path')
parser.add_argument('-f', '--folder', default='scene', help='stimuli folder')
parser.add_argument('--remove-sides', action='store_true', help='remove left and right, just and "X and Y"')
parser.add_argument('-d', '--decode', default='square', help='parameter to decode')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(clip.available_models()) ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']
model, preprocess = clip.load('RN50x16', device=device)

all_img_fns = glob(f"{args.root_path}/{args.folder}/*")
# all_captions = [op.basename(fn[0:-4]) for fn in all_img_fns] 
all_captions = [f"{op.basename(fn[0:-4])}" for fn in all_img_fns] #; the picture of a {op.basename(fn)[2:-4]}" for fn in all_img_fns]
# print(all_img_fns)
# print(all_captions)

if args.remove_sides:
    all_captions = [cap.replace("to the right of", "and").replace("to the left of", "and") for cap in all_captions]

perf = []
with torch.no_grad():
    texts = clip.tokenize(all_captions).to(device)
    # imgs = [io.imread(fn) for fn in all_img_fns]
    imgs = torch.stack([preprocess(Image.open(fn)).to(device) for fn in all_img_fns]) # .unsqueeze(0)
    image_features = model.encode_image(imgs)
    text_features = model.encode_text(texts)

# X = text_features.cpu().numpy()
X = image_features.cpu().numpy()
y = np.array([args.decode in cap for cap in all_captions])
print(y)
# if args.decode = "shape"

n_folds = 10
clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=10000, verbose=False)
cv = StratifiedKFold(n_splits=n_folds, shuffle=False)
pipeline = make_pipeline(RobustScaler(), clf)

AUC, acc = 0, 0
for train, test in cv.split(X, y):
    pipeline.fit(X[train, :], y[train])
    pred = pipeline.predict(X[test, :])

    AUC += roc_auc_score(y_true=y[test], y_score=pred) / n_folds
    acc += accuracy_score(y[test], pred>0.5) / n_folds



print(f"Average AUC is: {AUC}")
print(f"Average accuracy is: {acc}")
set_trace()

