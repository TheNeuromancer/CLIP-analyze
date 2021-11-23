from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import PIL.ImageOps    
import argparse
from glob import glob
import os.path as op
import os
import time
import numpy as np
from ipdb import set_trace
import skimage.io as io
from skimage.color import rgba2rgb
from skimage.util import random_noise, img_as_ubyte
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt 
from tqdm import tqdm
plt.ion()

from utils import *

parser = argparse.ArgumentParser(description='Ranking images and desriptions with CLIP')
# parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-r', '--root-path', default='/Users/tdesbordes/Documents/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--folder', default='scene', help='stimuli folder')
parser.add_argument('--remove-sides', action='store_true', help='remove left and right, just and "X and Y"')
parser.add_argument('--embs-out-fn', default='hug_v1', help='output directory for embeddings')
parser.add_argument('-w', '--overwrite', action='store_true', help='whether to overwrite the output directory or not')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


all_img_fns = glob(f"{args.root_path}/stimuli/original_images/{args.folder}/*")
all_img_fns = [f"{op.basename(fn)}" for fn in all_img_fns]
# all_img_fns = augment_text_with_colors(all_img_fns)
all_captions = [fn[2:-4].lower() for fn in all_img_fns] # remove first 2 char (= 'a ')

# img_path = f"{args.root_path}/stimuli/images"
# all_folders = glob(f"{op.dirname(args.root_path)}/stimuli/images/*")
# all_folders = [f"{op.basename(fn)}" for fn in all_folders]
# print(f"Found {len(all_folders)} folders of images")

img_path = f"{args.root_path}/stimuli/original_images/scene"
all_folders = [[img_path]]
print(f"Using original images")
# print(all_img_fns)
# print(all_captions)
separator = " to the X of a "

if args.remove_sides:
    all_captions = [cap.replace("to the right of", "next to").replace("to the left of", "next to") for cap in all_captions]
    separator = " next to a "

# get all correct labels for each image: the "classic" one and the one resulting from inversing the relation (left/right)
all_labels = []
for i, cap in enumerate(all_captions):
    all_labels.append([i])
    mirror_sent = get_inverse_sentence(cap, separator)
    for idx in [i for i, x in enumerate(all_captions) if x == mirror_sent]:
        all_labels[-1].append(idx)
all_labels = torch.tensor(all_labels)
print(all_labels.shape)
# np.random.shuffle(all_captions)
print(all_captions[12])
print(all_captions[104])

print(f"Chance is {1/len(np.unique(all_captions)):.3f}")

templates = ["a bad photo of a {}.",
             "an origami {}.",
             "a photo of a large {}.",
             "a {} in a video game.",
             "art of a {}.",
             "a drawing of a {}.",
             "a photo of a small {}."]

#### LOOPS OVER DATA AUGMENTATION (one folder contains the whole dataset for one augmentation)
all_logits = []
with torch.no_grad():
    for folder in tqdm(all_folders[0]):
        start = time.time()
        images = [Image.open(f"{folder}/{fn}") for fn in all_img_fns]
        images = [im.convert('RGB') for im in images]
        inputs = processor(text=all_captions, images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs, output_hidden_states=False, return_dict=True)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        all_logits.append(logits_per_image.cpu())
        # print(f"took {time.time() - start:.2f}s on device {device}")

final_logits = torch.stack(all_logits).mean(0).softmax(dim=1) # we can take the softmax to get the label probabilities
set_trace()
topk = (1,2,3,5,10)
accs = accuracy(final_logits, all_labels, topk=topk)

for i, k in enumerate(topk):
    top = (accs[i] / len(all_captions)) * 100
    print(f"Top-{k} accuracy: {top:.2f}")

