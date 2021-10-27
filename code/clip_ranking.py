import torch
import clip
from PIL import Image
import PIL.ImageOps    
import argparse
from glob import glob
import os
import os.path as op
import numpy as np
from ipdb import set_trace
import skimage.io as io
from skimage.color import rgba2rgb
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise, img_as_ubyte
from skimage.filters import gaussian
import matplotlib.pyplot as plt 
import time
from tqdm import tqdm
plt.ion()

from utils import *


parser = argparse.ArgumentParser(description='Ranking images and desriptions with CLIP')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--folder', default='scene', help='stimuli folder')
parser.add_argument('--remove-sides', action='store_true', help='remove left and right, just and "X and Y"')
parser.add_argument('-s', '--save-embs', action='store_true', help='remove left and right, just and "X and Y"')
parser.add_argument('--embs-out-fn', default='v2', help='output directory for embeddings')
parser.add_argument('-w', '--overwrite', action='store_true', help='whether to overwrite the output directory or not')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(clip.available_models()) ['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'RN50x16']
model, preprocess = clip.load('RN50x16', device=device)

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", model.visual.input_resolution)
print("Context length:", model.context_length)

if args.save_embs:
    out_dir = f"{args.root_path}/embeddings/{args.embs_out_fn}"
    if op.exists(out_dir):
        if args.overwrite:
            print(f"Output directory already exists ... overwriting")
        else:
            print(f"Output directory already exists and args.overwrite is set to False ... exiting")
            exit()
    else:
        os.makedirs(out_dir)

all_img_fns = glob(f"{args.root_path}/original_images/{args.folder}/*")
all_img_fns = [f"{op.basename(fn)}" for fn in all_img_fns]
# all_img_fns = augment_text_with_colors(all_img_fns)
all_captions = [fn[2:-4].lower() for fn in all_img_fns] # remove first 2 char (= 'a ')
all_folders = glob(f"{op.dirname(args.root_path)}/images/*")
all_folders = [f"{op.basename(fn)}" for fn in all_folders]
print(f"Found {len(all_folders)} folders of images")
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
# np.random.shuffle(all_captions)

# all_captions = [cap.replace("to the right of", "on the right-hand side.").replace("to the left of", "on the left-hand side.") for cap in all_captions]
# all_captions = [f"{cap} on the {'right' if 'left' in cap else 'left'}" for cap in all_captions]
# all_captions = [cap.replace("to the right of", "on the right-hand side.").replace("to the left of", "on the left-hand side.") for cap in all_captions]
# all_captions = [f"{cap} on the {'right' if 'left' in cap else 'left'}" for cap in all_captions]
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

zeroshot_weights = zeroshot_classifier(all_captions, templates, model=model, device=device)
# print(zeroshot_weights.shape)

all_image_features = []
for folder in tqdm(all_folders):
    start = time.time()
    images = []
    for i, img_fn in enumerate(all_img_fns):
        # print(f"{args.root_path}/images/{folder}/{img_fn}")
        image = Image.open(f"{args.root_path}/images/{folder}/{img_fn}")
        image = preprocess(image).to(device)
        images.append(image)

    images = torch.stack(images)
            
    # predict
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    all_image_features.append(image_features.cpu())

    # print(f"took {time.time() - start:.2f}s on device {device}")

if args.save_embs:
    embs_to_save = np.array([emb.numpy() for emb in all_image_features])
    np.save(f"{out_dir}/embs.npy", embs_to_save)

image_features = torch.mean(torch.stack(all_image_features), 0)
logits = 100. * image_features @ zeroshot_weights

# measure accuracy
topk = (1,2,3,5,10)
accs = accuracy(logits.to(device), all_labels.to(device), topk=topk)

        # all_probs = []
        # for noise in np.random.randn(20): #add random noise to the image
        #     noise = np.abs(noise) #np.clip(noise, -.5, .5)
        #     transf_image = img_as_ubyte(random_noise(image,var=noise/10**2))
        #     final_image = preprocess(Image.fromarray(transf_image)).unsqueeze(0).to(device)
        #     img_feat = model.encode_image(final_image)
        #     img_feat /= img_feat.norm(dim=-1, keepdim=True)
        #     logits = 100. * img_feat @ zeroshot_weights
        #     # logits_per_image, logits_per_text = model(final_image, text)
        #     # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #     all_probs.append(logits)
        # for angle in np.random.randint(-25, 25, size=10):
        #     transf_image = img_as_ubyte(rotate(image, angle=angle, mode='wrap'))
        #     final_image = preprocess(Image.fromarray(transf_image)).unsqueeze(0).to(device)
        #     img_feat = model.encode_image(final_image)
        #     img_feat /= img_feat.norm(dim=-1, keepdim=True)
        #     logits = 100. * img_feat @ zeroshot_weights
        #     # logits_per_image, logits_per_text = model(final_image, text)
        #     # probs = logits_per_image.softmax(dim=-1).numpy()
        #     all_probs.append(logits)
        # for transl in np.random.randint(-45, 45, size=20):
        #     transform = AffineTransform(translation=(transl,transl))
        #     wrapShift = img_as_ubyte(warp(image,transform,mode='wrap'))
        #     final_image = preprocess(Image.fromarray(transf_image)).unsqueeze(0).to(device)
        #     img_feat = model.encode_image(final_image)
        #     img_feat /= img_feat.norm(dim=-1, keepdim=True)
        #     logits = 100. * img_feat @ zeroshot_weights
        #     # logits_per_image, logits_per_text = model(final_image, text)
        #     # probs = logits_per_image.softmax(dim=-1).numpy()
        #     all_probs.append(logits)
        # logits = torch.mean(torch.stack(all_probs), 0)

for i, k in enumerate(topk):
    top = (accs[i] / len(all_captions)) * 100
    print(f"Top-{k} accuracy: {top:.2f}")




        # image = io.imread(img_fn) # Image.fromarray(
        # crop1 = image.shape[0] // 2 - 384 // 2
        # crop2 = image.shape[1] // 2 - 384 // 2
        # image = Image.fromarray(image[crop1:crop1+384, crop2:crop2+384])
        # image = Image.open(img_fn)
        # set_trace()
        # r,g,b,a = image.split()
        # image2 = Image.merge('RGB', (r,g,b))
        # set_trace()
        # image = PIL.ImageOps.invert(rgb_image)
        # set_trace()



        # templates = [
#     'a bad photo of a {}.',
#     'a sculpture of a {}.',
#     'a photo of a hard to see {}.',
#     'a low resolution photo of a {}.',
#     'a rendering of a {}.',
#     'graffiti of a {}.',
#     'a bad photo of a {}.',
#     'a cropped photo of a {}.',
#     'a tattoo of a {}.',
#     'a embroidered {}.',
#     'a photo of a hard to see {}.',
#     'a bright photo of a {}.',
#     'a photo of a clean {}.',
#     'a photo of a dirty {}.',
#     'a drawing of a {}.',
#     'a photo of a cool {}.',
#     'a close-up photo of a {}.',
#     'a painting of a {}.',
#     'a painting of a {}.',
#     'a pixelated photo of a {}.',
#     'a sculpture of a {}.',
#     'a bright photo of a {}.',
#     'a cropped photo of a {}.',
#     'a photo of a dirty {}.',
#     'a jpeg corrupted photo of a {}.',
#     'a blurry photo of a {}.',
#     'a photo of a {}.',
#     'a good photo of a {}.',
#     'a rendering of a {}.',
#     'a {} in a video game.',
#     'a photo of one {}.',
#     'a doodle of a {}.',
#     'a close-up photo of a {}.',
#     'a photo of a {}.',
#     'a origami {}.',
#     'a sketch of a {}.',
#     'a doodle of a {}.',
#     'a origami {}.',
#     'a low resolution photo of a {}.',
#     'a toy {}.',
#     'a rendition of a {}.',
#     'a photo of a clean {}.',
#     'a photo of a large {}.',
#     'a rendition of a {}.',
#     'a photo of a nice {}.',
#     'a photo of a weird {}.',
#     'a blurry photo of a {}.',
#     'a cartoon {}.',
#     'art of a {}.',
#     'a sketch of a {}.',
#     'a embroidered {}.',
#     'a pixelated photo of a {}.',
#     'itap of a {}.',
#     'a jpeg corrupted photo of a {}.',
#     'a good photo of a {}.',
#     'a plushie {}.',
#     'a photo of a nice {}.',
#     'a photo of a small {}.',
#     'a photo of a weird {}.',
#     'a cartoon {}.',
#     'art of a {}.',
#     'a drawing of a {}.',
#     'a photo of a large {}.',
#     'a black and white photo of a {}.',
#     'a plushie {}.',
#     'a dark photo of a {}.',
#     'itap of a {}.',
#     'graffiti of a {}.',
#     'a toy {}.',
#     'itap of my {}.',
#     'a photo of a cool {}.',
#     'a photo of a small {}.',
#     'a tattoo of a {}.',
# ]   
