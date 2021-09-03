import torch
import clip
from PIL import Image
import PIL.ImageOps    
import argparse
from glob import glob
import os.path as op
import numpy as np
from ipdb import set_trace
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise, img_as_ubyte
from skimage.filters import gaussian
import matplotlib.pyplot as plt 
plt.ion()


parser = argparse.ArgumentParser(description='Ranking images and desriptions with CLIP')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/clip/stims/', help='root path')
parser.add_argument('-f', '--folder', default='scene', help='stimuli folder')
parser.add_argument('--remove-sides', action='store_true', help='remove left and right, just and "X and Y"')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(clip.available_models()) ['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'RN50x16']
model, preprocess = clip.load('RN50x16', device=device)

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", model.visual.input_resolution)
print("Context length:", model.context_length)

all_img_fns = glob(f"{args.root_path}/{args.folder}/*")
# all_captions = [op.basename(fn[0:-4]) for fn in all_img_fns] 
all_captions = [f"{op.basename(fn[0:-4])}"[2::].lower() for fn in all_img_fns] # remove first 2 char (= 'a ')
# all_captions = [cap.replace("a ", "").replace("to the ", "") for cap in all_captions]
# all_captions = [cap.replace("to the ", "") for cap in all_captions]
# all_captions = [cap.replace("to the right of", "on the right of").replace("to the left of", "on the left of") for cap in all_captions]
# print(all_img_fns)
# print(all_captions)
separator = " to the X of a "

if args.remove_sides:
    all_captions = [cap.replace("to the right of", "next to").replace("to the left of", "next to") for cap in all_captions]
    separator = " next to a "

def get_inverse_sentence(sentence, sep):
    # gets the mirror image sentence
    if "X" in sep: sep = sep.replace("X", "right") if "right" in sentence else sep.replace("X", "left")
    first, second = sentence.split(sep)
    mirror_sent = sep.join([second, first])
    if "right" in sentence:
        mirror_sent = mirror_sent.replace("right", "left")
    elif "left" in sentence:
        mirror_sent = mirror_sent.replace("left", "right")
    return mirror_sent

# get all correct labels for each image: the "classic" one and the one resulting from inversing the relation (left/right)
all_labels = []
for i, cap in enumerate(all_captions):
    all_labels.append([i])
    mirror_sent = get_inverse_sentence(cap, separator)
    for idx in [i for i, x in enumerate(all_captions) if x == mirror_sent]:
        all_labels[-1].append(idx)
all_labels = torch.tensor(all_labels)
# set_trace()
# set_trace()
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

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        # for classname in tqdm(classnames):
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            if not zeroshot_weights: print(texts)
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

zeroshot_weights = zeroshot_classifier(all_captions, templates)
# print(zeroshot_weights.shape)

def accuracy(output, all_targets, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    all_corrects = []
    for i in range(all_targets.shape[1]): # nb of possible label for each image
        correct = pred.eq(all_targets[:,i].view(1, -1).expand_as(pred))
        all_corrects.append(correct)
    correct = torch.logical_or(*all_corrects)
    # remove duplicate for top>1
    correct = correct.cumsum(axis=0).cumsum(axis=0) == 1 
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

images = []
for i, (img_fn, caption) in enumerate(zip(all_img_fns, all_captions)):
    image = Image.fromarray(io.imread(img_fn))
    image = preprocess(image).cuda()
    images.append(image)

images = torch.stack(images)
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
        
    # predict
with torch.no_grad():
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    logits = 100. * image_features @ zeroshot_weights

    # measure accuracy
    topk = (1,2,3,5,10)
    accs = accuracy(logits, all_labels.cuda(), topk=topk)

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

