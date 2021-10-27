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
plt.ion()


parser = argparse.ArgumentParser(description='Ranking images and desriptions with CLIP')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/clip/stims/', help='root path')
parser.add_argument('-f', '--folder', default='scene', help='stimuli folder')
args = parser.parse_args()

all_img_fns = glob(f"{args.root_path}/{args.folder}/*")

np.random.seed(42)

ctr = 0
for noise in [0] + [1] * 10:
    for angle in [0] + [1] * 10: #, *np.random.randint(-25, 25, size=2)]:
        for transl in [0] + [1] * 10: #, *np.random.randint(-45, 45, size=2)]:
            for bg in ((1, 1, 1), (0, 0, 0), 'method'):
                if not op.exists(f"./images/{ctr}"):
                    os.mkdir(f"./images/{ctr}")
                start = time.time()
                images = []
                for i, img_fn in enumerate(all_img_fns):
                    image = Image.open(img_fn)
                    if bg == 'method':
                        image = image.convert('RGB')
                    else:
                        image = rgba2rgb(image, background=bg)
                        image = Image.fromarray((image * 255).astype(np.uint8))

                    # add noise
                    if noise:
                        noise = np.abs(np.random.randn(1))
                    transf_image = img_as_ubyte(random_noise(np.asarray(image), var=noise/10**2))

                    # rotate
                    if angle:
                        angle = np.random.randint(-25, 25, size=1)
                    transf_image = img_as_ubyte(rotate(transf_image, angle=angle, mode='wrap'))

                    if transl:
                        transl = np.random.randint(-45, 45, size=1)
                    transform = AffineTransform(translation=(transl,transl))
                    transf_image = img_as_ubyte(warp(transf_image, transform,mode='wrap'))

                    image = Image.fromarray(transf_image)

                    image.save(f"./images/{ctr}/{op.basename(img_fn)}")

                print(f"took {time.time() - start:.2f}s for ctr {ctr}")
                ctr += 1



