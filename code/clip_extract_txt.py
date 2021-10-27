from transformers import CLIPProcessor, CLIPModel
import torch
import argparse
from glob import glob
import os.path as op
import os
import time
import numpy as np
from ipdb import set_trace
import matplotlib.pyplot as plt 
from tqdm import tqdm
plt.ion()

from utils import *

parser = argparse.ArgumentParser(description='extract text embeddings from CLIP')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--folder', default='scene', help='stimuli folder')
parser.add_argument('-s', '--save-embs', action='store_true', help='remove left and right, just and "X and Y"')
parser.add_argument('--embs-out-fn', default='hug_v2', help='output directory for embeddings')
parser.add_argument('-n', '--ncolors', default=10, type=int, help='total number of colors')
parser.add_argument('-w', '--overwrite', action='store_true', help='whether to overwrite the output directory or not')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
all_img_fns = augment_text_with_colors(all_img_fns, ncolors=args.ncolors)
all_captions = [fn[:-4].lower() for fn in all_img_fns]
# print(all_captions)
# separator = " to the X of a "


if args.save_embs: 
    all_txt_embs = []
    all_txt_embs_seq = []
with torch.no_grad():
    start = time.time()
    inputs = processor(text=all_captions, return_tensors="pt", padding=True)
    outputs = model.text_model(**inputs, output_hidden_states=args.save_embs, return_dict=True)


if args.save_embs:
    ## Shape n_trials (162) * sequence length (11) * n_hidden (512)
    np.save(f"{out_dir}/txt_embs_seq_{args.ncolors}colors.npy", outputs.last_hidden_state.numpy())
    ## Shape n_layers * n_trials (162) * sequence length (11) * n_hidden (512)
    np.save(f"{out_dir}/all_txt_embs_{args.ncolors}colors.npy", np.stack(outputs.hidden_states))
    ## Shape n_trials (162) * n_hidden (512)
    np.save(f"{out_dir}/txt_embs_{args.ncolors}colors.npy", outputs.pooler_output.numpy())

print("All done")

# templates = ["a bad photo of a {}.",
#              "an origami {}.",
#              "a photo of a large {}.",
#              "a {} in a video game.",
#              "art of a {}.",
#              "a drawing of a {}.",
#              "a photo of a small {}."]