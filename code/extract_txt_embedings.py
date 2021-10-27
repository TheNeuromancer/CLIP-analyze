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
parser.add_argument('-f', '--input-file', default='/private/home/tdesbordes/codes/CLIP-analyze/tmp.txt', help='stimuli file')
parser.add_argument('--embs-out-dir', default='test', help='output directory for embeddings')
parser.add_argument('-m ', '--model', default='clip-vit-base-patch32', help='type of model')
parser.add_argument('-w', '--overwrite', action='store_true', help='whether to overwrite the output directory or not')
parser.add_argument('--save_layers', action='store_true', help='whether to save an embedding for each layer or just the last one')
parser.add_argument('--save_sequence', action='store_true', help='whether to save an embedding for each word in the sentence')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(f"openai/{args.model}")
processor = CLIPProcessor.from_pretrained(f"openai/{args.model}")

out_dir = f"{args.root_path}/embeddings/{args.embs_out_dir}"
if op.exists(out_dir):
    if args.overwrite:
        print(f"Output directory already exists ... overwriting")
    else:
        print(f"Output directory already exists and args.overwrite is set to False ... exiting")
        exit()
else:
    os.makedirs(out_dir)
out_fn = f"{op.basename(args.input_file)[0:-4]}_{args.model.replace('-', '_')}"
if args.save_layers: out_fn += "_all_layers"
if args.save_sequence: out_fn += "_all_sequence"


## Load input sentences
with open(args.input_file, "r") as f: 
    all_sentences = f.readlines()

all_sentences = [s.rstrip().lower() for s in all_sentences]

if args.save_sequence:
    assert np.all([len(all_sentences[0].split()) == len(s.split()) for s in all_sentences]), \
    "if saving all sequence element (one embedding per word in the sentence) then all sentences should be the same length"

all_txt_embs = []
all_txt_embs_seq = []
with torch.no_grad():
    start_time = time.time()
    print(f"Starting processing {len(all_sentences)} sentences from {args.input_file}")
    inputs = processor(text=all_sentences, return_tensors="pt", padding=True)
    outputs = model.text_model(**inputs, output_hidden_states=True, return_dict=True)


print(f"Done in {time.time() - start_time:.2f}s")
print(f"Saving outputs to f'{out_dir}/{out_fn}'") 

if args.save_layers:
    if args.save_sequence:
        ## Shape n_layers * n_trials (162) * sequence length (11) * n_hidden (512)
        np.save(f"{out_dir}/{out_fn}", np.stack(outputs.hidden_states))
    else:
        ## Shape n_layers * n_trials (162) * n_hidden (512)
        np.save(f"{out_dir}/{out_fn}", np.stack(outputs.hidden_states)[:,:,-1,:])
else:
    if args.save_sequence:
        ## Shape n_trials (162) * sequence length (11) * n_hidden (512)
        np.save(f"{out_dir}/{out_fn}", outputs.last_hidden_state.numpy())
    else:
        ## Shape n_trials (162) * n_hidden (512)
        np.save(f"{out_dir}/{out_fn}", outputs.pooler_output.numpy())

print("All done")

# templates = ["a bad photo of a {}.",
#              "an origami {}.",
#              "a photo of a large {}.",
#              "a {} in a video game.",
#              "art of a {}.",
#              "a drawing of a {}.",
#              "a photo of a small {}."]