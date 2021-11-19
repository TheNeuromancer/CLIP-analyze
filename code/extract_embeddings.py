from transformers import CLIPProcessor, CLIPModel
import torch
import argparse
from glob import glob
import os.path as op
import os
import time
import pickle
import numpy as np
from ipdb import set_trace
import matplotlib.pyplot as plt 
from PIL import Image
from tqdm import tqdm
plt.ion()

from utils import *

parser = argparse.ArgumentParser(description='extract text embeddings from CLIP')
parser.add_argument('-r', '--root-path', default='/private/home/tdesbordes/codes/CLIP-analyze/', help='root path')
parser.add_argument('-f', '--text-input-file', default='modifiers_color.txt', help='stimuli file (should be in {args.root_path}/stimuli/text/')
parser.add_argument('-d', '--image-input-dir', default='original_meg_images/scene', help='stimuli file (should be in {args.root_path}/stimuli/images')
parser.add_argument('--embs-out-dir', default='hug_v3', help='output directory for embeddings')
parser.add_argument('-m ', '--model', default='clip-vit-base-patch32', help='type of model')
parser.add_argument('--text-or-vision', default='text', help='part of the model to save embeddings from, can be "text", "vision", or "both"')
parser.add_argument('-w', '--overwrite', action='store_true', default=False, help='whether to overwrite the output directory or not')
args = parser.parse_args()
assert args.text_or_vision in ("text", "vision", "both")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(f"openai/{args.model}")
processor = CLIPProcessor.from_pretrained(f"openai/{args.model}")

# Setup outputs
out_dir = f"{args.root_path}/embeddings/{args.embs_out_dir}"
if not op.exists(out_dir): os.makedirs(out_dir)

out_fn = f"{args.model.replace('-', '_')}"
if args.text_or_vision in ("text", "both"): out_fn += f"_text-{args.text_input_file[0:-4]}-"
if args.text_or_vision in ("vision", "both"): out_fn += f"_vision-{args.image_input_dir.replace('/', '_')}-"
if op.exists(f"{out_dir}/{out_fn}.npy"):
    if args.overwrite:
        print(f"Output file already exists ... overwriting")
    else:
        print(f"Output file already exists and args.overwrite is set to False ... exiting")
        exit()
    

## Load inputs
if args.text_or_vision in ("text", "both"):
    with open(f"{args.root_path}/stimuli/text/{args.text_input_file}", "r") as f: 
        all_sentences = f.readlines()
    all_sentences = [s.rstrip().lower() for s in all_sentences]
    assert np.all([len(all_sentences[0].split()) == len(s.split()) for s in all_sentences]), "all sentences should be the same length"

if args.text_or_vision in ("vision", "both"):
    all_imgs_fns = os.listdir(f"{args.root_path}/stimuli/images/{args.image_input_dir}")
    all_imgs = [Image.open(f"{args.root_path}/stimuli/images/{args.image_input_dir}/{fn}") for fn in all_imgs_fns]
    all_imgs = [im.convert('RGB') for im in all_imgs]

if args.text_or_vision == "both":
    assert len(all_sentences) == len(all_imgs), f"Number of sentences ({len(all_sentences)}) and number of images ({ len(all_imgs)}) should match."
    # sort sentences based on img filenames
    img_labels = [s[0:-4].lower() for s in all_imgs_fns]
    img_labels = [s.replace("to the", "is").replace("of", "to") for s in img_labels]
    assert np.all([s in all_sentences for s in img_labels]), "Some image do not have a matching sentence in the text input file"
    # all sentences and img_labels contain the same images, but ordered differently
    # we just set all_sentences = img_labels to get the sentences ordered similarly to the images
    all_sentences = img_labels


## Run the model
with torch.no_grad():
    start_time = time.time()
    if args.text_or_vision == "text":
        print(f"Starting processing {len(all_sentences)} sentences")
        inputs = processor(text=all_sentences, return_tensors="pt", padding=True)
        model = model.text_model # keep only the language part of the model
    elif args.text_or_vision == "vision":
        print(f"Starting processing {len(all_imgs)} images")
        inputs = processor(images=all_imgs, return_tensors="pt", padding=True)
        model = model.vision_model # keep only the vision part of the model
    elif args.text_or_vision == "both":
        print(f"Starting processing {len(all_sentences)} sentences and {len(all_imgs)} images")
        inputs = processor(text=all_sentences, images=all_imgs, return_tensors="pt", padding=True)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # outputs = model.text_model(**inputs, output_hidden_states=True, return_dict=True)


print(f"Done in {time.time() - start_time:.2f}s")
print(f"Saving outputs to f'{out_dir}/{out_fn}.p'") 

out_dict = {}
if args.text_or_vision == "text":
    out_dict["sentences"] = all_sentences
    out_dict["pooled_output_text"] = outputs.pooler_output.numpy() # Shape n_trials (162) * n_hidden (512)
    out_dict["all_outputs_text"] = np.stack(outputs.hidden_states) # Shape n_layers * n_trials (162) * sequence length (11) * n_hidden (512)

elif args.text_or_vision == "vision":
    out_dict["image_fns"] = all_imgs_fns
    out_dict["pooled_output_vision"] = outputs.pooler_output.numpy() # Shape n_trials (162) * n_hidden (512)
    out_dict["all_outputs_vision"] = np.stack(outputs.hidden_states) # Shape n_layers * n_trials (162) * image "length" (50) * n_hidden (512)

elif args.text_or_vision == "both":
    out_dict["sentences"] = all_sentences
    out_dict["image_fns"] = all_imgs_fns
    out_dict["pooled_output_text"] = outputs.text_model_output.pooler_output.numpy() # Shape n_trials (162) * n_hidden (512)
    out_dict["all_outputs_text"] = np.stack(outputs.text_model_output.hidden_states) # Shape n_layers * n_trials (162) * sequence length (11) * n_hidden (512)
    out_dict["pooled_output_vision"] = outputs.vision_model_output.pooler_output.numpy() # Shape n_trials (162) * n_hidden (512)
    out_dict["all_outputs_vision"] = np.stack(outputs.vision_model_output.hidden_states) # Shape n_layers * n_trials (162) * image "length" (50) * n_hidden (512)

    # should we look at outputs.text_embeds and image_embeds? It is the same as pooled_output but after applying the projection layer (just before doing dot_product between the text and image vectors)
  
pickle.dump(out_dict, open(f"{out_dir}/{out_fn}.p", "wb"))

print("All done")

# templates = ["a bad photo of a {}.",
#              "an origami {}.",
#              "a photo of a large {}.",
#              "a {} in a video game.",
#              "art of a {}.",
#              "a drawing of a {}.",
#              "a photo of a small {}."]