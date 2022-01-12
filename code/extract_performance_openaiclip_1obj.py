import clip
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

parser = argparse.ArgumentParser(description='extract text embeddings from CLIP for the original MEG stimuli')
parser.add_argument('-r', '--root-path', default='/Users/tdesbordes/Documents/CLIP-analyze/', help='root path')
parser.add_argument('-d', '--image-input-dir', default='original_meg_images/object', help='stimuli file (should be in {args.root_path}/stimuli/images')
parser.add_argument('--out-dir', default='hug_v3', help='output directory for results')
parser.add_argument('-m ', '--model', default='RN50x16', help='type of model')
parser.add_argument('-w', '--overwrite', action='store_true', default=False, help='whether to overwrite the output directory or not')
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load(args.model, device=device)

# Setup outputs
out_dir = f"{args.root_path}/results/behavior/{args.out_dir}"
if not op.exists(out_dir): os.makedirs(out_dir)

out_fn = f"openai-{args.model.replace('/', '_')}_{args.image_input_dir.replace('/', '_')}-1obj"
if op.exists(f"{out_dir}/{out_fn}.npy"):
    if args.overwrite:
        print(f"Output file already exists ... overwriting")
    else:
        print(f"Output file already exists and args.overwrite is set to False ... exiting")
        exit()


## Load inputs
all_imgs_fns = os.listdir(f"{args.root_path}/stimuli/images/{args.image_input_dir}")
all_imgs = [Image.open(f"{args.root_path}/stimuli/images/{args.image_input_dir}/{fn}") for fn in all_imgs_fns]
all_imgs = [im.convert('RGB') for im in all_imgs]
all_sentences = [s[0:-4].lower() for s in all_imgs_fns]

shapes = ["triangle", "square", "circle"]
colors = ["blue", "red", "green"]
positions = {"S": 2, "C": 1}

def get_color_mismatch(sent):
    words = sent.split()
    old_color = words[positions["C"]]
    words[positions["C"]] = np.random.choice([c for c in colors if c != old_color])
    violation_position = positions["C"]
    violation_side = "None"
    return " ".join(words), violation_position, violation_side

def get_shape_mismatch(sent):
    words = sent.split()
    old_shape = words[positions["S"]]
    words[positions["S"]] = np.random.choice([s for s in shapes if s != old_shape])
    violation_position = positions["S"]
    violation_side = "None"
    return " ".join(words), violation_position, violation_side

def get_features(sent):
    words = sent.split()
    s1, c1 = (words[idx] for idx in positions.values())
    s2, c2, rel = "None", "None", "None"
    d, nb_shared, sharing = "None", "None", "None"
    return s1, c1, s2, c2, rel, d, nb_shared, sharing

def get_all_features(sents):
    Shape1, Shape2, Color1, Color2 = [], [], [], []
    Relation, D, Nb_shared, Sharing = [], [], [], []
    for sent in sents:
        if sent == "None": # for some error type some sentences are not defined, return nones
            s1, c1, s2, c2, rel, d, nb_shared, sharing =  "None", "None", "None", "None", "None", "None", "None", "None"
        else:
            s1, c1, s2, c2, rel, d, nb_shared, sharing = get_features(sent)
        Shape1.append(s1) # uppercase because it is relative to the image (= sentence without violation)
        Shape2.append(s2)
        Color1.append(c1)
        Color2.append(c2)
        Relation.append(rel)
        D.append(d)
        Nb_shared.append(nb_shared)
        Sharing.append(sharing)
    return Shape1, Color1, Shape2, Color2, Relation, D, Nb_shared, Sharing


shape_change_mismatches = []
color_change_mismatches = []
for sent in all_sentences:
    shape_mismatch, violation_position, violation_side = get_shape_mismatch(sent)
    shape_change_mismatches.append(shape_mismatch)
    
    color_mismatch, violation_position, violation_side = get_color_mismatch(sent)
    color_change_mismatches.append(color_mismatch)

# get unviolated features
Shape1, Color1, Shape2, Color2, Relation, D, Nb_shared, Sharing = get_all_features(all_sentences)
out_dict = {}
out_dict["original sentences"] = all_sentences
out_dict["Shape1"] = Shape1
out_dict["Shape2"] = Shape2
out_dict["Color1"] = Color1
out_dict["Color2"] = Color2
out_dict["Relation"] = Relation
out_dict["Difficulty"] = D
out_dict["# Shared features"] = Nb_shared
out_dict["Sharing"] = Sharing
out_dict["sent_id"] = [f"1obj-{i}" for i in np.arange(len(all_sentences))]


# similarity_true = []
# similarity_shape_change = []
# similarity_color_change = []
Perf = []
## Run the model
with torch.no_grad():
    start_time = time.time()
    print(f"Starting processing {len(all_sentences)}*4 sentences and {len(all_imgs)} images")

    for img, sent, shape_change, color_change in zip(all_imgs, all_sentences, shape_change_mismatches, color_change_mismatches):
        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize((sent, shape_change, color_change)).to(device)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        Perf.append(1 if probs.argmax().item()==0 else 0)

        # similarity_true.append(probs[0][0].item())
        # similarity_shape_change.append(probs[0][1].item())
        # similarity_color_change.append(probs[0][2].item())

    # proper way to get similarities
    n_sents = len(all_sentences)
    all_input_sents = all_sentences + shape_change_mismatches + color_change_mismatches
    image_inputs = torch.stack([preprocess(img) for img in all_imgs]).to(device)
    text_inputs = clip.tokenize(all_input_sents).to(device)

    image_features = model.encode_image(image_inputs)
    text_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    all_similarities = (100.0 * image_features @ text_features.T) #.softmax(dim=-1)
    similarity_true = np.diag(all_similarities[:,0:n_sents]).tolist()
    similarity_shape_change = np.diag(all_similarities[:,n_sents:n_sents*2]).tolist()
    similarity_color_change = np.diag(all_similarities[:,n_sents*2:n_sents*3]).tolist()
   



print(f"Done in {time.time() - start_time:.2f}s")
print(f"Average Performance is: {np.mean(Perf)}")

print(f"Saving outputs to f'{out_dir}/{out_fn}.p'") 

out_dict["image_fns"] = all_imgs_fns
out_dict["Trial type"] = ["Short" for _ in Perf]
out_dict["NbObjects"] = [1 for _ in Perf]
# pickle.dump(out_dict, open(f"{out_dir}/{out_fn}.p", "wb"))

# make a dataframe for each error type, then concatenate them
df_true = pd.DataFrame.from_dict(out_dict)
df_true["Perf"] = np.array(Perf).astype(float) # perf for normal sent is 1 if the similarity is lowest compared to all error types
df_true["similarity"] = similarity_true
df_true["Violation"] = "No"
df_true["Error_type"] = "None"
df_true["Violation on"] = "None"
df_true["Changed property"] = "None"
df_true["sentences"] = all_sentences
df_true["property_mismatches_positions"] = ["None" for s in all_sentences]
df_true["property_mismatches_order"] = ["None" for s in all_sentences]
df_true["property_mismatches_side"] = ["None" for s in all_sentences]
# get the new values for each property and save them with lowercase
df_true["shape1"] = df_true["Shape1"]
df_true["shape2"] = df_true["Shape2"]
df_true["color1"] = df_true["Color1"]
df_true["color2"] = df_true["Color2"]
df_true["relation"] = df_true["Relation"]

## SHAPE
df_shape_change = pd.DataFrame.from_dict(out_dict)
# perf for each error type is whether the similarity is lower compared top this error type only.
df_shape_change["Perf"] = (np.array(similarity_true) > np.array(similarity_shape_change)).astype(float)
df_shape_change["similarity"] = similarity_shape_change
df_shape_change["Violation"] = "Yes"
df_shape_change["Error_type"] = "Shape"
df_shape_change["Violation on"] = "Shape"
df_shape_change["Changed property"] = "Shape"
df_shape_change["sentences"] = shape_change_mismatches
df_shape_change["property_mismatches_positions"] = [positions["S"] for s in all_sentences]
df_shape_change["property_mismatches_order"] = ["None" for s in all_sentences]
df_shape_change["property_mismatches_side"] = ["None" for s in all_sentences]
# get the new values for each shape_change and save them with lowercase
shape1, color1, shape2, color2, relation, D, Nb_shared, Sharing = get_all_features(shape_change_mismatches)
df_shape_change["shape1"] = shape1
df_shape_change["shape2"] = shape2
df_shape_change["color1"] = color1
df_shape_change["color2"] = color2
df_shape_change["relation"] = relation
df_shape_change["Difficulty"] = D
df_shape_change["# Shared features"] = Nb_shared
df_shape_change["Sharing"] = Sharing


## COLOR
df_color_change = pd.DataFrame.from_dict(out_dict)
# perf for each error type is whether the similarity is lower compared top this error type only.
df_color_change["Perf"] = (np.array(similarity_true) > np.array(similarity_color_change)).astype(float)
df_color_change["similarity"] = similarity_color_change
df_color_change["Violation"] = "Yes"
df_color_change["Error_type"] = "Color"
df_color_change["Violation on"] = "Color"
df_color_change["Changed property"] = "Color"
df_color_change["sentences"] = color_change_mismatches
df_shape_change["property_mismatches_positions"] = [positions["C"] for s in all_sentences]
df_shape_change["property_mismatches_order"] = ["None" for s in all_sentences]
df_shape_change["property_mismatches_side"] = ["None" for s in all_sentences]
# get the new values for each color_change and save them with lowercase
shape1, color1, shape2, color2, relation, D, Nb_shared, Sharing = get_all_features(color_change_mismatches)
df_color_change["shape1"] = shape1
df_color_change["shape2"] = shape2
df_color_change["color1"] = color1
df_color_change["color2"] = color2
df_color_change["relation"] = relation
df_color_change["Difficulty"] = D
df_color_change["# Shared features"] = Nb_shared
df_color_change["Sharing"] = Sharing


df = pd.concat((df_true, df_shape_change, df_color_change))
df.reset_index(inplace=True, drop=True) # reset index
df["Error rate"] = np.logical_not(df["Perf"]).astype(float)
df.to_csv(f"{out_dir}/{out_fn}.csv")
set_trace()
print("All done")
