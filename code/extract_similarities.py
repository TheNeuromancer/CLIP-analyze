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

parser = argparse.ArgumentParser(description='extract text embeddings from CLIP for the original MEG stimuli')
parser.add_argument('-r', '--root-path', default='/Users/tdesbordes/Documents/CLIP-analyze/', help='root path')
parser.add_argument('-d', '--image-input-dir', default='original_meg_images/scene', help='stimuli file (should be in {args.root_path}/stimuli/images')
parser.add_argument('--out-dir', default='hug_v3', help='output directory for results')
parser.add_argument('-m ', '--model', default='openai/clip-vit-base-patch32', help='type of model') # openai/clip-vit-base-patch32, M-CLIP/M-BERT-Base-69, flax-community/clip-rsicd-v2
parser.add_argument('-w', '--overwrite', action='store_true', default=False, help='whether to overwrite the output directory or not')
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(args.model)
processor = CLIPProcessor.from_pretrained(args.model)

# Setup outputs
out_dir = f"{args.root_path}/results/behavior/{args.out_dir}"
if not op.exists(out_dir): os.makedirs(out_dir)

out_fn = f"{args.model.replace('/', '_')}_{args.image_input_dir.replace('/', '_')}-2obj"
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
all_sentences = [s.replace("to the left of", "is left to").replace("to the right of", "is right to") for s in all_sentences]

shapes = ["triangle", "square", "circle"]
colors = ["blue", "red", "green"]
positions = {"S1": 2, "C1": 1, "S2": 8, "C2": 7}

def get_property_mismatch(sent):
    viol = np.random.choice(("S1", "C1", "S2", "C2"))
    changed_property = "Shape" if viol[0]=='S' else "Color"
    words = sent.split()
    old_value = words[positions[viol]]
    if viol[0] == "S": # shape
        words[positions[viol]] = np.random.choice([s for s in shapes if s != old_value])
    else: # color
        words[positions[viol]] = np.random.choice([c for c in colors if c != old_value])
    violation_position = positions[viol]
    violation_side = "Left" if (violation_position < 3 and "left" in sent) or (violation_position > 3 and "right" in sent) else "Right"
    return " ".join(words), violation_position, violation_side, changed_property

def get_binding_mismatch(sent):
    words = sent.split()
    S1, C1, S2, C2 = (words[idx] for idx in positions.values())
    # if the 2 objects share any property then this error is not defined.
    if S1 == S2: return "None"
    if C1 == C2: return "None"

    viol = np.random.choice(("S", "C"))
    if viol == "S": # shape
        words[positions["S1"]], words[positions["S2"]] = words[positions["S2"]], words[positions["S1"]]
    else: # color
        words[positions["C1"]], words[positions["C2"]] = words[positions["C2"]], words[positions["C1"]]
    return " ".join(words)

def get_relation_mismatch(sent):
    # if the 2 objects share both properties then this error is not defined.
    words = sent.split()
    S1, C1, S2, C2 = (words[idx] for idx in positions.values())
    if S1 == S2 and C1 == C2: return "None"

    if "left" in sent:
        return sent.replace("left", "right")
    if "right" in sent:
        return sent.replace("right", "left")

def get_features(sent):
    words = sent.split()
    s1, c1, s2, c2 = (words[idx] for idx in positions.values())
    d = 2 # get difficulty
    sharing = []# which features are shared
    if s1 == s2: 
        d -= 1
        sharing.append(True)
    else:
        sharing.append(False)
    if c1 == c2: 
        d -= 1
        sharing.append(True)
    else:
        sharing.append(False)
    nb_shared = 2 - d
    if sum(sharing) == 2: 
        sharing = "Both"
    elif sharing[0]:
        sharing = "Shape"
    elif sharing[1]:
        sharing = "Color"
    else:
        sharing = "None"
    if "left" in sent:
        rel = "left"
    else:
        rel = "right"
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


property_mismatches = []
binding_mismatches = []
relation_mismatches = []
property_mismatch_position = []
property_mismatch_side = []
changed_properties = []
for sent in all_sentences:
    property_mismatch, violation_position, violation_side, changed_property = get_property_mismatch(sent)
    property_mismatches.append(property_mismatch)
    property_mismatch_position.append(violation_position)
    property_mismatch_side.append(violation_side)
    binding_mismatches.append(get_binding_mismatch(sent))
    relation_mismatches.append(get_relation_mismatch(sent))
    changed_properties.append(changed_property)

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
out_dict["sent_id"] = np.arange(len(all_sentences))


## Run the model
with torch.no_grad():
    start_time = time.time()
    print(f"Starting processing {len(all_sentences)}*4 sentences and {len(all_imgs)} images")
    n_sents = len(all_sentences)
    # all_input_sents = np.hstack((all_sentences, property_mismatches, binding_mismatches, relation_mismatches))
    all_input_sents = all_sentences + property_mismatches + binding_mismatches + relation_mismatches
    all_input_imgs = all_imgs
    inputs = processor(text=all_input_sents, images=all_input_imgs, return_tensors="pt", padding=True)
    outputs = model(**inputs, output_hidden_states=False, return_dict=True)

    # set_trace()
    similarity_true = np.diag(outputs['logits_per_image'][:,0:n_sents]).tolist()
    similarity_property = np.diag(outputs['logits_per_image'][:,n_sents:n_sents*2]).tolist()
    similarity_binding = np.diag(outputs['logits_per_image'][:,n_sents*2:n_sents*3]).tolist()
    similarity_relation = np.diag(outputs['logits_per_image'][:,n_sents*3:n_sents*4]).tolist()

    Perf = np.logical_and(np.logical_and(np.array(similarity_true) > np.array(similarity_property),
                                         np.array(similarity_true) > np.array(similarity_binding)),
                                         np.array(similarity_true) > np.array(similarity_relation))

    # go from tensor to tupple
    # for img, sent, prop, bind, rel in zip(all_imgs, all_sentences, property_mismatches, binding_mismatches, relation_mismatches):
    #     inputs = processor(text=(sent, prop, bind, rel), images=img, return_tensors="pt", padding=True)
    #     outputs = model(**inputs, output_hidden_states=False, return_dict=True)
    #     Perf.append(1 if outputs['logits_per_image'].argmax().item()==0 else 0)
    #     similarity_true.append(outputs['logits_per_image'][0][0].item())
    #     similarity_property.append(outputs['logits_per_image'][0][1].item())
    #     similarity_binding.append(outputs['logits_per_image'][0][2].item())
    #     similarity_relation.append(outputs['logits_per_image'][0][3].item())
    #     # The model is run for a single sentence and one instance of each violatioh
    #     # results are stored separately, then put in a copy of the original dataframe, 
    #     # and then concatenated

print(f"Done in {time.time() - start_time:.2f}s")
print(f"Average Performance is: {np.mean(Perf)}")

print(f"Saving outputs to f'{out_dir}/{out_fn}.p'") 

out_dict["image_fns"] = all_imgs_fns
out_dict["Trial type"] = ["Long" for _ in Perf]
out_dict["NbObjects"] = [2 for _ in Perf]
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

## PROPERTTY
df_property = pd.DataFrame.from_dict(out_dict)
# perf for each error type is whether the similarity is lower compared top this error type only.
df_property["Perf"] = (np.array(similarity_true) > np.array(similarity_property)).astype(float)
df_property["similarity"] = similarity_property
df_property["Violation"] = "Yes"
df_property["Error_type"] = "l0"
df_property["Violation on"] = "property"
df_property["Changed property"] = changed_properties
df_property["sentences"] = property_mismatches
df_property["property_mismatches_positions"] = property_mismatch_position
df_property["property_mismatches_order"] = ["First" if p<3 else "Second" for p in property_mismatch_position ]
df_property["property_mismatches_side"] = property_mismatch_side
# get the new values for each property and save them with lowercase
shape1, color1, shape2, color2, relation, D, Nb_shared, Sharing = get_all_features(property_mismatches)
df_property["shape1"] = shape1
df_property["shape2"] = shape2
df_property["color1"] = color1
df_property["color2"] = color2
df_property["relation"] = relation
df_property["Difficulty"] = D
df_property["# Shared features"] = Nb_shared
df_property["Sharing"] = Sharing


## BINDING
df_binding = pd.DataFrame.from_dict(out_dict)
# perf for each error type is whether the similarity is lower compared top this error type only.
df_binding["Perf"] = (np.array(similarity_true) > np.array(similarity_binding)).astype(float)
df_binding["similarity"] = similarity_binding
df_binding["Violation"] = "Yes"
df_binding["Error_type"] = "l1"
df_binding["Violation on"] = "binding"
df_true["Changed property"] = "None"
df_binding["sentences"] = binding_mismatches
df_binding["property_mismatches_positions"] = ["None" for s in all_sentences]
df_binding["property_mismatches_order"] = ["None" for s in all_sentences]
df_binding["property_mismatches_side"] = ["None" for s in all_sentences]
# get the new values for each property and save them with lowercase
shape1, color1, shape2, color2, relation, D, Nb_shared, Sharing = get_all_features(binding_mismatches)
df_binding["shape1"] = shape1
df_binding["shape2"] = shape2
df_binding["color1"] = color1
df_binding["color2"] = color2
df_binding["relation"] = relation
df_binding["Difficulty"] = D
df_binding["# Shared features"] = Nb_shared
df_binding["Sharing"] = Sharing


## RELATION
df_relation = pd.DataFrame.from_dict(out_dict)
# perf for each error type is whether the similarity is lower compared top this error type only.
df_relation["Perf"] = (np.array(similarity_true) > np.array(similarity_relation)).astype(float)
df_relation["similarity"] = similarity_relation
df_relation["Violation"] = "Yes"
df_relation["Error_type"] = "l2"
df_relation["Violation on"] = "relation"
df_true["Changed property"] = "None"
df_relation["sentences"] = relation_mismatches
df_relation["property_mismatches_positions"] = ["None" for s in all_sentences]
df_relation["property_mismatches_order"] = ["None" for s in all_sentences]
df_relation["property_mismatches_side"] = ["None" for s in all_sentences]
# get the new values for each property and save them with lowercase
shape1, color1, shape2, color2, relation, D, Nb_shared, Sharing = get_all_features(relation_mismatches)
df_relation["shape1"] = shape1
df_relation["shape2"] = shape2
df_relation["color1"] = color1
df_relation["color2"] = color2
df_relation["relation"] = relation
df_relation["Difficulty"] = D
df_relation["# Shared features"] = Nb_shared
df_relation["Sharing"] = Sharing


df = pd.concat((df_true, df_property, df_binding, df_relation))
df.reset_index(inplace=True, drop=True) # reset index
df["Error rate"] = np.logical_not(df["Perf"]).astype(float)
df.to_csv(f"{out_dir}/{out_fn}.csv")
print(f"Average Performance for property is: {df_property.Perf.mean()}")
print(f"Average Performance for binding is: {df_binding.Perf.mean()}")
print(f"Average Performance for relation is: {df_relation.Perf.mean()}")
# set_trace()
print("All done")
