# Does't work on GPU
from PIL import Image
import numpy as  np
import os
from tqdm import tqdm
from ipdb import set_trace
import pickle
import argparse
from glob import glob
import os.path as op
import matplotlib.pyplot as plt 
import io
import time
from transformers import BertTokenizerFast
# flaxbert
from transformers import CLIPProcessor
from modeling_clip_vision_bert import FlaxCLIPVisionBertForSequenceClassification
# visualbert
from transformers import VisualBertForQuestionAnswering
from visualbert_utils import Config, get_data
from visualbert_modeling_frcnn import GeneralizedRCNN
from visualbert_processing_image import Preprocess
# lxmert
from transformers import LxmertForQuestionAnswering, LxmertTokenizer

from utils import *


parser = argparse.ArgumentParser(description='extract text embeddings from CLIP for the original MEG stimuli')
parser.add_argument('-r', '--root-path', default='/Users/tdesbordes/Documents/CLIP-analyze/', help='root path')
parser.add_argument('-d', '--image-input-dir', default='original_meg_images/scene', help='stimuli file (should be in {args.root_path}/stimuli/images')
parser.add_argument('--out-dir', default='lxmert-vqa', help='output directory for results')
# parser.add_argument('-m', '--model', default='uclanlp/lxmert-vqa', help='type of model') # flax-community/clip-vision-bert-vqa-ft-6k
parser.add_argument('-k', '--model-type', default='visualbert', help='kind of model, "lxmert" or "flaxbert"')
parser.add_argument('-w', '--overwrite', action='store_true', default=False, help='whether to overwrite the output directory or not')
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using model: {args.model_type}")

# Setup outputs
out_dir = f"{args.root_path}/results/behavior/{args.out_dir}"
if not op.exists(out_dir): os.makedirs(out_dir)

out_fn = f"{args.model_type}_{args.image_input_dir.replace('/', '_')}-2obj"
if op.exists(f"{out_dir}/{out_fn}.npy"):
    if args.overwrite:
        print(f"Output file already exists ... overwriting")
    else:
        print(f"Output file already exists and args.overwrite is set to False ... exiting")
        exit()

templates = ["Is there a {} to the {} of a {} in the image?"]
             # "Is the {} on the {} of the {}?",
             # "The {} is on the {} of the {}",
             # "There is a {} to the {} of a {}",
             # "There is a {} on the {} and a {}",
             # "Is there is a {} on the {} and a {}?"]
# template = "Is there a {} to the {} of a {} in the image?"
# template = "Is there is a {} on the {} and a {} on the {}?"
print(f"Using templates \"{templates}\"")


# ## Load models
# clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
# model = FlaxCLIPVisionBertForSequenceClassification.from_pretrained(args.model)
if args.model_type == "flaxbert":
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    model = FlaxCLIPVisionBertForSequenceClassification.from_pretrained('flax-community/clip-vision-bert-vqa-ft-6k')
elif args.model_type == "visualbert":
    VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"
    vqa_answers = get_data(VQA_URL)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-vqa-pre')
    model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-vqa')
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
elif args.model_type == "lxmert":
    VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
    vqa_answers = get_data(VQA_URL)
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
    tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
    model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")

else:
    raise NotImplementedError(f"Uknown type of model {args.model_type}")

## Load inputs
all_imgs_fns = os.listdir(f"{args.root_path}/stimuli/images/{args.image_input_dir}")
all_sentences = [s[0:-4].lower() for s in all_imgs_fns]
all_sentences = [s.replace("to the left of", "is left to").replace("to the right of", "is right to") for s in all_sentences]

shapes = ["triangle", "square", "circle"]
colors = ["blue", "red", "green"]
positions = {"S1": 2, "C1": 1, "S2": 8, "C2": 7}

def get_property_mismatch(sent):
    viol = np.random.choice(("S1", "C1", "S2", "C2"))
    words = sent.split()
    old_value = words[positions[viol]]
    if viol[0] == "S": # shape
        words[positions[viol]] = np.random.choice([s for s in shapes if s != old_value])
    else: # color
        words[positions[viol]] = np.random.choice([c for c in colors if c != old_value])
    violation_position = positions[viol]
    violation_side = "Left" if (violation_position < 3 and "left" in sent) or (violation_position > 3 and "right" in sent) else "Right"
    return " ".join(words), violation_position, violation_side

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

def get_questions(template, *argv):
    # from sentence label go to question using the question template
    all_questions = []
    for sent in argv:
        if sent == "None": # for some error type some sentences are not defined, return nones
            s1, c1, s2, c2, rel, d, nb_shared, sharing =  "None", "None", "None", "None", "None", "None", "None", "None"
        else:
            s1, c1, s2, c2, rel, _, _, _ = get_features(sent)
        all_questions.append(template.format(f"{c1} {s1}", rel, f"{c2} {s2}"))
        # relrev = {"left": 'right', "right": 'left'}
        # all_questions.append(template.format(f"{c1} {s1}", rel, f"{c2} {s2}", relrev))
    return all_questions


property_mismatches = []
binding_mismatches = []
relation_mismatches = []
property_mismatch_position = []
property_mismatch_side = []
for sent in all_sentences:
    property_mismatch, violation_position, violation_side = get_property_mismatch(sent)
    property_mismatches.append(property_mismatch)
    property_mismatch_position.append(violation_position)
    property_mismatch_side.append(violation_side)
    binding_mismatches.append(get_binding_mismatch(sent))
    relation_mismatches.append(get_relation_mismatch(sent))

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
# default information
out_dict["image_fns"] = all_imgs_fns
out_dict["Trial type"] = ["Long" for _ in all_imgs_fns]
out_dict["NbObjects"] = [2 for _ in all_imgs_fns]

perf_dict = {"match": [], "property": [], "binding": [], "relation": []}
with torch.no_grad():
    for template in templates:
        print(f"Starting tempalte {template}")
        start_time = time.time()
        print(f"Starting processing {len(all_sentences)}*4 trials")
        for img_fn, match, prop, bind, rel in tqdm(zip(all_imgs_fns, all_sentences, property_mismatches, binding_mismatches, relation_mismatches)):
            img_path = f"{args.root_path}/stimuli/images/{args.image_input_dir}/{img_fn}"
            questions = get_questions(template, match, prop, bind, rel)

            # inputs = tokenizer([q_match, q_prop, q_bind, q_rel], return_tensors="pt")
            if args.model_type == "flaxbert":
                # image model
                img = Image.open(img_path).convert('RGB')
                clip_outputs = clip_processor(images=img)
                clip_outputs['pixel_values'][0] = clip_outputs['pixel_values'][0].transpose(1,2,0) # Need to transpose images as model expected channel last images.
                pixel_values = np.concatenate([clip_outputs['pixel_values']])
                pixel_values = np.repeat(pixel_values, 4, axis=0) # copy 4 times because we have 4 questions
                # text model
                outputs = model(pixel_values=pixel_values, **inputs)
                logits = outputs.logits
            elif args.model_type == "visualbert":
                # image model
                image, sizes, scales_yx = image_preprocess(img_path)
                output_dict = frcnn(image, sizes, scales_yx=scales_yx, padding="max_detections", max_detections=frcnn_cfg.max_detections, return_tensors="pt")
                # normalized_boxes = output_dict.get("normalized_boxes")
                visual_embeds = output_dict.get("roi_features")
                visual_embeds = torch.repeat_interleave(visual_embeds, 4, dim=0)
                visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
                # text model
                inputs = tokenizer(questions, padding="max_length", max_length=20, truncation=True, return_token_type_ids=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")
                inputs.update({"visual_embeds": visual_embeds, "visual_token_type_ids": visual_token_type_ids, "visual_attention_mask": visual_attention_mask})
                outputs = model(**inputs)
                logits = outputs.logits
            elif args.model_type == "lxmert":
                # image model
                image, sizes, scales_yx = image_preprocess(img_path)
                output_dict = frcnn(image, sizes, scales_yx=scales_yx, padding="max_detections", max_detections=frcnn_cfg.max_detections, return_tensors="pt")
                normalized_boxes = output_dict.get("normalized_boxes") # Very important that the boxes are normalized
                visual_feats = output_dict.get("roi_features")
                # text model
                inputs = tokenizer(questions, padding="max_length", max_length=20, truncation=True, return_token_type_ids=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, visual_feats=visual_feats, visual_pos=normalized_boxes, token_type_ids=inputs.token_type_ids, output_attentions=False)
                logits = outputs["question_answering_score"]

            # get performance for each kind of sentence (original and 3 types of mismaches)
            for sent, preds, kind in zip((match, prop, bind, rel), logits, ("match", "property", "binding", "relation")):
                if sent == "None": # if the error is not defined store nan as performance
                    perf_dict[kind] == np.nan
                    continue
                
                # preds = outputs.logits[0]
                pred_idx = preds.argmax()   

                if args.model_type == "flaxbert":
                    pred = model.config.id2label[pred_idx.item()]
                elif args.model_type in ("visualbert", "lxmert"):
                    pred = vqa_answers[pred_idx]

                if kind == "match":
                    if pred in ["yes", "right"]:
                        perf_dict[kind].append(1)
                    elif pred == "no":
                        perf_dict[kind].append(0)
                    else:
                        perf_dict[kind].append(0)
                        print(f"Unknown model's answer: {pred}")
                else:
                    if pred == "no":
                        perf_dict[kind].append(1)
                    elif pred in ["yes", "right"]:
                        perf_dict[kind].append(0)
                    else:
                        perf_dict[kind].append(0)
                        print(f"Unknown model's answer: {pred}")

        print(f"Performance so far: {np.mean(perf_dict['match'])}, {np.mean(perf_dict['property'])}, {np.mean(perf_dict['binding'])}, {np.mean(perf_dict['relation'])}", "\n")

print(f"Done in {time.time() - start_time:.2f}s")

for kind in ("match", "property", "binding", "relation"):
    print(f"Perf for kind {kind}: {np.nanmean(perf_dict[kind]):.2f}")

set_trace()

# out_path = f"../results/VQA_behavior/isXinImg_all_{lang}.p"
# pickle.dump(perf_dict, open(out_path, "wb"))


print(f"Saving outputs to f'{out_dir}/{out_fn}.p'") 
# pickle.dump(out_dict, open(f"{out_dir}/{out_fn}.p", "wb"))

# make a dataframe for each error type, then concatenate them
df_true = pd.DataFrame.from_dict(out_dict)
df_true["Perf"] = np.array(Perf).astype(float) # perf for normal sent is 1 if the similarity is lowest compared to all error types
df_true["similarity"] = similarity_true
df_true["Violation"] = "No"
df_true["Error_type"] = "None"
df_true["Violation on"] = "None"
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
set_trace()
print("All done")
