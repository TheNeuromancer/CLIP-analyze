# Does't work on GPU
from PIL import Image
import numpy as  np
import os
from transformers import BertTokenizerFast
from tqdm import tqdm
from ipdb import set_trace
import pickle
import torch

kind = "visualbert" 
lang = "en"

if lang == "en":
    targets = ["red triangle", "green triangle", "blue triangle", "red square", "green square", "blue square", "red circle", "green circle", "blue circle"] 
elif lang == "fr":
    targets = ["triangle rouge", "triangle vert", "triangle bleu", "carré rouge", "carré vert", "carré bleu", "cercle rouge", "cercle vert", "cercle bleu"] 

if kind == "flaxbert":
    from transformers import CLIPProcessor
    from modeling_clip_vision_bert import FlaxCLIPVisionBertForSequenceClassification
elif kind == "visualbert":
    from transformers import VisualBertForQuestionAnswering
    from visualbert_utils import Config, get_data
    from visualbert_modeling_frcnn import GeneralizedRCNN
    from visualbert_processing_image import Preprocess
else:
    raise NotImplementedError(f"Unknown kind of model {kind}")

if kind == "flaxbert":
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    model = FlaxCLIPVisionBertForSequenceClassification.from_pretrained('flax-community/clip-vision-bert-vqa-ft-6k')
elif kind == "visualbert":
    VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"
    vqa_answers = get_data(VQA_URL)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-vqa-pre')
    model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-vqa')
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)


path_to_images = '../stimuli/images/original_meg_images/scene' 
image_fns = os.listdir(path_to_images)


perf_dict = {}
with torch.no_grad():
    for target in targets:
        print(f"doing target {target}")
        if lang == "en":
            text = f"Is there a {target} in the image?"
        elif lang == "fr":
            text = f"Y-a-t'il un {target} dans l'image ?"
        inputs = tokenizer([text], return_tensors="pt")

        perf = []
        for img_fn in tqdm(image_fns):
            img_path = os.path.join(path_to_images, img_fn)
            if kind == "flaxbert":
                img = Image.open(img_path).convert('RGB')
                clip_outputs = clip_processor(images=img)
                clip_outputs['pixel_values'][0] = clip_outputs['pixel_values'][0].transpose(1,2,0) # Need to transpose images as model expected channel last images.
                pixel_values = np.concatenate([clip_outputs['pixel_values']])
                outputs = model(pixel_values=pixel_values, **inputs)
            elif kind == "visualbert":
                # run frcnn to get visual features
                images, sizes, scales_yx = image_preprocess(img_path)
                output_dict = frcnn(images, sizes, scales_yx=scales_yx, padding="max_detections", max_detections=frcnn_cfg.max_detections, return_tensors="pt")
                visual_embeds = output_dict.get("roi_features")
                # visual_embeds = get_visual_embeddings(image).unsqueeze(0)
                visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
                inputs.update({"visual_embeds": visual_embeds, "visual_token_type_ids": visual_token_type_ids, "visual_attention_mask": visual_attention_mask })
                outputs = model(**inputs)

            preds = outputs.logits[0]
            pred_idx = preds.argmax()
            if kind == "flaxbert":
                pred = model.config.id2label[pred_idx.item()]
            elif kind == "visualbert":
                pred = vqa_answers[pred_idx]
            if target in img_fn:
                if pred == "yes":
                    perf.append(1)
                elif pred == "no":
                    perf.append(0)
                else:
                    perf.append(0)
                    print(pred)
            else:
                if pred == "no":
                    perf.append(1)
                elif pred == "yes":
                    perf.append(0)
                else:
                    perf.append(0)
                    print(pred)
        perf_dict[target] = perf
        print(f"Performance for target: {target} = {np.mean(perf):.2f}", "\n")

out_path = f"../results/VQA_behavior/isXinImg_all_{lang}.p"
pickle.dump(perf_dict, open(out_path, "wb"))