import clip
import pandas as pd
import numpy as np
import torch
from ipdb import set_trace
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches


additional_colors = ["yellow", "brown", "purple", "orange", "pink", "gold", "gray"]


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


def zeroshot_classifier(classnames, templates, model, device):
    with torch.no_grad():
        zeroshot_weights = []
        # for classname in tqdm(classnames):
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            if not zeroshot_weights: print(texts)
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cpu()
    return zeroshot_weights


def fns2pandas(fns):
    """ get features and put them in a pandas dataframe
    for decoding
    """
    info = {"right_shape": [], "right_color": [],
            "left_shape": [], "left_color": [],
            "first_shape": [], "first_color": [],
            "second_shape": [], "second_color": [],
            "relation":[], "sentence": []} 
    for fn in fns:
        fn = fn.lower().replace(".png", "")
        info["first_shape"].append(fn.split(" to the ")[0].split(" ")[2])
        info["first_color"].append(fn.split(" to the ")[0].split(" ")[1])
        info["second_shape"].append(fn.split(" to the ")[1].split(" ")[4])
        info["second_color"].append(fn.split(" to the ")[1].split(" ")[3])
        info["relation"].append(fn.split(" ")[5])
        info["sentence"].append(fn)
        if "right" in fn:
            info["right_shape"].append(fn.split(" to the right of a ")[0].split(" ")[2])
            info["right_color"].append(fn.split(" to the right of a ")[0].split(" ")[1])
            info["left_shape"].append(fn.split(" to the right of a ")[1].split(" ")[1])
            info["left_color"].append(fn.split(" to the right of a ")[1].split(" ")[0])
        else:
            info["left_shape"].append(fn.split(" to the left of a ")[0].split(" ")[2])
            info["left_color"].append(fn.split(" to the left of a ")[0].split(" ")[1])
            info["right_shape"].append(fn.split(" to the left of a ")[1].split(" ")[1])
            info["right_color"].append(fn.split(" to the left of a ")[1].split(" ")[0])
    return pd.DataFrame(info)


def get_queries(colors):
    """ Find all possible queries for decoding
    """
    queries = []
    groups = []
    ## Simple queries
    for place in ["first", "second", "left", "right"]:
        for shape in ["square", "circle", "triangle"]:
            queries.append(f"{place}_shape=='{shape}'")
            groups.append(f"{place}_shape")
        for color in colors:
            queries.append(f"{place}_color=='{color}'")
            groups.append(f"{place}_color")
    ## Joint queries
    for place1 in ["first", "second", "left", "right"]:
        for place2 in ["first", "second", "left", "right"]:
            if place1 in ["first", "second"] and place2 not in ["first", "second"] or \
               place1 in ["left", "right"] and place2 not in ["left", "right"]:
               # not the same type of relation, not interesting
               continue
            for shape in ["square", "circle", "triangle"]:
                for color in colors:
                    queries.append(f"{place1}_shape=='{shape}' and {place2}_color=='{color}'")
                    if place1 == place2: # joint object
                        groups.append(f"{place1}_object")
                    else:
                        groups.append(f"{'-'.join(sorted([place1, place2]))}_not_joint")
            # for shape1 in ["square", "circle", "triangle"]:
            #     for shape2 in ["square", "circle", "triangle"]:
            #         if place1 == place2 and shape1 != shape2: continue # can't have different shapes at the same place
            #         queries.append(f"{place1}_shape=='{shape1}' and {place2}_shape=='{shape2}'")
            #         groups.append(f"{place1}_shape-{place2}_shape")
            # for color1 in ["red", "green", "blue"]:
            #     for color2 in ["red", "green", "blue"]:
            #         if place1 == place2 and color1 != color2: continue # can't have different colors at the same place
            #         queries.append(f"{place1}_color=='{color1}' and {place2}_color=='{color2}'")                
            #         groups.append(f"{place1}_color-{place2}_color")
    return queries, groups


def decode(X, y, pipeline, cv):
    n_folds = cv.get_n_splits(X, y)
    AUC, acc = 0, 0
    for train, test in cv.split(X, y):
        pipeline.fit(X[train, :], y[train])
        pred = pipeline.predict(X[test, :])
        AUC += roc_auc_score(y_true=y[test], y_score=pred) / n_folds
        acc += accuracy_score(y[test], pred>0.5) / n_folds
    return AUC, acc


def get_encoding_features(df, features, ncolors):
    """ Features should be one of "properties", "objects", "sentences"
    """
    all_feats = {}
    df = add_objects_to_df(df, add_first_second=True)
    if "properties" in features:
        all_feats.update(get_properties_feats(df, ncolors))
    if "objects" in features:
        all_feats.update(get_objects_feats(df, ncolors))
    if "sentences" in features:
        all_feats.update(get_sentence_feats(df, ncolors))
    if "sentOrder" in features:
        all_feats.update(get_sent_order_feats(df, ncolors))

    if len(all_feats) == 0: 
        print(f"No feature left, choose at least one of {['properties', 'objects', 'sentences']} ")
    all_feats_names, all_feats_values = [], []
    for name, vals in all_feats.items():
        all_feats_names.append(name)
        all_feats_values.append(vals)
    all_feats_values = np.stack(all_feats_values, axis=-1)
    try:
        assert np.all(np.sum(all_feats_values, 0) > 0)
    except:
        set_trace()
    return all_feats_values, all_feats_names


def add_objects_to_df(df, add_first_second):
    # sides = ['right', 'left', 'first', 'second'] if add_first_second else ['right', 'left']
    # for side in sides:
    #     df[f'{side}_object'] = [f"{df.iloc[i][f'{side}_color']}_{df.iloc[i][f'{side}_shape']}" for i in range(len(df))]
    sides = ['right', 'left', 'first', 'second'] if add_first_second else ['right', 'left']
    for side in sides:
        df[f'{side}_object'] = [f"{df.iloc[i][f'{side}_color']}_{df.iloc[i][f'{side}_shape']}" for i in range(len(df))]
    return df


def get_sentence_feats(df, ncolors):
    colors = ["red", "green", "blue"] + additional_colors
    colors = colors[:ncolors] # keep only correct nb of colors    
    shapes = ["square", "triangle", "circle"]
    orders = ["right", "left"]
    feats = {f"{o}_{c}_{s}": [] for c in colors for s in shapes for o in orders}
    for i, row in df.iterrows():
        for c in colors:
            for s in shapes:
                for o in orders:
                    feats[f"{o}_{c}_{s}"].append(1 if f"{c}_{s}" in row[f'{o}_object'] else 0)
    return feats


def get_sent_order_feats(df, ncolors):
    colors = ["red", "green", "blue"] + additional_colors
    colors = colors[:ncolors] # keep only correct nb of colors    
    shapes = ["square", "triangle", "circle"]
    orders = ["first", "second"]
    feats = {f"{o}_{c}_{s}": [] for c in colors for s in shapes for o in orders}
    for i, row in df.iterrows():
        for c in colors:
            for s in shapes:
                for o in orders:
                    feats[f"{o}_{c}_{s}"].append(1 if f"{c}_{s}" in row[f"{o}_object"] else 0)
    return feats


def get_objects_feats(df, ncolors):
    colors = ["red", "green", "blue"] + additional_colors
    colors = colors[:ncolors] # keep only correct nb of colors    
    shapes = ["square", "triangle", "circle"]
    feats = {f"{c}_{s}": [] for c in colors for s in shapes}
    for i, row in df.iterrows():
        for c in colors:
            for s in shapes:
                feats[f"{c}_{s}"].append(1 if f"{c}_{s}" in row.left_object or f"{c}_{s}" in row.right_object else 0)
    return feats


def get_properties_feats(df, ncolors):
    colors = ["red", "green", "blue"] + additional_colors
    colors = colors[:ncolors] # keep only correct nb of colors    
    shapes = ["square", "triangle", "circle"]
    feats = {w: [] for w in colors+shapes}
    print(feats)
    for i, row in df.iterrows():
        for c in colors:
            feats[c].append(1 if c in row.left_color or c in row.right_color else 0)
        for s in shapes:
            feats[s].append(1 if s in row.left_shape or s in row.right_shape else 0)
    return feats


def encode(X, y, pipeline, cv):
    n_folds = cv.get_n_splits(X, y)
    r2 = 0
    n_units = y.shape[1]
    all_Rs, all_R2s = np.zeros(n_units), np.zeros(n_units)
    all_weights = np.zeros((n_units, X.shape[1]))
    # set_trace()
    for train, test in cv.split(X, y):
        pipeline.fit(X[train, :], y[train])
        pred = pipeline.predict(X[test, :])
        for d in range(n_units):
            all_Rs[d] += pearsonr(y[test, d], pred[:, d])[0] / n_folds
            all_R2s[d] += r2_score(y[test, d], pred[:, d]) / n_folds
        r2 += r2_score(y[test], pred) / n_folds
        all_weights += pipeline[-1].coef_ / n_folds
    return all_Rs, all_R2s, all_weights, r2


def plot_encoding_hierachy(R1, R2, R3, out_fn, sort_with='R1', colors=['r','g', 'b']):
    R3 = np.clip(R3, 0, 1)
    R2 = np.clip(R2, 0, R3)
    R1 = np.clip(R1, 0, R2)
    if sort_with == "R1":
        sorted_indices = np.argsort(R1)
    elif sort_with == "R2":
        sorted_indices = np.argsort(R2)
    elif sort_with == "R3":
        sorted_indices = np.argsort(R3)
    elif sort_with == "R2-R1":
        sorted_indices = np.argsort(R2-R1)
    elif sort_with == "R3-R2":
        sorted_indices = np.argsort(R3-R2)
    R1, R2, R3 = R1[sorted_indices], R2[sorted_indices], R3[sorted_indices]
    fig, ax = plt.subplots(dpi=2000)
    n_units = len(R1)
    plt.bar(range(n_units), R1, color=[colors[0]]*n_units)
    plt.bar(range(n_units), np.clip(R2-R1, 0, 1), bottom=R1, color=[colors[1]]*n_units)
    plt.bar(range(n_units), np.clip(R3-R2, 0, 1), bottom=R2, color=[colors[2]]*n_units)
    
    plt.ylabel("Correlation coefficient")
    plt.xlabel(f"{'Vision' if 'img_embs' in out_fn else 'Language'} CLIP units")
    plt.tight_layout()
    plt.savefig(f"{out_fn}_R_plot_{sort_with}_sorted.png")


def make_legend_encoding(out_fn):
    fig = plt.figure(dpi=200)
    red_patch = mpatches.Patch(color='red', label='Properties')
    green_patch = mpatches.Patch(color='green', label='Objects')
    blue_patch = mpatches.Patch(color='blue', label='Scenes')
    fig.legend(handles=[red_patch, green_patch, blue_patch], loc='center')
    plt.savefig(f'{out_fn}_encoding_legend.png')


def load_embeddings(args):
    # Maybe do an exception for when we load augmented images (one more dimension...)
    embs = np.load(f"{args.root_path}/{args.folder}/{args.version}/{args.in_file}_{args.ncolors}colors.npy")
    print(f"Found embeddings of size: {embs.shape}")
    n_layers, n_trials, seq_length, n_units = embs.shape
    print(f"ie: n_layers: {n_layers}, n_trials: {n_trials}, seq_legnth: {seq_length}, n_units: {n_units}")
    if args.layer == "all": 
        print(f"using units from all layers, ie a total of {n_units*n_layers} dimensions")
        embs = embs.transpose((1,2,3,0)).reshape((n_trials, seq_length, n_units*n_layers))
    else:
        print(f"using units from layer {args.layer}, ie {n_units} dimensions")
        embs = embs[int(args.layer), :, :]
    return embs


def augment_text_with_colors(sents, ncolors=10):
    new_sents = [s for s in sents]
    for color in additional_colors[:ncolors-3]:
        for old_color in ['red', 'blue', 'green']:
            for sent in sents:
                new_sent = sent.replace(old_color, color)
                new_sents.append(new_sent)
    return new_sents


def plot_all_pca_trajectory(trajs, out_fn, colors=[], labels=[], xlabels='[BoS] a [color1] [shape1] to the [side] of a [color2] [shape2] [EoS]'):
    if colors: assert len(colors) == len(trajs)
    if labels: assert len(labels) == len(trajs)
    # set_trace()
    # trajs is n_trials * n_times * n_PCs
    trajs = trajs.transpose((2,0,1)) # n_PC * n_trials * n_times
    n_times = trajs.shape[2]
    n_col = len(trajs) // 8
    n_row = 8
    fig, axes = plt.subplots(n_row, n_col, sharex='col', dpi=400, figsize=(12,8))
    for i, traj in enumerate(trajs):
        col = i // n_row
        row = i % n_row
        for i_c, cond in enumerate(traj):
            if colors:
                if labels:
                    axes[row, col].plot(range(1,len(cond)+1), cond, colors[i_c], label=labels[i_c])
                else:
                    axes[row, col].plot(range(1,len(cond)+1), cond, colors[i_c])
            else:
                if labels: 
                    axes[row, col].plot(range(1,len(cond)+1), cond, label=labels[i_c])
                else:
                    axes[row, col].plot(range(1,len(cond)+1), cond)
            
        if i == len(trajs)-1 and labels:
            plt.legend(loc='best')
        if row == n_row-1:
            axes[row, col].set_xticks(np.arange(1, len(xlabels.split())+1))
            axes[row, col].set_xticklabels(xlabels.split(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_fn)
    

def plot_hist_weights(weights, indices, labels, out_fn):
    fig, ax = plt.subplots(dpi=500)
    offset = .75 / len(labels)
    start =offset * len(labels) / 2
    x = np.arange(len(labels))
    for i, (w, lab) in enumerate(zip(weights[indices], labels)):
        try:
            ax.bar(x-start+(i*offset), w, width=offset, align='center', label=lab)
        except:
            set_trace()
    ax.set_xticks(x)
    xlabels = [f"unit {nb}" for nb in indices]
    ax.set_xticklabels(xlabels, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, .99), ncol=1, fancybox=True, shadow=True, prop={'size': 8})
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(out_fn)
    plt.close(fig)


