#!/usr/bin/env python
from ipdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from glob import glob
from pathlib import Path
# import statsmodels
# import statsmodels.formula.api as smf
# import statsmodels.api as sm
import seaborn as sns
from statannotations.Annotator import Annotator


out_dir = Path("../results/behavior/")
model_behavior_path = Path("./model_results.csv")
model_beh = pd.read_csv(model_behavior_path)

# human_behavior_path = Path("./human_results.csv")
# human_beh = pd.read_csv(human_behavior_path)

# def get_sent_difficulty(sent):
# 	s1, s2 = sent.split()[2], sent.split()[-1]
# 	c1, c2 = sent.split()[1], sent.split()[-2]
# 	d = 2
# 	if s1 == s2: d -= 1
# 	if c1 == c2: d -= 1
# 	return d
# model_beh["Difficulty"] = [get_sent_difficulty(s) for s in model_beh.sentence]

def get_feats(sent):
	s1, s2 = sent.split()[2], sent.split()[-1]
	c1, c2 = sent.split()[1], sent.split()[-2]
	rel = sent.split()[4]
	d = 2
	if s1 == s2: d -= 1
	if c1 == c2: d -= 1
	return s1, s2, c1, c2, rel, f"D{d}"

S1, S2, C1, C2, REL, D = [], [], [], [], [], []
for sent in model_beh.sentence:
	s1, s2, c1, c2, rel, d = get_feats(sent)
	S1.append(s1)
	S2.append(s2)
	C1.append(c1)
	C2.append(c2)
	REL.append(rel)
	D.append(d)
model_beh["Shape1"] = S1
model_beh["Shape2"] = S2
model_beh["Colour1"] = C1
model_beh["Colour2"] = C2
model_beh["Relation"] = REL
model_beh["Difficulty"] = D


def make_sns_barplot(df, x, y, hue=None, box_pairs=[], kind='bar', out_fn="tmp.png", ymin=None, ymax=None, 
                     hline=None, rotate_ticks=False, tight=False, ncol=1, order=None, hue_order=None):
    sns.set(font_scale = 2)
    if kind=="box":
        g = sns.catplot(x=x, y=y, hue=hue, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False, showfliers=False) #, order=order, hue_order=hue_order) # ci=68 <=> standard error
    else:
        g = sns.catplot(x=x, y=y, hue=hue, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False) #, order=order, hue_order=hue_order) # ci=68 <=> standard error
    # if ymin is not None or ymax is not None: ## SHOULD BE BEFORE CALLING ANNOTATOR!!
    #     g.set(ylim=(ymin, ymax))
    g.ax.set_xlabel(x,fontsize=20)
    g.ax.set_ylabel(y, fontsize=20)
    g.ax.tick_params(axis='both', which='major', labelsize=14)
    if hline is not None:
        g.ax.axhline(y=hline, lw=1, ls='--', c='grey', zorder=-10)
    if rotate_ticks:
        for tick in g.ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
    
    if box_pairs and kind != "point":
        annotator = Annotator(g.ax, box_pairs, plot=f'{kind}plot', data=df, x=x, hue=hue, y=y, text_format='star', order=order, hue_order=hue_order) #,), line_offset_to_box=-1
        annotator.configure(test='Mann-Whitney', verbose=False, loc="outside", comparisons_correction="bonferroni", fontsize=12, use_fixed_offset=True).apply_and_annotate()
        # , line_offset=.0001, line_offset_to_group=.0001
    if tight:
        plt.tight_layout()
    if ncol > 1 or hue is not None:
        g.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol, fontsize=12) # Put the legend out of the figure
    plt.savefig(out_fn, transparent=True, bbox_inches='tight', dpi=400)
    plt.close()


# ### COLOUR ###
box_pairs = [("green", "red"), ("red", "blue"), ("green", "blue")]
ymin, ymax = model_beh.perf.mean()-.05, model_beh.perf.mean()+.05
make_sns_barplot(model_beh, x='Colour1', y='perf', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_color1_perf.png', tight=True)

ymin, ymax = model_beh.perf.mean()-.05, model_beh.perf.mean()+.05
make_sns_barplot(model_beh, x='Colour2', y='perf', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_color2_perf.png', tight=True)

### SHAPE ###
box_pairs = [("square", "circle"), ("circle", "triangle"), ("square", "triangle")]
ymin, ymax = model_beh.perf.mean()-.05, model_beh.perf.mean()+.05
make_sns_barplot(model_beh, x='Shape1', y='perf', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_shape1_perf.png', tight=True)

ymin, ymax = model_beh.perf.mean()-.05, model_beh.perf.mean()+.05
make_sns_barplot(model_beh, x='Shape2', y='perf', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_2obj_shape2_perf.png', tight=True)


### DIFFICULTY ###
box_pairs = [("D0", "D1"), ("D1", "D2"), ("D0", "D2")]
ymin, ymax = model_beh.perf.mean()-.05, model_beh.perf.mean()+.05
make_sns_barplot(model_beh, x='Difficulty', y='perf', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_difficulty_perf.png', tight=True, order=["D0", "D1", "D2"])
make_sns_barplot(model_beh, x='Difficulty', y='perf', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_difficulty_perf.png', tight=True, order=["D0", "D1", "D2"])
make_sns_barplot(model_beh, x='Difficulty', y='perf', kind='violin', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_difficulty_perf.png', tight=True, order=["D0", "D1", "D2"])



# ## With interaction
# ### DIFFICULTY ###
# box_pairs = [(("D0", "None"), ("D0", "l0")), 
# 			 (("D1", "None"), ("D1", "l0")),(("D1", "l0"), ("D1", "l2")),
# 			 (("D2", "None"), ("D2", "l0")),(("D2", "l0"), ("D2", "l1")),(("D2", "l1"), ("D2", "l2")),(("D2", "l0"), ("D2", "l2"))] 
# ymin, ymax = model_beh.perf.mean()-.1, model_beh.perf.mean()+.05
# make_sns_barplot(model_beh, x='Difficulty', y='perf', hue='Error_type', box_pairs=box_pairs, ymin=ymin, ymax=ymax, out_fn=f'{out_dir}/stat_bar_2obj_difficulty_error_type_perf.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])
# make_sns_barplot(model_beh, x='Difficulty', y='perf', hue='Error_type', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_difficulty_error_type_perf1.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])
# make_sns_barplot(model_beh, x='Difficulty', y='perf', hue='Error_type', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_difficulty_error_type_perf1.png', tight=True, order=["D0", "D1", "D2"], hue_order=["None", "l0", "l1", "l2"])

