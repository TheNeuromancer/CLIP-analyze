
#!/usr/bin/env python
from ipdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import statsmodels
from glob import glob
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
import argparse
import seaborn as sns
from statannotations.Annotator import Annotator

parser = argparse.ArgumentParser(description='extract text embeddings from CLIP for the original MEG stimuli')
parser.add_argument('-r', '--root-path', default='/Users/tdesbordes/Documents/CLIP-analyze/', help='root path')
parser.add_argument('-d', '--results-dir', default='hug_v3', help='directry to get results and store figures')
parser.add_argument('-w', '--overwrite', action='store_true', default=False, help='whether to overwrite the output directory or not')
args = parser.parse_args()


def make_sns_barplot(df, x, y, hue=None, box_pairs=[], kind='point', col=None, out_fn="tmp.png", ymin=None, ymax=None, 
     hline=None, rotate_ticks=False, tight=True, ncol=1, order=None, hue_order=None, legend=True, dodge=True, jitter=True, colors=None):
    sns.set(font_scale = 2)
    if kind=="box":
        g = sns.catplot(x=x, y=y, hue=hue, col=col, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False, palette=colors, showfliers=False)# ci=68 <=> standard error
    elif kind=="point":
        g = sns.catplot(x=x, y=y, hue=hue, col=col, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False, palette=colors, jitter=jitter, dodge=dodge)# ci=68 <=> standard error
    else:
        g = sns.catplot(x=x, y=y, hue=hue, col=col, data=df, kind=kind, order=order, hue_order=hue_order, ci=68, legend=False, palette=colors)# ci=68 <=> standard error
    if ymin is not None or ymax is not None: ## SHOULD BE BEFORE CALLING ANNOTATOR!!
        g.set(ylim=(ymin, ymax))
    axes = g.axes[0] if col is not None else [g.ax]
    for ax in axes:
        ax.set_xlabel(x,fontsize=20)
        ax.set_ylabel(y, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if hline is not None:
            ax.axhline(y=hline, lw=1, ls='--', c='k', zorder=-10)
        if rotate_ticks:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')
        
        if box_pairs and kind != "point":
            annotator = Annotator(ax, box_pairs, plot=f'{kind}plot', data=df, x=x, hue=hue, col=col, y=y, text_format='star', order=order, hue_order=hue_order) #,), line_offset_to_box=-1
            annotator.configure(test='Mann-Whitney', verbose=False, loc="inside", comparisons_correction="bonferroni", fontsize=12, use_fixed_offset=True).apply_and_annotate()
            # , line_offset=.0001, line_offset_to_group=.0001

        if tight:
            plt.tight_layout()
        if (ncol > 1 or hue is not None) and legend:
            ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=ncol, fontsize=12) # Put the legend out of the figure
    plt.savefig(out_fn, transparent=True, bbox_inches='tight', dpi=400)
    plt.close()


out_dir = Path(f"{args.root_path}/figures/{args.results_dir}/")
out_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path(f"{args.root_path}/results/behavior/{args.results_dir}/")
path_1obj = list(results_dir.glob("*-1obj.csv"))
print(path_1obj)
results_1obj = pd.read_csv(path_1obj[0], encoding='ISO-8859-1')

path_2obj = list(results_dir.glob("*-2obj.csv"))
print(path_2obj)
results_2obj = pd.read_csv(path_2obj[0], encoding='ISO-8859-1')
print("loaded")

# concatenate 1obj and 2obj trials
results = pd.concat((results_1obj, results_2obj))
results.reset_index(inplace=True, drop=True) # reset index

# remove non-entries
# results = results.dropna()
results = results.query("sentences != 'None'")
# put Perf as type float 
results.Perf = results.Perf.apply(float)

# get back marginal after rejection and re-indexing
results_1obj = deepcopy(results.query("NbObjects==1"))
results_2obj = deepcopy(results.query("NbObjects==2"))

# results_2obj["Violation ordinal position"] = results_2obj["property_mismatches_positions"].apply(lambda x: "First" if x < 3 else "Second")
results_2obj["Violation ordinal position"] = results_2obj["property_mismatches_positions"]
results_2obj["Violation side"] = results_2obj["property_mismatches_side"]
results_2obj["# Shared features"] = results_2obj["# Shared features"].astype(int)
results_2obj["Violation ordinal position"] = results_2obj["property_mismatches_order"] 

print("starting")
box_pairs = []
# set_trace()

default_colors = sns.color_palette()

##################
#### OVERALL #####
##################
## Error rate are intersting, but similarity between unvilated sentences is not informative. 
make_sns_barplot(deepcopy(results).query("Violation=='No'"), x='Trial type', y='Error rate', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_errorrate_point_dodge.png')
# make_sns_barplot(deepcopy(results).query("Violation=='No'"), x='Trial type', y='similarity', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_overall_similarity_point_dodge.png')

# split by violation
make_sns_barplot(deepcopy(results).query("Violation=='No'"), x='Trial type', y='Error rate', dodge=.02, col='Violation', kind='point', out_fn=f'{out_dir}/stat_overall*violation_errorrate_point_dodge.png')
# make_sns_barplot(deepcopy(results).query("Violation=='No'"), x='Trial type', y='similarity', dodge=.02, col='Violation', kind='point', out_fn=f'{out_dir}/stat_overall*violation_similarity_point_dodge.png')


##################
###### EASY ######
##################

# ### COLOUR ###
# # box_pairs = [("green", "red"), ("red", "blue"), ("green", "blue")]
# make_sns_barplot(results_1obj, x='color1', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_errorrate_point.png')
# make_sns_barplot(results_1obj, x='color1', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_errorrate_box.png')

# make_sns_barplot(results_1obj, x='color1', y='similarity', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_similarity_point.png')
# make_sns_barplot(results_1obj, x='color1', y='similarity', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_color_similarity_box.png')

# ### SHAPE ###
# # box_pairs = [("square", "circle"), ("circle", "triangle"), ("square", "triangle")]
# make_sns_barplot(deepcopy(results_1obj).query("Violation=='No'"), x='shape1', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_errorrate_point.png')
# make_sns_barplot(deepcopy(results_1obj).query("Violation=='No'"), x='shape1', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_errorrate_box.png')

# make_sns_barplot(deepcopy(results_1obj).query("Violation=='No'"), x='shape1', y='similarity', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_similarity_point.png')
# make_sns_barplot(deepcopy(results_1obj).query("Violation=='No'"), x='shape1', y='similarity', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_shape_similarity_box.png')

# ### SHAPE * COLOR ###
# make_sns_barplot(deepcopy(results_1obj).query("Violation=='No'"), x='shape1', hue='color1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_1obj_shape*color_errorrate_point.png', order=['square', 'triangle', 'circle'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})
# make_sns_barplot(deepcopy(results_1obj).query("Violation=='No'"), x='shape1', hue='color1', y='similarity', kind='point', out_fn=f'{out_dir}/stat_1obj_shape*color_similarity_point.png', order=['square', 'triangle', 'circle'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})

# ### VIOLATION ###
# box_pairs = [("No", "Yes")]
# make_sns_barplot(results_1obj, x='Violation', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_errorrate_point.png', order=["No", "Yes"])
# # make_sns_barplot(results_1obj, x='Violation', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_errorrate_box.png', order=["No", "Yes"])

# # make_sns_barplot(results_1obj, x='Violation', y='similarity', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_similarity.png', order=["No", "Yes"])
# make_sns_barplot(results_1obj, x='Violation', y='similarity', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_similarity_point.png', order=["No", "Yes"])
# # make_sns_barplot(results_1obj, x='Violation', y='similarity', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_1obj_violation_similarity_box.png', order=["No", "Yes"])

# ### VIOLATION ON ### 
# box_pairs = []
# make_sns_barplot(results_1obj, x='Violation on', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_1obj_violon_errorrate_point.png', order=["Shape", "Color"])
# make_sns_barplot(results_1obj, x='Violation on', y='similarity', kind='point', out_fn=f'{out_dir}/stat_1obj_violon_similarity_point.png', order=["Shape", "Color"])

# ### SHAPE * VIOLATION ON ###
# make_sns_barplot(results_1obj, x='shape1', y='Error rate', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1objshape*violon_errorrate_point.png', order=['square', 'triangle', 'circle'], hue_order=["Shape", "Color"])
# make_sns_barplot(results_1obj, x='shape1', y='similarity', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1obj_shape*violon_similarity_point.png', order=['square', 'triangle', 'circle'], hue_order=["Shape", "Color"])

# ### COLOR * VIOLATION ON ###
# make_sns_barplot(results_1obj, x='color1', y='Error rate', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1objcolor*violon_errorrate_point.png', order=['red', 'green', 'blue'], hue_order=["Shape", "Color"])
# make_sns_barplot(results_1obj, x='color1', y='similarity', kind='point', hue='Violation on', out_fn=f'{out_dir}/stat_1obj_color*violon_similarity_point.png', order=['red', 'green', 'blue'], hue_order=["Shape", "Color"])


# #################
# ##### HARD ######
# #################

### COLOUR ###
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='color1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_color1_errorrate_point.png')
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='color2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_color2_errorrate_point.png')

make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='color1', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_color1_similarity_point.png')
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='color2', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_color2_similarity_point.png')

### SHAPE ###
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1_errorrate_point.png')
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2_errorrate_point.png')

make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape1', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1_similarity_point.png')
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape2', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2_similarity_point.png')

### SHAPE * COLOR ###
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape1', hue='color1', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*color1_errorrate_point.png', order=['square', 'triangle', 'circle'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape1', hue='color1', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*color1_similarity_point.png', order=['square', 'triangle', 'circle'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})

make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape2', hue='color2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2*color2_errorrate_point.png', order=['square', 'triangle', 'circle'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape2', hue='color2', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_shape2*color2_similarity_point.png', order=['square', 'triangle', 'circle'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})

### SHAPE * SHAPE ####
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape1', hue='shape2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*shape2_errorrate_point.png', order=['square', 'triangle', 'circle'], hue_order=['square', 'triangle', 'circle'])
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='shape1', hue='shape2', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_shape1*shape2_similarity_point.png', order=['square', 'triangle', 'circle'], hue_order=['square', 'triangle', 'circle'])

### COLOR * COLOR
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='color1', hue='color2', y='Error rate', kind='point', out_fn=f'{out_dir}/stat_2obj_color1*color2_errorrate_point.png', order=['red', 'green', 'blue'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x='color1', hue='color2', y='similarity', kind='point', out_fn=f'{out_dir}/stat_2obj_color1*color2_similarity_point.png', order=['red', 'green', 'blue'], hue_order=['red', 'green', 'blue'], colors={"red":(1,0,0), "green":(0,1,0), "blue":(0,0,1)})

# ### RELATION ###
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x="relation", y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_relation_errorrate_point.png', order=["à droite d'", "à gauche d'"])
make_sns_barplot(deepcopy(results_2obj).query("Violation=='No'"), x="relation", y='similarity', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_relation_similarity_point.png', order=["à droite d'", "à gauche d'"])

# ### RELATION * VIOLATION ###
make_sns_barplot(results_2obj, x="relation", y='Error rate', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_2obj_relation*violation_errorrate_point.png', order=["à droite d'", "à gauche d'"], hue_order=["No", "Yes"])
make_sns_barplot(results_2obj, x="relation", y='similarity', kind='point', hue='Violation', out_fn=f'{out_dir}/stat_2obj_relation*violation_similarity_point.png', order=["à droite d'", "à gauche d'"], hue_order=["No", "Yes"])


### VIOLATION ###
box_pairs = [("No", "Yes")]
make_sns_barplot(results_2obj, x='Violation', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_errorrate_point.png', order=["No", "Yes"])
make_sns_barplot(results_2obj, x='Violation', y='Error rate', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_errorrate_box.png', order=["No", "Yes"])

make_sns_barplot(results_2obj, x='Violation', y='similarity', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_similarity.png', order=["No", "Yes"])
make_sns_barplot(results_2obj, x='Violation', y='similarity', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_similarity_point.png', order=["No", "Yes"])
make_sns_barplot(results_2obj, x='Violation', y='similarity', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_2obj_violation_similarity_box.png', order=["No", "Yes"])

# ## SHARED FEATURES IN 2 OBJ TRIALS
make_sns_barplot(results_2obj, x='Sharing', y='Error rate', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_errorrate_point_dodge.png', order=["Both", "Shape", "Color", "None"])
make_sns_barplot(deepcopy(results_2obj).query("Violation=='Yes'"), x='Sharing', y='similarity', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_similarity_point_dodge.png', order=["Both", "Shape", "Color", "None"])

make_sns_barplot(results_2obj, x='# Shared features', y='similarity', hue='Violation', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_by_violation_similarity_point_dodge.png', order=[2, 1, 0], hue_order=["No", "Yes"])

make_sns_barplot(results_2obj, x='Sharing', y='similarity', hue='Violation', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_by_violation_similarity_point_dodge.png', order=["Both", "Shape", "Color", "None"], hue_order=["No", "Yes"])


### ERROR_TYPE ### 
# remove non violated for error rate because error rates are not comparable (perf for correct is relative to all types of errors, whereas for eaach type of error it is relative to the correct sent)
make_sns_barplot(deepcopy(results_2obj).query("Violation=='Yes'"), x='Violation on', y='Error rate', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violon_errorrate.png', order=["None", "property", "binding", "relation"])
make_sns_barplot(results_2obj, x='Violation on', y='similarity', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violon on_similarity.png', order=["None", "property", "binding", "relation"])

make_sns_barplot(deepcopy(results_2obj).query("Violation=='Yes'"), x='Violation on', y='Error rate', hue='Sharing', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_violon_by_sharing_errorrate_point_dodge.png', order=["None", "property", "binding", "relation"], hue_order=["Both", "Shape", "Color", "None"])
make_sns_barplot(results_2obj, x='Violation on', y='similarity', hue='Sharing', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_violon_by_sharing_similarity_point_dodge.png', order=["None", "property", "binding", "relation"], hue_order=["Both", "Shape", "Color", "None"])


## SHARED FEATURES BY ERROR TYPE
make_sns_barplot(deepcopy(results_2obj).query("Violation=='Yes'"), x='# Shared features', y='Error rate', hue='Violation on', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_by_Violon_errorrate_point_dodge.png', order=[2, 1, 0], hue_order=["None", "property", "binding", "relation"])
make_sns_barplot(results_2obj, x='# Shared features', y='similarity', hue='Violation on', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_#shared_by_Violon_similarity_point_dodge.png', order=[2, 1, 0], hue_order=["property", "binding", "relation"])

## SHARING BY ERROR TYPE
make_sns_barplot(results_2obj, x='Sharing', y='Error rate', hue='Violation on', hline=.5, dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_by_Violon_errorrate_point_dodge.png', order=["Both", "Shape", "Color", "None"], hue_order=["property", "binding", "relation"], colors={"property": default_colors[1], "binding": default_colors[2], "relation": default_colors[3]})
make_sns_barplot(results_2obj, x='Sharing', y='similarity', hue='Violation on', dodge=.2, kind='point', out_fn=f'{out_dir}/stat_sharing_by_Violon_similarity_point_dodge.png', order=["Both", "Shape", "Color", "None"], hue_order=["None", "property", "binding", "relation"])


# ### VIOLATION POSITION ###
# Violated_position = 3 <-> L2 (relation inversion) error 
# Violated_position in [1,2,4,5] <-> L0 (dumb change) error
# Violated_position in [1.5, 4.5] <-> L1 (binding) error
if True:
	local_results_2obj = deepcopy(results_2obj).query('Error_type=="l0"')
	
	# Spatial positon
	make_sns_barplot(local_results_2obj, x='Violation side', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_side_error_rate.png', order=["Left", "Right"])
	make_sns_barplot(local_results_2obj, x='Violation side', y='similarity', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_side_similarity.png', order=["Left", "Right"])

	# ordinal positon
	make_sns_barplot(local_results_2obj, x='Violation ordinal position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_ordinal_position_error_rate.png', order=["First", "Second"])
	make_sns_barplot(local_results_2obj, x='Violation ordinal position', y='similarity', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violation_ordinal_position_similarity.png', order=["First", "Second"])	


	# # Just position
	# # box_pairs = [(1.0, 2.0), (1.0, 4.0), (1.0, 5.0)] #, (2.0, 4.0), (2.0, 5.0), (4.0, 5.0)]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position_similarity.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position_similarity.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position_similarity.png')

	# # position*relation
	# # box_pairs = [((1.0, "à gauche d'"), (1.0, "à droite d'")), ((1.0, "à gauche d'"), (2.0, "à gauche d'")), ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation_similarity.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation_similarity.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation_similarity.png')

	# # split by difficulty
	# # box_pairs = []
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='# Shared features', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*#Shared_features_similarity.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='# Shared features', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*#Shared_features_similarity.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='# Shared features', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*#Shared_features_similarity.png')



	# ### SAME WITH PERF
	# # box_pairs = [(1.0, 2.0), (1.0, 4.0), (1.0, 5.0)] #, (2.0, 4.0), (2.0, 5.0), (4.0, 5.0)]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position_error_rate.png')

	# # position*relation
	# # box_pairs = [((1.0, "à gauche d'"), (1.0, "à droite d'")), ((1.0, "à gauche d'"), (2.0, "à gauche d'")), ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*Relation_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*Relation_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*Relation_error_rate.png')

	# # split by difficulty
	# # box_pairs = []
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='# Shared features', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_Violated_position*#_Shared_features_error_rate.png')
	# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='# Shared features', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_Violated_position*#_Shared_features_error_rate.png')
	# # make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='# Shared features', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_Violated_position*#_Shared_features_error_rate.png')




# ## L1 trials
# local_results_2obj = deepcopy(results_2obj).query('Error_type=="l1"')
# box_pairs = [(1.5, 4.5)]
# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position_similarity.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position_similarity.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position_similarity.png')

# # position*relation
# box_pairs = []
# box_pairs = [((1.5, "à gauche d'"), (1.5, "à droite d'")), ((4.5, "à gauche d'"), (4.5, "à gauche d'"))] #, ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Relation_similarity.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Relation_similarity.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='similarity', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Relation_similarity.png')

# ## L1 trials -- Error rate
# local_results_2obj = deepcopy(results_2obj).query('Error_type=="l1"')
# box_pairs = [(1.5, 4.5)]
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue=None, kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position_error_rate.png')

# # position*relation
# box_pairs = []
# box_pairs = [((1.5, "à gauche d'"), (1.5, "à droite d'")), ((4.5, "à gauche d'"), (4.5, "à gauche d'"))] #, ((1.0, "à gauche d'"), (4.0, "à gauche d'"))]
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_bar_2obj_L1_Violated_position*Relation_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='point', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_point_2obj_L1_Violated_position*Relation_error_rate.png')
# make_sns_barplot(local_results_2obj, x='Violated_position', y='Error rate', hue='Relation', kind='box', box_pairs=box_pairs, out_fn=f'{out_dir}/stat_box_2obj_L1_Violated_position*Relation_error_rate.png')


## stats
# # packnames = ('lme4', 'lmerTest', 'emmeans', "geepack")
# # from rpy2.robjects.packages import importr
# # from rpy2.robjects.vectors import StrVector
# # utils = importr("utils")
# # utils.chooseCRANmirror(ind=1)
# # utils.install_packages(StrVector(packnames))


# # # #Import necessary packages
# # # from rpy2.robjects.packages import importr
# # import rpy2.robjects as robjects
# # from rpy2.robjects import pandas2ri
# # #Must be activated
# # pandas2ri.activate()
# # from rpy2.robjects import FloatVector
# # from rpy2.robjects.packages import importr

# # stats = importr('stats')
# # base = importr('base')
# # lme4 = importr('lme4')

# # # ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
# # # trt = FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])
# # # group = base.gl(2, 10, 20, labels = ['Ctl','Trt'])
# # # weight = ctl + trt
# # # robjects.globalenv['color1'] = results_1obj.color1
# # # robjects.globalenv['shape1'] = results_1obj.shape1

# # results_1obj["Matching"][results_1obj["Matching"]=="match"] = 1
# # results_1obj["Matching"][results_1obj["Matching"]=="nonmatch"] = -1
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_left"] = 1
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_right"] = -1
# # robjects.globalenv['Matching'] = results_1obj.Matching
# # robjects.globalenv['color1'] = results_1obj.color1
# # robjects.globalenv['shape1'] = results_1obj.shape1

# # results_1obj["red"] = pd.get_dummies(results_1obj.color1).iloc[:,0]
# # results_1obj["blue"] = pd.get_dummies(results_1obj.color1).iloc[:,1]
# # results_1obj["green"] = pd.get_dummies(results_1obj.color1).iloc[:,2]
# # robjects.globalenv['red'] = pd.get_dummies(results_1obj.color1).iloc[:,0]
# # robjects.globalenv['green'] = results_1obj.blue
# # robjects.globalenv['blue'] = results_1obj.green
# # # robjects.globalenv['Matching'] = (results_1obj.Matching == "nonmatch").values.astype(int)
# # robjects.globalenv['similarity'] = results_1obj.similarity
# # robjects.globalenv['Subject'] = results_1obj.Subject
# # lm = lme4.lmer('similarity ~ Matching + red + green + blue + shape1 + (1|Subject)')
# # print(base.summary(lm))



# # results_1obj["red"] = pd.get_dummies(results_1obj.color1).iloc[:,0]
# # results_1obj["blue"] = pd.get_dummies(results_1obj.color1).iloc[:,1]
# # results_1obj["green"] = pd.get_dummies(results_1obj.color1).iloc[:,2]
# # res = smf.mixedlm("similarity ~ 1 + red + blue + green", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs")
# # print(res.summary())


# # results_1obj["Matching"][results_1obj["Matching"]=="match"] = .5
# # results_1obj["Matching"][results_1obj["Matching"]=="nonmatch"] = -.5
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_left"] = .5
# # results_1obj["Mapping"][results_1obj["Mapping"]=="correct_right"] = -.5

# ## Stats single object
# smf.mixedlm("similarity ~ Matching + Mapping + color1 + shape1", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("Perf ~ Matching + Mapping + color1 + shape1", results_1obj, groups=results_1obj["Subject"]).fit(method="bfgs").summary()

# ## Stats two objects
# # smf.mixedlm("similarity ~ Matching + Mapping + color1 + shape1 + color2 + shape2 + Relation", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("similarity ~ Matching + Mapping + color1 + shape1 + color2 + shape2 + Relation + Difficulty", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# smf.mixedlm("Perf ~ Matching + Mapping + color1 + shape1 + color2 + shape2 + Relation + Difficulty", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()


# smf.mixedlm("similarity ~ ordered(Difficulty)  ", results_2obj, groups=results_2obj["Subject"]).fit(method="bfgs").summary()
# set_trace()

# # Error types
# # violated_results_2obj = results_2obj[results_2obj["Error_type"] != "None"]
# # # violated_results_2obj["L0"] = pd.get_dummies(violated_results_2obj.Error_type).iloc[:,0]
# # # violated_results_2obj["L1"] = pd.get_dummies(violated_results_2obj.Error_type).iloc[:,1]
# # # violated_results_2obj["L2"] = pd.get_dummies(violated_results_2obj.Error_type).iloc[:,2]
# # smf.mixedlm("similarity ~ Difficulty + Error_type ", violated_results_2obj, groups=violated_results_2obj["Subject"]).fit(method="bfgs").summary()

# # Singular matrx error
# # smf.mixedlm("similarity ~ Difficulty + Error_type + Difficulty*Error_type ", violated_results_2obj, groups=violated_results_2obj["Subject"]).fit(method="bfgs").summary()
# # set_trace()

