#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:33:57 2021

@author: yl254115
"""

import os
import argparse
from Stimuli import Stimuli

##########
# SCENCE #
##########
n_nouns = 2
relations = ['is left to', 'is right to']

# FEATURES
features = {}
features['head'] = {}
features['modifier'] = {}
features['head']['shape'] = ['triangle', 'circle', 'square']
features['modifier']['color'] = ['blue', 'red', 'green']
features['modifier']['size'] = ['small', 'big']


####################
# GENERATE STIMULI #
####################
for modifiers in [None, '', 'color', 'size',
                      'color_size', 'size_color']:
    print(f'Limit scene to modifiers: {modifiers}')
    # If modifiers is not None, then scenes will be composed of nouns,
    # which have modifiers as specificed by modifiers.
    # If modifiers is an empty list '', then nouns will be composed of heads
    # only.
    # If modifiers is None,
    # then all possible combinations of nouns are generated
    if modifiers is not None: 
        str_modifiers = modifiers
    else: # All combinations
        str_modifiers = 'all_combinations'
    fn = f'../stimuli/text/modifiers_{str_modifiers}.txt'
    
    # Init
    stimuli = Stimuli(n_nouns, features, relations, verbose=False)
    
    # NOUNS
    stimuli.add_nouns()
    
    if modifiers is None:
        nouns = []
        [nouns.extend(stimuli.nouns[key]) for key in stimuli.nouns.keys()]
    else:
        nouns = stimuli.nouns[modifiers]
    n_nouns = len(nouns)
    print(f'Total number of nouns: {n_nouns}\n{nouns}')
    # print('-'*100)
    
    # SCENES
    stimuli.add_scenes(modifiers=modifiers)
    stimuli.to_txt(fn)
    print(f'Total number of scenes: {len(stimuli.scenes)}')
    print(f'Stimuli saved to: {fn}')
    print('-'*100)