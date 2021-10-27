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


fn = '../stimuli/text/all.txt'

####################
# GENERATE STIMULI #
####################
stimuli = Stimuli(n_nouns, features, relations, verbose=True)
stimuli.add_nouns()
stimuli.add_scenes()
stimuli.to_txt(fn)