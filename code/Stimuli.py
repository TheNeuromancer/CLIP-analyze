#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:33:57 2021

@author: yl254115
"""

import os
import argparse
import itertools


class Stimuli():
    def __init__(self, n_nouns, features, relations, verbose=True):
        '''
        

        Parameters
        ----------
        n_nouns : int
            Number of nouns in a sentence
        features : dict of features
            E.g., 
            features['head']['shape'] = ['triangle', 'circle']
            features['modifier']['color'] = ['blue', 'red', 'green']
            features['modifier']['size'] = ['small', 'big']
            
        relations : list of strings
            E.g., ['left to', 'right to', 'above']

        Returns
        -------
        None.

        '''
        self.n_nouns = n_nouns
        self.features = features
        self.relations = relations
        
        if n_nouns > 1:
            assert relations
        
        # features dict must have a single head
        assert ('head' in features.keys()) and len(features['head'].keys())==1
    
        self.verbose = verbose


    def add_nouns(self):
        '''
        

        Returns
        -------
        Append single nouns to class objects.
        Scenes could be then contructed based on these nouns

        '''
        head_feature = list(self.features['head'].keys())[0] # E.g. shape
        modifiers = list(self.features['modifier'].keys())
        n_modifiers = len(modifiers)
        powerset_modifiers = list(powerset(modifiers))
        
        # Loop over all possible subsets of modifier list, e.g.,
        # [(), ('color',), ('size',), ('color', 'size')]
        nouns = {} # 
        for set_modifiers in powerset_modifiers: 
            # For a given subset of modifiers, e.g., ('color', 'size'),
            # loop over all possible orders
            # E.g., [('color', 'size'), ('size', 'color')]
            for ordered_list_modifiers in list(itertools.permutations(set_modifiers)):
                # Get speciic values of each modifier, e.g.,
                # For color: 'red', 'green', 'blue'.
                ordered_modifier_values = [] # list of lists
                for curr_modifier in ordered_list_modifiers:
                    # Each sublist contains all modifier values, e.g.
                    # [['blue', 'red', 'green'], ['small', 'big']])
                    ordered_modifier_values.append(self.features['modifier'][curr_modifier])
                # All possible combinations of modifier values, e.g.,
                # [('blue', 'small'), ('blue', 'big'), ('red', 'small'), ...]
                modifier_value_combinations = list(itertools.product(*ordered_modifier_values))
            
            
        
            # Loop over all head values
            # E.g., 'A blue small circle', 'A blue small triangle', ...
            nouns['_'.join(set_modifiers)] = []
            for head_value in self.features['head'][head_feature]:
                for modifier_value_combination in modifier_value_combinations:
                    str_modifiers = ' '.join(modifier_value_combination)                
                    if str_modifiers:
                        noun = f'A {str_modifiers} {head_value}'
                    else:
                        noun = f'A {head_value}'
                    nouns['_'.join(set_modifiers)].append(noun)
                    if self.verbose:
                        print(noun)
        self.nouns = nouns
            
                
    def add_scenes(self):
        '''
        

        Returns
        -------
        Apped scenes to class object.

        '''
        scenes = []
        
        
        modifiers = list(self.features['modifier'].keys())
        powerset_modifiers = list(powerset(modifiers))
        
        for set_modifiers in powerset_modifiers: 
            nouns = self.nouns['_'.join(set_modifiers)]
            nouns_in_scence = list(itertools.product(nouns, repeat=2))
            for noun_pair in nouns_in_scence:
                for relation in self.relations:
                    scene = f"{noun_pair[0]} {relation} {noun_pair[1]}".lower()
                    if self.verbose:
                        print(scene)
                    scenes.append(scene)
        
        
        self.scenes = scenes
        
    
    def to_txt(self, fn):
        '''
        

        Parameters
        ----------
        fn : str
            Path to where to save the scenes in a txt file.

        Returns
        -------
        None.

        '''
        assert self.scenes
        
        folder = os.path.dirname(fn)
        os.makedirs(folder, exist_ok=True)
        
        with open(fn, 'w') as f:
            for scene in self.scenes:
                f.write(f'{scene}\n')

        
def powerset(iterable):
    '''
    

    Parameters
    ----------
    iterable : iterable
        DESCRIPTION.

    Returns
    -------
    iterable
        All sublists (2^n_elements) of the input list.

    '''
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
    