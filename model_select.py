# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:57:29 2018

@author: Patrick
"""

import numpy as np
import copy
from itertools import chain
import pickle
from scipy.stats import wilcoxon


def load_data(fname):
    with open(fname,'rb') as f:
        data = pickle.load(f)
    return data

def select_model(cdict):
    ''' perform forward search procedure to choose the best model based on
    cross-validation results '''
    
    #variables we'll be looking at
    variables = [('allo',),('speed',),('center_ego',),('center_dist',)]
    #models we'll start with are single-variable models
    models = copy.deepcopy(variables)
    #we haven't found a best model yet so set to NONE
    best_model = None
        
    #while we haven't found a winning model...
    while best_model == None:
        
        #make models frozensets to make things easier
        for i in range(len(models)):
            models[i] = frozenset(models[i])
        
        #start dict for llps measures
        ll_dict = {}
        
        #for each fold in the cross-val...
        for fold in range(10):
            #for each model...
            for modeltype in models:
                #start an entry for that model if we're just starting
                if fold == 0:
                    ll_dict[modeltype] = []
                #collect the llps increase compared to the uniform model
                ll_dict[modeltype].append(cdict[fold][modeltype]['llps']-cdict[fold]['uniform']['llps'])
        
        #make a dict that contains the median llps value for each model
        median_dict = {}
        for modeltype in models:
            median_dict[modeltype] = np.median(ll_dict[modeltype])
        #let's say the potential best new model is the one with the highest median score
        top_model = max(median_dict.iterkeys(), key=(lambda key: median_dict[key]))
        
        print top_model

        #if the top model is a single variable...
        if len(top_model) == 1:
            #set the top model llps data as 'last_model' data
            last_model = ll_dict[top_model]
            #set the top model the 'last_modeltype'
            last_modeltype = top_model
            #create the next set of models -- the current best model plus each new variable
            #-- then start over
            models = []
            for var in variables:
                if var[0] not in top_model:
                    models.append(frozenset(chain(list(top_model),list(var))))
        #otherwise...
        else:
            #use wilcoxon signed ranks to see if the new model is better than the last model
            w,p = wilcoxon(last_model,ll_dict[top_model])
            print p
            #if the new model is better...
            if np.median(last_model) < np.median(median_dict[top_model]) and p < .1:
                #if we can't add any more variables...
                if len(top_model) == len(variables):
                    #test to see if the top model is better than the null model
                    w,p = wilcoxon(ll_dict[top_model],np.zeros(10))
                    print p
                    #if it is, then this is the best model!
                    if np.median(ll_dict[top_model]) > 0 and p < .1:
                        best_model = top_model
                    #otherwise, the cell is unclassifiable
                    else:
                        best_model = 'uniform'
                    
                #otherwise, set this model's llps data as 'last_model' data
                last_model = ll_dict[top_model]
                #set this modeltype as 'last_modeltype'
                last_modeltype = top_model
                #make new set of models -- current best model plus each new variable
                #-- then start over
                models = []
                for var in variables:
                    if var[0] not in top_model:
                        models.append(frozenset(chain(list(top_model),list(var))))
            #otherwise, the best model is probably the last model
            else:
                #check if the last model is better than the null  model
                w,p = wilcoxon(last_model,np.zeros(10))
                print p
                print last_model
                #if it is, then this is the best model!
                if np.median(last_model) > 0 and p < .1:
                    best_model = last_modeltype
                #otherwise, the cell is unclassifiable
                else:
                    best_model = 'uniform'
                    
    #return the best model
    return best_model

