# -*- coding: utf-8 -*-
"""
Created on Sat Jul 07 12:19:47 2018

interpret results of cell classification/modeling

@author: Patrick
"""

import pickle
import os
from itertools import chain

import GLM as full_classify


def load_data(fname):
    ''' load pickled numpy arrays '''

    with open(fname,'rb') as f:
        data = pickle.load(f)
    
    return data

def calc_contribs(best_model,model_dict,spike_train,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs):
    ''' calculate the effect on different measures of goodness-of-fit when adding
    or subtracting each variable '''
        
    #start a dict for contribs
    contribs = {}
    #note which variables we're looking at
    variables = [('allo',),('center_ego',),('center_dist',),('speed',)]
    
    #if the best model is a single-variable model... and not null...
    if len(best_model) == 1 and 'uniform' not in best_model:
        #calculate goodness-of-fit measures for the null model
        uniform_model_dict = full_classify.run_final('uniform',1.,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
        #calculate difference in goodness-of-fit measures between the best model and the null
        #model -- these are the contributions for the single encoded variable
        contribs[best_model] = {}
        contribs[best_model]['ll'] = model_dict['ll'] - uniform_model_dict['ll']
        contribs[best_model]['llps'] = model_dict['llps'] - uniform_model_dict['llps']
        contribs[best_model]['explained_var'] = model_dict['explained_var'] - uniform_model_dict['explained_var']
        contribs[best_model]['corr_r'] = model_dict['corr_r'] - uniform_model_dict['corr_r']
        contribs[best_model]['pseudo_r2'] = model_dict['pseudo_r2'] - uniform_model_dict['pseudo_r2']
        
        #for each variable...
        for var in variables:
            #not including the one we just looked at...
            if frozenset(var) != best_model:
                #create a new model which contains the single encoded variable as well as
                #this new variable
                new_model = frozenset(chain(list(best_model),list(var)))

                #calc a new scale factor
                new_scale_factor = full_classify.calc_scale_factor(new_model,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                #run the new model and collect the result
                new_model_dict = full_classify.run_final(new_model,new_scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                
                #calculate difference in goodness-of-fit measures between the new model and
                #the best model -- these are the contributions from the added variable
                contribs[frozenset(var)] = {}
                contribs[frozenset(var)]['ll'] = new_model_dict['ll'] - model_dict['ll']
                contribs[frozenset(var)]['llps'] = new_model_dict['llps'] - model_dict['llps']
                contribs[frozenset(var)]['explained_var'] = new_model_dict['explained_var'] - model_dict['explained_var']
                contribs[frozenset(var)]['corr_r'] = new_model_dict['corr_r'] - model_dict['corr_r']
                contribs[frozenset(var)]['pseudo_r2'] = new_model_dict['pseudo_r2'] - model_dict['pseudo_r2']
        
    #otherwise, if there are multiple variables in the best model...
    elif len(best_model) > 1:
        #for each variable in the whole list...
        for var in variables:
            #if this variable is included in the best model...
            if var[0] in best_model:
                #create a new model that includes all the variables in the best
                #model EXCEPT this one
                new_model = []
                for i in best_model:
                    if i != var[0]:
                        new_model.append(i)
                new_model = frozenset(new_model)

                #calculate the new scale factor
                new_scale_factor = full_classify.calc_scale_factor(new_model,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                #run the new model and collect the result
                new_model_dict = full_classify.run_final(new_model,new_scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                #calculate difference in goodness-of-fit measures between the best model and
                #the new model -- these are the contributions from the subtracted variable
                contribs[frozenset(var)] = {}
                contribs[frozenset(var)]['ll'] = model_dict['ll'] - new_model_dict['ll']
                contribs[frozenset(var)]['llps'] = model_dict['llps'] - new_model_dict['llps']
                contribs[frozenset(var)]['explained_var'] = model_dict['explained_var'] - new_model_dict['explained_var']
                contribs[frozenset(var)]['corr_r'] = model_dict['corr_r'] - new_model_dict['corr_r']
                contribs[frozenset(var)]['pseudo_r2'] = model_dict['pseudo_r2'] - new_model_dict['pseudo_r2']
                
            #otherwise...
            else:
                #make a new model that adds the current variable to the best model
                new_model = frozenset(chain(list(best_model),list(var)))

                #calc the new scale factor
                new_scale_factor = full_classify.calc_scale_factor(new_model,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                #run the new model and collect the result
                new_model_dict = full_classify.run_final(new_model,new_scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                #calculate difference in goodness-of-fit measures between the new model and
                #the best model -- these are the contributions from the added variable
                contribs[frozenset(var)] = {}
                contribs[frozenset(var)]['ll'] = new_model_dict['ll'] - model_dict['ll']
                contribs[frozenset(var)]['llps'] = new_model_dict['llps'] - model_dict['llps']
                contribs[frozenset(var)]['explained_var'] = new_model_dict['explained_var'] - model_dict['explained_var']
                contribs[frozenset(var)]['corr_r'] = new_model_dict['corr_r'] - model_dict['corr_r']
                contribs[frozenset(var)]['pseudo_r2'] = new_model_dict['pseudo_r2'] - model_dict['pseudo_r2']
    
    #add the new stuff to the model dict
    model_dict['contribs'] = contribs
    model_dict['best_model'] = best_model
    
    #return the model dict
    return model_dict

def area_contribs(fdir,cwd):
    
    import csv
    
    csv_row = []
    csv_row += ['animal','area','cell','trial']
    csv_row += ['center ego','center ego llps','center ego ll','center ego explained var','center ego corr']
    csv_row += ['center dist','center dist llps','center dist ll','center dist explained var','center dist corr']
    csv_row += ['hd','hd llps','hd ll','hd explained var','hd corr']
    csv_row += ['speed','speed llps','speed ll','speed explained var','speed corr']
    csv_row += ['base aic','base bic','base ll']
    csv_row += ['wall ego aic','wall ego bic','wall ego ll']
    csv_row += ['ebc aic','ebc bic','ebc ll']
    csv_row += ['movement dir aic','movement dir bic','movement dir ll']
    csv_row += ['pos aic','pos bic','pos ll']
    csv_row += ['wall dist aic','wall dist bic','wall dist ll']
    csv_row += ['polar wall dist aic','polar wall dist bic','polar wall dist ll']


    csv_fname = fdir + '/classified.txt'
    with open(csv_fname, 'wb') as csvfile:
        #create a writer
        writer = csv.writer(csvfile,dialect='excel-tab')
        writer.writerow(csv_row)

    for animal in os.listdir(fdir):
        animaldir = fdir + '/' + animal
        if not os.path.isdir(animaldir):
            continue
        print animal
        for area in os.listdir(animaldir):
            if os.path.isdir(animaldir+'/'+area):
                areadir = animaldir + '/' + area
                print area
                for trial in os.listdir(areadir):
                    trialdir = areadir + '/' + trial
                    if not os.path.isdir(trialdir):
                        continue
                    clusters = []
                    for f in os.listdir(trialdir):
                        if f.startswith('TT') and f.endswith('.txt'):
                            clusters.append(f)
                        elif f.startswith('ST') and f.endswith('.txt'):
                            clusters.append(f)
                    for cluster in clusters:
                        fname = cwd+'/class_dicts/%s_%s_%s_best_model_%s.pickle' % (animal, trial, area, cluster)
                        model_dict = load_data(fname)
                        if 'center_ego' in model_dict.keys():
                            model_dict['center_ego_params'] = model_dict['center_ego']
                        if 'center_dist' in model_dict.keys():
                            model_dict['center_dist_params'] = model_dict['center_dist']
                        
                        try:
                            print model_dict['best_model']
                        except:
                            csv_row = []
                            csv_row += [animal,area,cluster,trial]
                            csv_row += [0,0,0,0,0]*6
                            with open(csv_fname, 'ab') as csvfile:
                                #create a writer
                                writer = csv.writer(csvfile,dialect='excel-tab')
                                writer.writerow(csv_row)
                            continue

                        
                        csv_row = []
                        
                        csv_row += [animal,area,cluster,trial]
                        
                        if 'center_ego' in model_dict['best_model']:
                            csv_row += [1]
                        else:
                            csv_row += [0]
                        cego_contribs = model_dict['contribs'][frozenset(('center_ego',))]
                        csv_row += [cego_contribs['llps'],cego_contribs['ll'],cego_contribs['explained_var'],cego_contribs['corr_r']]

                        if 'center_dist' in model_dict['best_model']:
                            csv_row += [1]
                        else:
                            csv_row += [0]
                        cdist_contribs = model_dict['contribs'][frozenset(('center_dist',))]
                        csv_row += [cdist_contribs['llps'],cdist_contribs['ll'],cdist_contribs['explained_var'],cdist_contribs['corr_r']]

                        if 'allo' in model_dict['best_model']:
                            csv_row += [1]
                        else:
                            csv_row += [0]
                        hd_contribs = model_dict['contribs'][frozenset(('allo',))]
                        csv_row += [hd_contribs['llps'],hd_contribs['ll'],hd_contribs['explained_var'],hd_contribs['corr_r']]

                        if 'speed' in model_dict['best_model']:
                            csv_row += [1]
                        else:
                            csv_row += [0]
                        speed_contribs = model_dict['contribs'][frozenset(('speed',))]
                        csv_row += [speed_contribs['llps'],speed_contribs['ll'],speed_contribs['explained_var'],speed_contribs['corr_r']]

                        if 'center_ego' in model_dict['best_model'] or 'center_dist' in model_dict['best_model']:
                            csv_row += [model_dict['base_aic'],model_dict['base_bic'],model_dict['base_ll']]
                        else:
                            csv_row += [0,0,0]

                        if 'center_ego' in model_dict['best_model']:
                            csv_row += [model_dict['wall_ego_aic'],model_dict['wall_ego_bic'],model_dict['wall_ego_ll']]
                        else:
                            csv_row += [0,0,0]
                            
                        if 'center_ego' in model_dict['best_model'] or 'center_dist' in model_dict['best_model']:
                            csv_row += [model_dict['ebc_aic'],model_dict['ebc_bic'],model_dict['ebc_ll']]
                        else:
                            csv_row += [0,0,0]
                        
                        if 'center_ego' in model_dict['best_model']:
                            csv_row += [model_dict['movement_dir_aic'],model_dict['movement_dir_bic'],model_dict['movement_dir_ll']]
                        else:
                            csv_row += [0,0,0]


                        if 'center_dist' in model_dict['best_model']:
                            csv_row += [model_dict['2d_pos_aic'],model_dict['2d_pos_bic'],model_dict['2d_pos_ll']]
                            csv_row += [model_dict['wall_dist_aic'],model_dict['wall_dist_bic'],model_dict['wall_dist_ll']]
                            csv_row += [model_dict['polar_wall_dist_aic'],model_dict['polar_wall_dist_bic'],model_dict['polar_wall_dist_ll']]

                        else:
                            csv_row += [0,0,0]
                            csv_row += [0,0,0]
                            csv_row += [0,0,0]

                        with open(csv_fname, 'ab') as csvfile:
                            #create a writer
                            writer = csv.writer(csvfile,dialect='excel-tab')
                            writer.writerow(csv_row)

