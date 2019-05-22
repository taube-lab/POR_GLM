# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:17:55 2018

-10-fold cross validation to classify cells as encoding: allocentric direction, center bearing, center distance, or linear speed
-also runs alternative models:
    -cartesian wall bearing
    -cartesian wall distance
    -polar wall distance
    -egocentric boundary encoding
    -2D position encoding
    -

@author: Patrick
"""
import os
import numpy as np
import math
from utilities import collect_data
import model_select
import interpret
from scipy.optimize import minimize
from scipy.sparse import spdiags, csr_matrix
from itertools import chain, combinations
from scipy.stats import pearsonr
from astropy.convolution.kernels import Gaussian1DKernel
from astropy.convolution import convolve
import pickle

hd_bins = 30
dist_bins = 10
ebc_angle_bins = 12
ebc_dist_bins = 8

def compute_diags():
    ''' create diagonal matrices for grouped penalization -- implementation 
    modified from Hardcastle 2017 '''
    
    'diagonal matrix for computing differences between adjacent circular 1D bins'

    pos_ones = np.ones(hd_bins)
    circ1 = spdiags([-pos_ones,pos_ones],[0,1],hd_bins-1,hd_bins)
    circ_diag = circ1.T * circ1
    circ_diag=np.asarray(circ_diag.todense())
    circ_diag[0] = np.roll(circ_diag[1],-1)
    circ_diag[hd_bins-1] = np.roll(circ_diag[hd_bins-2],1)

    'also one for noncircular'

    pos_ones = np.ones(dist_bins)
    noncirc1 = spdiags([-pos_ones,pos_ones],[0,1],dist_bins-1,dist_bins)
    noncirc_diag = noncirc1.T * noncirc1
    noncirc_diag = np.asarray(noncirc_diag.todense())
        
    return circ_diag, noncirc_diag


def objective(params,X,spike_train,smoothers,smooth=True):
    
    ''' compute objective and gradient with optional smoothing (to prevent overfitting) '''
        
    u = X * params
    rate = np.exp(u)
    
    f = np.sum(rate - spike_train * u)
    grad = X.T * (rate - spike_train)
    
    if smooth:
        fpen,fgrad = penalize(params,X,spike_train,smoothers)
        f += fpen
        grad += fgrad
    
    print f
    return f,grad

def ebc_objective(params,Xebc,Xa,Xs,spike_train,smoothers):
    
    ''' compute objective without smoothing for egocentric boundary comparison 
    (grad is difficult to determine) '''
        
    ebc_params = params[:ebc_angle_bins*ebc_dist_bins]
    allo_params = params[ebc_angle_bins*ebc_dist_bins:(ebc_angle_bins*ebc_dist_bins+hd_bins)]
    speed_params = params[(ebc_angle_bins*ebc_dist_bins+hd_bins):]
    
    rate = ((Xebc*np.exp(ebc_params))/np.array(np.sum(Xebc,axis=1)).flatten())
    rate[np.isnan(rate)] = 1e-6
    rate = rate * (Xa*np.exp(allo_params)) * (Xs*np.exp(speed_params))
    
    f = np.sum(rate - spike_train*np.log(rate))
    
#    print f
    return f

def penalize(params,X,spike_train,smoothers):
    
    ''' apply smoothing penalties to objective and gradient '''
    
    circ_diag = smoothers[0]
    noncirc_diag = smoothers[1]
    
    abeta = 20.
    ebeta = 20.
    dbeta = 20.
    sbeta = 20.
    
    f = np.sum(abeta * .5 * np.dot(params[:hd_bins].T, circ_diag) * params[:hd_bins] )
    f += np.sum(dbeta * .5 * np.dot(params[hd_bins:(hd_bins+dist_bins)].T, noncirc_diag) * params[hd_bins:(hd_bins+dist_bins)] )
    f += np.sum(ebeta * .5 * np.dot(params[(hd_bins+dist_bins):(hd_bins*2+dist_bins)].T, circ_diag) * params[(hd_bins+dist_bins):(hd_bins*2+dist_bins)] )
    f += np.sum(sbeta * .5 * np.dot(params[(hd_bins*2+dist_bins):(hd_bins*2+dist_bins*2)].T, noncirc_diag) * params[(hd_bins*2+dist_bins):] )

    grad = abeta * np.dot(circ_diag,params[:hd_bins])
    grad = np.concatenate((grad,dbeta * np.dot(noncirc_diag,params[hd_bins:(hd_bins+dist_bins)])))
    grad = np.concatenate((grad,ebeta * np.dot(circ_diag,params[(hd_bins+dist_bins):(hd_bins*2+dist_bins)])))
    grad = np.concatenate((grad,sbeta * np.dot(noncirc_diag,params[(hd_bins*2+dist_bins):])))

    return f,grad

def make_X(trial_data,fdir):
    
    ''' make base design matrix '''
    
    center_ego_angles = np.asarray(trial_data['center_ego_angles'])
    center_ego_dists = np.asarray(trial_data['radial_dists'])
    angles = np.asarray(trial_data['angles'])
    speeds = np.asarray(trial_data['speeds'])

    center_ego_bins = np.digitize(center_ego_angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
    center_dist_bins = np.digitize(center_ego_dists,np.linspace(0,np.max(center_ego_dists),dist_bins,endpoint=False))-1
    angle_bins = np.digitize(angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
    speed_bins = np.digitize(speeds,np.linspace(0,40,dist_bins,endpoint=False)) - 1

    Xe = np.zeros((len(center_ego_angles),hd_bins))
    Xd = np.zeros((len(center_ego_angles),dist_bins))
    Xa = np.zeros((len(center_ego_angles),hd_bins))
    Xs = np.zeros((len(center_ego_angles),dist_bins))

    for i in range(len(angle_bins)):
        Xe[i][center_ego_bins[i]] = 1.
        Xd[i][center_dist_bins[i]] = 1.
        Xa[i][angle_bins[i]] = 1.
        Xs[i][speed_bins[i]] = 1.
        
    X = np.concatenate((Xe,Xd,Xa,Xs),axis=1)
    X = csr_matrix(X)

    return X,csr_matrix(Xe),csr_matrix(Xd),csr_matrix(Xa),csr_matrix(Xs)

def compare_X(trial_data,fdir,Xe,Xd,Xa,Xs):
    
    ''' make design matrices for our alternative models '''
    
    wall_ego_angles = np.asarray(trial_data['wall_ego_angles'])
    wall_dists = np.asarray(trial_data['wall_dists'])
    polar_wall_dists = np.asarray(trial_data['polar_wall_dists'])
    center_md_angles = np.asarray(trial_data['md_center_ego_angles'])
    center_x = np.asarray(trial_data['center_x'])
    center_y = np.asarray(trial_data['center_y'])
    
    wall_ego_bins = np.digitize(wall_ego_angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
    wall_dist_bins = np.digitize(wall_dists,np.linspace(0,np.max(wall_dists),dist_bins,endpoint=False))- 1
    polar_wall_dist_bins = np.digitize(polar_wall_dists,np.linspace(0,np.max(polar_wall_dists),dist_bins,endpoint=False)) - 1
    md_angle_bins = np.digitize(center_md_angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
    xbins = np.digitize(center_x,np.linspace(np.min(center_x),np.max(center_x),dist_bins,endpoint=False)) - 1
    ybins = np.digitize(center_y,np.linspace(np.min(center_y),np.max(center_y),dist_bins,endpoint=False))- 1

    Xwe = np.zeros((len(wall_ego_angles),hd_bins))
    Xwd = np.zeros((len(wall_ego_angles),dist_bins))
    Xpwd = np.zeros((len(wall_ego_angles),dist_bins))
    Xmd = np.zeros((len(wall_ego_angles),hd_bins))
    Xp = np.zeros((len(center_x),dist_bins,dist_bins))
    
    for i in range(len(wall_ego_bins)):
        Xwe[i][wall_ego_bins[i]] = 1.
        Xwd[i][wall_dist_bins[i]] = 1.
        Xpwd[i][polar_wall_dist_bins[i]] = 1.
        Xmd[i][md_angle_bins[i]] = 1.
        Xp[i][xbins[i]][ybins[i]] = 1.
        
    Xe = Xe.todense()
    Xd = Xd.todense()
    Xa = Xa.todense()
    Xs = Xs.todense()
        
    we_X = np.concatenate((Xwe,Xd,Xa,Xs),axis=1)
    we_X = csr_matrix(we_X)
    
    wd_X = np.concatenate((Xe,Xwd,Xa,Xs),axis=1)
    wd_X = csr_matrix(wd_X)
    
    pwd_X = np.concatenate((Xe,Xpwd,Xa,Xs),axis=1)
    pwd_X = csr_matrix(pwd_X)
    
    md_X = np.concatenate((Xmd,Xd,Xa,Xs),axis=1)
    md_X = csr_matrix(md_X)
        
    Xebc = collect_data.calc_ebc_vals(trial_data,angle_bins=ebc_angle_bins,dist_bins=ebc_dist_bins)
    ebc_X = np.concatenate((Xebc,Xa,Xs),axis=1)
    ebc_X = csr_matrix(ebc_X)
    
    Xp = np.reshape(Xp,(len(center_x),dist_bins**2))
    pos_X = np.concatenate((Xe,Xp,Xa,Xs),axis=1)
    pos_X = csr_matrix(pos_X)
    
    return we_X,wd_X,pwd_X,md_X,ebc_X,pos_X,csr_matrix(Xwe),csr_matrix(Xwd),csr_matrix(Xpwd),csr_matrix(Xmd),csr_matrix(Xebc),csr_matrix(Xp)


def split_data(X,Xe,Xd,Xa,Xs,spike_train,fold):
    
    ''' split data into 10 parts for our cross-validation '''

    break_points = np.linspace(0,len(spike_train),51).astype(np.int)


    slices = np.r_[break_points[fold]:break_points[fold + 1],break_points[fold + 10]:break_points[fold + 11],break_points[fold + 20]:break_points[fold + 21],
                          break_points[fold + 30]:break_points[fold + 31],break_points[fold + 40]:break_points[fold + 41]]
    
    test_spikes = spike_train[slices]
    test_Xe = csr_matrix(Xe.todense()[slices])
    test_Xd = csr_matrix(Xd.todense()[slices])
    test_Xa = csr_matrix(Xa.todense()[slices])
    test_Xs = csr_matrix(Xs.todense()[slices])
    
    train_spikes = np.delete(spike_train,slices,axis=0)
    train_X = csr_matrix(np.delete(X.todense(),slices,axis=0))
    train_Xe = csr_matrix(np.delete(Xe.todense(),slices,axis=0))
    train_Xd = csr_matrix(np.delete(Xd.todense(),slices,axis=0))
    train_Xa = csr_matrix(np.delete(Xa.todense(),slices,axis=0))
    train_Xs = csr_matrix(np.delete(Xs.todense(),slices,axis=0))

    
    return test_spikes,test_Xe,test_Xd,test_Xa,test_Xs,train_spikes,train_X,train_Xe,train_Xd,train_Xa,train_Xs
    

def calc_scale_factor(model,center_ego_params,center_dist_params,allo_params,speed_params,train_Xe,train_Xd,train_Xa,train_Xs,train_spikes):
    
    ''' drawing parameters from the fully optimized model keeps parameters independent
    but disrupts response scaling - compute a scale factor to return the predicted spike
    train to the correct scale '''
    
    u = np.zeros(len(train_spikes))
    
    if 'center_ego' in model:
        u += train_Xe * center_ego_params
    if 'center_dist' in model:
        u += train_Xd * center_dist_params
    if 'allo' in model:
        u += train_Xa * allo_params
    if 'speed' in model:
        u += train_Xs * speed_params
    
    rate = np.exp(u)
    
    scale_factor = np.sum(train_spikes)/np.sum(rate)
    
    return scale_factor
    

def run_final(model,scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train):
    
    ''' run the model and collect the results '''
    
    if model != 'uniform':
    
        u = np.zeros(len(spike_train))
        
        if 'center_ego' in model:
            u += Xe * center_ego_params
        if 'center_dist' in model:
            u += Xd * center_dist_params
        if 'allo' in model:
            u += Xa * allo_params
        if 'speed' in model:
            u += Xs * speed_params
        
        rate = np.exp(u)
        
    else:
        
        rate = np.mean(spike_train)
    
    f = -np.sum(rate * scale_factor - spike_train*np.log(rate * scale_factor))
    
    lgammas = np.zeros(len(spike_train))
    for h in range(len(spike_train)):
        lgammas[h] = np.log(math.gamma(spike_train[h]+1))
        
    f -= np.sum(lgammas)
    
    #change from nats to bits
    f = f/np.log(2)

    llps = f/np.sum(spike_train)
    
    #calculate pearson r between the estimated spike train and
    #actual spike train, first smoothing with a gaussian filter
    smoothed_spikes = convolve(spike_train, Gaussian1DKernel(stddev=2,x_size=11))

    r,p = pearsonr(smoothed_spikes,rate*scale_factor)
    if np.isnan(r):
        r = 0
        
    #now calculate the percent of variance explained by the model estimates
    mean_fr = np.mean(smoothed_spikes)
    explained_var = 1 - np.nansum((smoothed_spikes - rate*scale_factor)**2)/np.sum((smoothed_spikes - mean_fr)**2)
    
    pseudo_r2 = 1 - np.nansum(spike_train * np.log(spike_train / (rate*scale_factor)) - (spike_train - rate*scale_factor)) / np.nansum(spike_train * np.log(spike_train / np.mean(spike_train)))
    uniform_r2 = 0
    
    pseudo_r2 = pseudo_r2 - uniform_r2
    
    print('-----------------------')
    print(model)
    print(' ')
    print('scale_factor: %f' % scale_factor)
    print('log-likelihood: %f' % f)
    print('llps: %f' % llps)
    print('correlation: %f' % r)
    print('explained_var: %f' % explained_var)
    print('pseudo_r2: %f' % pseudo_r2)
    print(' ')
    print('-----------------------')


    cdict = {}
    #add relevant variables to the appropriate dictionary
    cdict['ll'] = f
    if np.sum(spike_train) > 0:
        cdict['llps'] = float(f/np.sum(spike_train))
    else:
        cdict['llps'] = f
    cdict['lambda'] = rate * scale_factor
    cdict['corr_r'] = r
    cdict['pseudo_r2'] = pseudo_r2
    cdict['explained_var'] = explained_var
    cdict['test_spikes'] = spike_train
    cdict['tot_spikes'] = np.sum(spike_train)
    cdict['scale_factor'] = scale_factor

    return cdict

def comp_scale_factor(model,center_ego_params,center_dist_params,allo_params,speed_params,comp_params,train_Xe,train_Xd,train_Xa,train_Xs,Xcomp,train_spikes):
    
    ''' compute a scale factor for our alternative model '''
    
    rate = np.ones(len(spike_train))
    
    if 'ebc' in model:
        rate = ((Xcomp*np.exp(comp_params))/np.array(np.sum(Xcomp,axis=1)).flatten())
        rate[np.isnan(rate)] = 1e-6
    else:
        rate = rate * (Xcomp * np.exp(comp_params))

    if 'center_ego' in model:
        rate = rate * (Xe * np.exp(center_ego_params))
    if 'center_dist' in model:
        rate = rate * (Xd * np.exp(center_dist_params))
    if 'allo' in model:
        rate = rate * (Xa * np.exp(allo_params))
    if 'speed' in model:
        rate = rate * (Xs * np.exp(speed_params))
    
    scale_factor = np.sum(train_spikes)/np.sum(rate)
    
    return scale_factor
    

def run_comp(model,scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,comp_params,Xe,Xd,Xa,Xs,Xcomp,spike_train,base=False):
    
    ''' run our alternate model and collect the result '''
    
    if model != 'uniform':
    
        param_count = 0
        rate = np.ones(len(spike_train))
        
        if 'ebc' in model:
            rate = ((Xcomp*np.exp(comp_params))/np.array(np.sum(Xcomp,axis=1)).flatten())
            rate[np.isnan(rate)] = 1e-6
        elif not base:
            rate = rate * (Xcomp * np.exp(comp_params))
            
        if not base:
            param_count += len(comp_params)
                
        if 'center_ego' in model:
            rate = rate * (Xe * np.exp(center_ego_params))
            param_count += len(center_ego_params)
        if 'center_dist' in model:
            rate = rate * (Xd * np.exp(center_dist_params))
            param_count += len(center_dist_params)
        if 'allo' in model:
            rate = rate * (Xa * np.exp(allo_params))
            param_count += len(allo_params)
        if 'speed' in model:
            rate = rate * (Xs * np.exp(speed_params))
            param_count += len(speed_params)
                
    else:
        
        rate = np.mean(spike_train)
    
    f = -np.sum(rate * scale_factor - spike_train*np.log(rate * scale_factor))
    
    lgammas = np.zeros(len(spike_train))
        
    for h in range(len(spike_train)):
        lgammas[h] = np.log(math.gamma(spike_train[h]+1))
        
    f -= np.sum(lgammas)

    aic = 2*param_count - 2*f
    bic = np.log(len(spike_train))*param_count - 2*f
    
    return aic,bic,f
    

def get_all_models(variables):
    ''' convenience function for calculating all possible
    combinations of nagivational variables '''
    
    def powerset(variables):
        return list(chain.from_iterable(combinations(variables, r) for r in range(1,len(variables)+1)))
    
    all_models = powerset(variables)
    
    for i in range(len(all_models)):
        all_models[i] = frozenset(all_models[i])
    
    return all_models
    


if __name__ == '__main__':
    
    variables = [('allo'),('center_ego'),('center_dist'),('speed')]
    all_models = get_all_models(variables)
    
    cwd = os.getcwd()
    fdir = cwd + '/example_data'
        
    for animal in os.listdir(fdir):
        print animal
        animaldir = fdir + '/' + animal
        
        if not os.path.isdir(animaldir):
            continue
        
        for session in os.listdir(animaldir):
            area=session
            sessiondir = animaldir + '/' + session
            
            if not os.path.isdir(sessiondir):
                continue

            for trial in os.listdir(sessiondir):
                print trial
                tracking_fdir = sessiondir + '/' + trial
                if not os.path.isdir(tracking_fdir):
                    continue 
                
                timestamps,center_x,center_y,angles = collect_data.read_video_file(tracking_fdir + '/tracking_data.txt')
                trial_data = {'timestamps':timestamps,'center_x':center_x,'center_y':center_y,'angles':angles}
                    
                spike_timestamps = []
                for i in range(len(timestamps)):
                    if i < len(timestamps)-1:
                        increment = (timestamps[i+1]-timestamps[i])/(1000./(30.))
                        for j in range(int(1000./((30.)))):
                            spike_timestamps.append(timestamps[i]+j*increment)
                trial_data['spike_timestamps'] = spike_timestamps
                
                #calculate speed, egocentric variables, and movement directions
                trial_data = collect_data.ego_stuff(trial_data)
                trial_data = collect_data.speed_stuff(trial_data,ahv=False)
                trial_data = collect_data.calc_movement_direction(trial_data)
                trial_data = collect_data.calc_md_radial_dist(trial_data)
                
                #make design matrix for basic model
                X,Xe,Xd,Xa,Xs = make_X(trial_data,fdir)
                #make additional design matrices for alternative models
                we_X,wd_X,pwd_X,md_X,ebc_X,pos_X,Xwe,Xwd,Xpwd,Xmd,Xebc,Xp = compare_X(trial_data,fdir,Xe,Xd,Xa,Xs)
                #make smoothing matrices for regularization in cross-validation
                smoothers = compute_diags()
                
                clusters = os.listdir(tracking_fdir)
                clusters.remove('tracking_data.txt')
                
                for cell in clusters:
                    
                    cdict = {}
                    cluster_data = {}
                    
                    save_dir = cwd + '/class_dicts'
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    ts_file = tracking_fdir + '/' + cell
                    cluster_data['spike_list'] = collect_data.ts_file_reader(ts_file)
                    spike_data, cluster_data = collect_data.create_spike_lists(trial_data,cluster_data)
                    spike_train = spike_data['ani_spikes']
        
                    for fold in range(10):
                        cdict[fold] = {}
                        test_spikes,test_Xe,test_Xd,test_Xa,test_Xs,train_spikes,train_X,train_Xe,train_Xd,train_Xa,train_Xs = split_data(X,Xe,Xd,Xa,Xs,spike_train,fold)
                        params = np.zeros(np.shape(train_X)[1])
                        result = minimize(objective,params,args=(train_X,train_spikes,smoothers),jac=True,method='L-BFGS-B')
        
                        params = result.x
                        center_ego_params = params[:hd_bins]
                        center_dist_params = params[hd_bins:(hd_bins+dist_bins)]
                        allo_params = params[(hd_bins+dist_bins):(2*hd_bins+dist_bins)]
                        speed_params = params[(2*hd_bins+dist_bins):]
                        
                        for model in all_models:
                            
                            scale_factor = calc_scale_factor(model,center_ego_params,center_dist_params,allo_params,speed_params,train_Xe,train_Xd,train_Xa,train_Xs,train_spikes)
                            cdict[fold][model] = run_final(model,scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,test_Xe,test_Xd,test_Xa,test_Xs,test_spikes)
                        
                        cdict[fold]['uniform'] = run_final('uniform',1.,center_ego_params,center_dist_params,allo_params,speed_params,test_Xe,test_Xd,test_Xa,test_Xs,test_spikes)
                
                
                    best_model = model_select.select_model(cdict)
                    print best_model
                    
                    if best_model != 'uniform':
                        params = np.zeros(np.shape(train_X)[1])
                        full_result = minimize(objective,params,args=(X,spike_train,smoothers,False),jac=True,method='L-BFGS-B')
                        params = full_result.x
                        center_ego_params = params[:hd_bins]
                        center_dist_params = params[hd_bins:(hd_bins+dist_bins)]
                        allo_params = params[(hd_bins+dist_bins):(2*hd_bins+dist_bins)]
                        speed_params = params[(2*hd_bins+dist_bins):]
                        
                        scale_factor = calc_scale_factor(best_model,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                        model_dict = run_final(best_model,scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
                    
                        if 'center_ego' in best_model:

                            new_params = np.zeros(np.shape(ebc_X)[1])
                            ebc_result = minimize(ebc_objective,new_params,args=(Xebc,Xa,Xs,spike_train,smoothers),jac=False,method='L-BFGS-B')
                            new_params = ebc_result.x
                            nebc_params = new_params[:np.shape(Xebc)[1]]
                            nallo_params = new_params[(np.shape(Xebc)[1]):(hd_bins+np.shape(Xebc)[1])]
                            nspeed_params = new_params[(hd_bins+np.shape(Xebc)[1]):]
                            
                            model = ('ebc',)
                            print 'running ebc model'
                            if 'speed' in best_model:
                                model += ('speed',)
                            if 'allo' in best_model:
                                model += ('allo',)
                            model = frozenset(model)
                            new_scale_factor = comp_scale_factor(model,center_ego_params,center_dist_params,nallo_params,nspeed_params,nebc_params,Xe,Xd,Xa,Xs,Xebc,spike_train)
                            ebc_aic,ebc_bic,ebc_ll = run_comp(model,new_scale_factor,center_ego_params,center_dist_params,nallo_params,nspeed_params,nebc_params,Xe,Xd,Xa,Xs,Xebc,spike_train)
                            
                            
                            new_params = np.zeros(np.shape(we_X)[1])
                            we_result = minimize(objective,new_params,args=(we_X,spike_train,smoothers,False),jac=True,method='L-BFGS-B')
                            new_params = we_result.x
                            nwe_params = new_params[:np.shape(Xwe)[1]]
                            ncenter_dist_params = new_params[np.shape(Xwe)[1]:(dist_bins+np.shape(Xwe)[1])]
                            nallo_params = new_params[(dist_bins+np.shape(Xwe)[1]):(dist_bins+hd_bins+np.shape(Xwe)[1])]
                            nspeed_params = new_params[(dist_bins+hd_bins+np.shape(Xwe)[1]):]
                            
                            model = ('wall_ego',)
                            print 'running cartesian wall ego model'
                            if 'center_dist' in best_model:
                                model += ('center_dist',)
                            if 'speed' in best_model:
                                model += ('speed',)
                            if 'allo' in best_model:
                                model += ('allo',)
                            model = frozenset(model)
                            new_scale_factor = comp_scale_factor(model,center_ego_params,ncenter_dist_params,nallo_params,nspeed_params,nwe_params,Xe,Xd,Xa,Xs,Xwe,spike_train)
                            we_aic,we_bic,we_ll = run_comp(model,new_scale_factor,center_ego_params,ncenter_dist_params,nallo_params,nspeed_params,nwe_params,Xe,Xd,Xa,Xs,Xwe,spike_train)


                            new_params = np.zeros(np.shape(md_X)[1])
                            md_result = minimize(objective,new_params,args=(md_X,spike_train,smoothers,False),jac=True,method='L-BFGS-B')
                            new_params = md_result.x
                            nmd_params = new_params[:np.shape(Xmd)[1]]
                            ncenter_dist_params = new_params[np.shape(Xmd)[1]:(dist_bins+np.shape(Xmd)[1])]
                            nallo_params = new_params[(dist_bins+np.shape(Xmd)[1]):(dist_bins+hd_bins+np.shape(Xmd)[1])]
                            nspeed_params = new_params[(dist_bins+hd_bins+np.shape(Xmd)[1]):]
                            
                            model = ('movement_dir',)
                            print 'running movement direction ego model'
                            if 'center_dist' in best_model:
                                model += ('center_dist',)
                            if 'speed' in best_model:
                                model += ('speed',)
                            if 'allo' in best_model:
                                model += ('allo',)
                            model = frozenset(model)
                            new_scale_factor = comp_scale_factor(model,center_ego_params,ncenter_dist_params,nallo_params,nspeed_params,nmd_params,Xe,Xd,Xa,Xs,Xmd,spike_train)
                            md_aic,md_bic,md_ll = run_comp(model,new_scale_factor,center_ego_params,ncenter_dist_params,nallo_params,nspeed_params,nmd_params,Xe,Xd,Xa,Xs,Xmd,spike_train)

                        if 'center_dist' in best_model:

                            if 'center_ego' not in best_model:
                                new_params = np.zeros(np.shape(ebc_X)[1])
                                ebc_result = minimize(ebc_objective,new_params,args=(Xebc,Xa,Xs,spike_train,smoothers),jac=False,method='L-BFGS-B')
                                new_params = ebc_result.x
                                nebc_params = new_params[:np.shape(Xebc)[1]]
                                nallo_params = new_params[(np.shape(Xebc)[1]):(hd_bins+np.shape(Xebc)[1])]
                                nspeed_params = new_params[(hd_bins+np.shape(Xebc)[1]):]
                                
                                model = ('ebc',)
                                print 'running ebc model'
                                if 'speed' in best_model:
                                    model += ('speed',)
                                if 'allo' in best_model:
                                    model += ('allo',)
                                model = frozenset(model)
                                new_scale_factor = comp_scale_factor(model,center_ego_params,center_dist_params,nallo_params,nspeed_params,nebc_params,Xe,Xd,Xa,Xs,Xebc,spike_train)
                                ebc_aic,ebc_bic,ebc_ll = run_comp(model,new_scale_factor,center_ego_params,center_dist_params,nallo_params,nspeed_params,nebc_params,Xe,Xd,Xa,Xs,Xebc,spike_train)
                            
                            
                            
                            new_params = np.zeros(np.shape(wd_X)[1])
                            wd_result = minimize(objective,new_params,args=(wd_X,spike_train,smoothers,False),jac=True,method='L-BFGS-B')
                            new_params = wd_result.x
                            ncenter_ego_params = new_params[:hd_bins]
                            nwd_params = new_params[hd_bins:(hd_bins+np.shape(Xwd)[1])]
                            nallo_params = new_params[(hd_bins+np.shape(Xwd)[1]):(2*hd_bins+np.shape(Xwd)[1])]
                            nspeed_params = new_params[(2*hd_bins+np.shape(Xwd)[1]):]
                            
                            model = ('wall_dist',)
                            print 'running cartesian wall dist model'
                            if 'center_ego' in best_model:
                                model += ('center_ego',)
                            if 'speed' in best_model:
                                model += ('speed',)
                            if 'allo' in best_model:
                                model += ('allo',)
                            model = frozenset(model)
                            new_scale_factor = comp_scale_factor(model,center_ego_params,nwd_params,nallo_params,nspeed_params,nwd_params,Xe,Xd,Xa,Xs,Xwd,spike_train)
                            wd_aic,wd_bic,wd_ll = run_comp(model,new_scale_factor,center_ego_params,nwd_params,nallo_params,nspeed_params,nwd_params,Xe,Xd,Xa,Xs,Xwd,spike_train)



                            new_params = np.zeros(np.shape(pwd_X)[1])
                            pwd_result = minimize(objective,new_params,args=(pwd_X,spike_train,smoothers,False),jac=True,method='L-BFGS-B')
                            new_params = pwd_result.x
                            ncenter_ego_params = new_params[:hd_bins]
                            npwd_params = new_params[hd_bins:(hd_bins+np.shape(Xpwd)[1])]
                            nallo_params = new_params[(hd_bins+np.shape(Xpwd)[1]):(2*hd_bins+np.shape(Xpwd)[1])]
                            nspeed_params = new_params[(2*hd_bins+np.shape(Xpwd)[1]):]
                            
                            model = ('polar_wall_dist',)
                            print 'running polar wall dist model'
                            if 'center_ego' in best_model:
                                model += ('center_ego',)
                            if 'speed' in best_model:
                                model += ('speed',)
                            if 'allo' in best_model:
                                model += ('allo',)
                            model = frozenset(model)
                            new_scale_factor = comp_scale_factor(model,center_ego_params,npwd_params,nallo_params,nspeed_params,nwd_params,Xe,Xd,Xa,Xs,Xpwd,spike_train)
                            pwd_aic,pwd_bic,pwd_ll = run_comp(model,new_scale_factor,center_ego_params,nwd_params,nallo_params,nspeed_params,npwd_params,Xe,Xd,Xa,Xs,Xpwd,spike_train)

                            
                            new_params = np.zeros(np.shape(pos_X)[1])
                            pos_result = minimize(objective,new_params,args=(pos_X,spike_train,smoothers,False),jac=True,method='L-BFGS-B')
                            new_params = pos_result.x
                            ncenter_ego_params = new_params[:hd_bins]
                            npos_params = new_params[hd_bins:(hd_bins+np.shape(Xp)[1])]
                            nallo_params = new_params[(hd_bins+np.shape(Xp)[1]):(2*hd_bins+np.shape(Xp)[1])]
                            nspeed_params = new_params[(2*hd_bins+np.shape(Xp)[1]):]
                            
                            model = ('2d_pos',)
                            print 'running 2D position model'
                            if 'center_ego' in best_model:
                                model += ('center_ego',)
                            if 'speed' in best_model:
                                model += ('speed',)
                            if 'allo' in best_model:
                                model += ('allo',)
                            model = frozenset(model)
                            new_scale_factor = comp_scale_factor(model,center_ego_params,nwd_params,nallo_params,nspeed_params,npos_params,Xe,Xd,Xa,Xs,Xp,spike_train)
                            pos_aic,pos_bic,pos_ll = run_comp(model,new_scale_factor,center_ego_params,nwd_params,nallo_params,nspeed_params,npos_params,Xe,Xd,Xa,Xs,Xp,spike_train)

                        base_aic,base_bic,base_ll = run_comp(best_model,scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,[],Xe,Xd,Xa,Xs,[],spike_train,base=True)
       
                    else:
                        model_dict = run_final('uniform',1.,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)
        
                    if best_model != 'uniform':
                        model_dict = interpret.calc_contribs(best_model,model_dict,spike_train,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs)                        
                        if 'center_ego' in best_model:
                            model_dict['center_ego_params'] = np.exp(center_ego_params)
                            model_dict['ebc_aic'] = ebc_aic
                            model_dict['ebc_bic'] = ebc_bic
                            model_dict['ebc_ll'] = ebc_ll
                            model_dict['movement_dir_aic'] = md_aic
                            model_dict['movement_dir_bic'] = md_bic
                            model_dict['movement_dir_ll'] = md_ll
                            model_dict['wall_ego_aic'] = we_aic
                            model_dict['wall_ego_bic'] = we_bic
                            model_dict['wall_ego_ll'] = we_ll
                        if 'center_dist' in best_model:
                            model_dict['center_dist_params'] = np.exp(center_dist_params)
                            model_dict['ebc_aic'] = ebc_aic
                            model_dict['ebc_bic'] = ebc_bic
                            model_dict['ebc_ll'] = ebc_ll
                            model_dict['wall_dist_aic'] = wd_aic
                            model_dict['wall_dist_bic'] = wd_bic
                            model_dict['wall_dist_ll'] = wd_ll
                            model_dict['polar_wall_dist_aic'] = pwd_aic
                            model_dict['polar_wall_dist_bic'] = pwd_bic
                            model_dict['polar_wall_dist_ll'] = pwd_ll
                            model_dict['2d_pos_aic'] = pos_aic
                            model_dict['2d_pos_bic'] = pos_bic
                            model_dict['2d_pos_ll'] = pos_ll
                        if 'allo' in best_model:
                            model_dict['allo_params'] = np.exp(allo_params)
                        if 'speed' in best_model:
                            model_dict['speed_params'] = np.exp(speed_params)
                            
                        model_dict['base_aic'] = base_aic
                        model_dict['base_bic'] = base_bic
                        model_dict['base_ll'] = base_ll


                    with open((save_dir+'/%s_%s_%s_cval_%s.pickle' % (animal, trial, session, cell)),'wb') as f:
                        pickle.dump(cdict,f,protocol=2)
                        
                    with open((save_dir+'/%s_%s_%s_best_model_%s.pickle' % (animal, trial, session, cell)),'wb') as f:
                        pickle.dump(model_dict,f,protocol=2)
                        
    interpret.area_contribs(fdir,cwd)
