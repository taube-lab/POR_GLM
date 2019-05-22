# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:39:35 2018

calculate behavioral correlate values from tracking data

@author: Patrick
"""

import csv
import bisect
import numpy as np

adv = {}
adv['framerate'] = 30.
ops = {}
ops['acq'] = 'neuralynx'
                        
def read_video_file(video_file):
    
    timestamps = []
    xpos = []
    ypos = []
    hdangles = []
    
    with open(video_file,'rb') as f:
        reader = csv.reader(f,dialect='excel-tab')
        for row in reader:
            timestamps.append(float(row[0]))
            xpos.append(float(row[1]))
            ypos.append(float(row[2]))
            hdangles.append(float(row[3]))
            
    return np.array(timestamps), np.array(xpos), np.array(ypos), np.array(hdangles)


def ego_stuff(trial_data):
    
    trial_data = calc_center_ego(trial_data)
    trial_data = calc_wall_ego(trial_data)
    
    return trial_data
        
def speed_stuff(trial_data,ahv=True):
    ''' calculate speed and ahv info from tracking data '''
    
    trial_data['speeds'] = []
    
    print('processing speed data...')
    #calculate running speeds for each frame
    trial_data = calc_speed(trial_data)
        
    return trial_data

def calc_speed(trial_data):
    """calculates 'instantaneous' linear speeds for each video frame"""
    
    #grab appropriate tracking data
    center_x=np.array(trial_data['center_x'])
    center_y=np.array(trial_data['center_y'])
    
    #make an array of zeros to assign speeds to
    speeds = np.zeros(len(center_x),dtype=np.float)
    #for every frame from 2 to total - 2...
    for i in range(2,len(center_x)-2):
        #grab 5 x and y positions centered on that frame
        x_list = center_x[(i-2):(i+3)]
        y_list = center_y[(i-2):(i+3)]
        #find the best fit line for those 5 points (slopes are x and y components
        #of velocity)
        xfitline = np.polyfit(range(0,5),x_list,1)
        yfitline = np.polyfit(range(0,5),y_list,1)
        #total velocity = framerate * sqrt(x component squared plus y component squared)
        speeds[i] = adv['framerate']*np.sqrt(xfitline[0]**2 + yfitline[0]**2)
    #set unassigned speeds equal to closest assigned speed
    speeds[0] = speeds[2]
    speeds[1] = speeds[2]
    speeds[len(speeds)-1] = speeds[len(speeds)-3]
    speeds[len(speeds)-2] = speeds[len(speeds)-3]
    
    #return calculated speeds
    trial_data['speeds'] = speeds
    return trial_data

def calc_movement_direction(trial_data):
    
    #grab appropriate tracking data
    center_x=np.array(trial_data['center_x'])
    center_y=np.array(trial_data['center_y'])
    
    #make an array of zeros to assign speeds to
    mds = np.zeros(len(center_x),dtype=np.float)
    #for every frame from 2 to total - 2...
    for i in range(2,len(center_x)-2):
        #grab 5 x and y positions centered on that frame
        x_list = center_x[(i-2):(i+3)]
        y_list = center_y[(i-2):(i+3)]
        #find the best fit line for those 5 points (slopes are x and y components
        #of velocity)
        xfitline = np.polyfit(range(0,5),x_list,1)
        yfitline = np.polyfit(range(0,5),y_list,1)
        #total velocity = framerate * sqrt(x component squared plus y component squared)
        mds[i] = np.rad2deg(np.arctan2(yfitline[0],xfitline[0]))%360
#        adv['framerate']*np.sqrt(xfitline[0]**2 + yfitline[0]**2)
    #set unassigned speeds equal to closest assigned speed
    mds[0] = mds[2]
    mds[1] = mds[2]
    mds[len(mds)-1] = mds[len(mds)-3]
    mds[len(mds)-2] = mds[len(mds)-3]
    
    #return calculated speeds
    trial_data['movement_directions'] = mds
    return trial_data

def calc_md_radial_dist(trial_data):
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['movement_directions'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2

    center_ego_angles = (np.rad2deg(np.arctan2(-center_y,-center_x)))%360
    center_ego_angles = (center_ego_angles-angles)%360
    
    trial_data['md_center_ego_angles'] = center_ego_angles
    
    return trial_data

def calc_center_ego(trial_data):
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2

    radial_dists = np.sqrt(center_x**2 + center_y**2)
    center_ego_angles = (np.rad2deg(np.arctan2(-center_y,-center_x)))%360
    center_ego_angles = (center_ego_angles-angles)%360
    
    trial_data['radial_dists'] = radial_dists
    trial_data['center_ego_angles'] = center_ego_angles
    
    return trial_data

def calc_wall_ego(trial_data):
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2
    
    #cartesian 
    
    w1 = center_x - np.min(center_x)
    w2 = -(center_x - np.max(center_x))
    w3 = center_y - np.min(center_y)
    w4 = -(center_y - np.max(center_y))
    
    all_dists = np.stack([w1,w2,w3,w4])
    wall_dists = np.min(all_dists,axis=0)
    wall_ids = np.argmin(all_dists,axis=0)
    wall_angles = np.array([180.,0.,270.,90.])
    wall_ego_angles = wall_angles[wall_ids]
    wall_ego_angles = (wall_ego_angles - angles)%360
    
    trial_data['wall_dists'] = wall_dists
    trial_data['wall_ego_angles'] = wall_ego_angles
    
    #polar
    
    slopes = center_y/center_x
    wy1 = slopes * np.min(center_x)
    wx1 = np.min(center_x)
    w1_dists = np.sqrt((wy1 - center_y)**2 + (wx1 - center_x)**2)
    
    wy2 = slopes * np.max(center_x)
    wx2 = np.max(center_x)
    w2_dists = np.sqrt((wy2 - center_y)**2 + (wx2 - center_x)**2)
    
    wx3 = np.min(center_y) / slopes
    wy3 = np.min(center_y)
    w3_dists = np.sqrt((wy3 - center_y)**2 + (wx3 - center_x)**2)
    
    wx4 = np.max(center_y) / slopes
    wy4 = np.max(center_y)
    w4_dists = np.sqrt((wy4 - center_y)**2 + (wx4 - center_x)**2)
    
    polar_all_dists = np.stack([w1_dists,w2_dists,w3_dists,w4_dists])
    polar_wall_dists = np.nanmin(polar_all_dists,axis=0)
    
    trial_data['polar_wall_dists'] = polar_wall_dists
    
    return trial_data


def calc_ebc_vals(trial_data,angle_bins=12,dist_bins=8):
    
    gr = 64
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2
    
    xcoords = np.linspace(np.min(center_x),np.max(center_x),gr,endpoint=False)
    ycoords = np.linspace(np.min(center_y),np.max(center_y),gr,endpoint=False)
    w1 = np.stack((xcoords,np.repeat(np.max(ycoords),gr)))
    w3 = np.stack((xcoords,np.repeat(np.min(ycoords),gr)))
    w2 = np.stack((np.repeat(np.max(xcoords),gr),ycoords))
    w4 = np.stack((np.repeat(np.min(xcoords),gr),ycoords))
    
    all_walls = np.concatenate((w1,w2,w3,w4),axis=1)
    
    wall_x = np.zeros((len(all_walls[0]),len(center_x)))
    wall_y = np.zeros((len(all_walls[1]),len(center_y)))
    
    for i in range(len(center_x)):
        wall_x[:,i] = all_walls[0] - center_x[i]
        wall_y[:,i] = all_walls[1] - center_y[i]
        
    wall_ego_angles = (np.rad2deg(np.arctan2(wall_y,wall_x))%360 - angles)%360
    wall_ego_dists = np.sqrt(wall_x**2 + wall_y**2)
    
    ref_angles = np.linspace(0,360,angle_bins,endpoint=False)
    radii = np.linspace(0,np.min((np.max(center_x)-np.min(center_x),np.max(center_y)-np.min(center_y)))/2.,dist_bins,endpoint=False)
    d_bins = np.digitize(wall_ego_dists,radii) - 1
    
    ebc_X = np.zeros((len(center_x),angle_bins,dist_bins))
    cutoff = np.min((np.max(center_x)-np.min(center_x),np.max(center_y)-np.min(center_y)))/2.
    
    for i in range(len(center_x)):
        for a in range(len(ref_angles)):
            diffs = np.abs(wall_ego_angles[:,i] - ref_angles[a])
            closest_pt = np.where(diffs==np.min(diffs))[0][0]
            if wall_ego_dists[closest_pt,i] < cutoff:
                ebc_X[i,a,d_bins[closest_pt,i]] = 1.
#                
#    import matplotlib.pyplot as plt
#    xedges = np.linspace(0,360,12)
#    yedges = np.arange(8)
#    spikes = np.zeros((12,8))
#    occ = np.zeros((12,8))
#    for i in range(len(center_x)):
#        heatmap = np.zeros((12,8))
#        for a in range(len(ref_angles)):
#            diffs = np.abs(wall_ego_angles[:,i] - ref_angles[a])
#            closest_pt = np.where(diffs==np.min(diffs))[0][0]
#            if wall_ego_dists[closest_pt,i] < cutoff:
#                occ[a,d_bins[closest_pt,i]] += 1.
#                spikes[a,d_bins[closest_pt,i]] += spike_data['ani_spikes'][i]
#                
#    heatmap = spikes/occ
#                
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='polar')
#    ax.set_theta_zero_location("N")
#    ax.set_yticks([])
#    fig.tight_layout(pad=2.5)
#
#    ax.pcolormesh(np.deg2rad(xedges),yedges,heatmap.T) 
#    
#    plt.show()
#    
#    return heatmap
            
    ebc_X = np.reshape(ebc_X,(len(center_x),angle_bins*dist_bins))

    return ebc_X

def ts_file_reader(ts_file):
    """reads the spike ASCII timestamp file and assigns timestamps to list"""
    
    #make a list for spike timestamps
    spike_list = []
    #read txt file, assign each entry to spike_list
    reader = csv.reader(open(ts_file,'r'),dialect='excel-tab')

    for row in reader:
        spike_list.append(int(row[0]))
                
    #return it!
    return spike_list

def create_spike_lists(trial_data,cluster_data):
    """makes lists of spike data"""

    #dictionary for spike data
    spike_data = {}

    #creates array of zeros length of spike_timestamps to create spike train
    spike_train = np.zeros(len(trial_data['spike_timestamps']))
    #array of zeros length of video timestamps for plotting/animation purposes
    ani_spikes = np.zeros(len(trial_data['timestamps']),dtype=np.int)
    timestamps = np.array(trial_data['timestamps'])
    #for each spike timestamp...
    for i in cluster_data['spike_list']:
        diff = np.abs(timestamps-i)
        ani_spikes[np.where(diff==np.min(diff))] += 1

        #find closest entry in high precision 'spike timestamps' list
        spike_ind = bisect.bisect_left(trial_data['spike_timestamps'],i)

        if spike_ind < len(spike_train):
            #add 1 to spike train at appropriate spot
            spike_train[spike_ind] = 1
    #find the video timestamp at the halfway point
    halfway_ind = bisect.bisect_left(cluster_data['spike_list'],trial_data['timestamps'][np.int(len(trial_data['timestamps'])/2)]) - 1

    spike_data['ani_spikes'] = ani_spikes
    spike_data['spike_train'] = spike_train
    cluster_data['halfway_ind'] = halfway_ind
    
    return spike_data, cluster_data