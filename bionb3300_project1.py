#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:15:53 2026

@author: wynnsu

BIONB 3300 Project 1 Part 1
Using Leaky-Integrate-Fire (LIF) neuron to simulate bar orientation computation
"""

import numpy as np
import matplotlib.pyplot as plt

# parameters
total_t = 50  # total time for simulation
dt = 1        # time step in ms
interval = 10 # interval of LGN neuron firing
w = 7         # weight
theta = 3    # threshold potential
tau = 5       # time constant in ms

# =============================================================================
# Layer 1 - Lateral Geniculate Nucleus (LGN)
# Each neuron fzires when light appears in the center of its receptive field
# 3*5 array
# =============================================================================

# possible bar angles
angles = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5])

# cells that will lit on at specific angles
bar_cells = [[(1,1), (1,2), (1,3)], # 0
             [(1,1), (1,2), (0,3)], # 22.5
             [(2,1), (1,2), (0,3)], # 45
             [(2,1), (1,2), (0,2)], # 67.5
             [(0,2), (1,2), (2,2)], # 90
             [(0,2), (1,2), (2,3)], # 112.5
             [(0,1), (1,2), (2,3)], # 135
             [(0,1), (1,2), (1,3)]] # 157.5

bars = np.zeros((len(angles), 3, 5)) # (bar, row, col)
 
# preset which LGN neurons will fire given angle orientation
for each in range(len(bar_cells)):
    for entry in bar_cells[each]:
        bars[each,entry[0],entry[1]] = 1

# setting up V1 best response angle
best_resp = np.zeros((3, 3, 5))
best_resp[0, 1, 1:4] = 1 # best response orientation for V1-H
best_resp[1, :, 2] = 1   # best response orientation for V1-V
best_resp[2, 2, 1] = 1   # best response orientation for V1-45
best_resp[2, 1, 2] = 1   # best response orientation for V1-45
best_resp[2, 0, 3] = 1   # best response orientation for V1-45

# LGN spikes - one spike every 10 ms for a lit cell
pulse = np.zeros(total_t)
for i in range(len(pulse)):
    if i%interval == 0:
        pulse[i] = 1

# =============================================================================
# Layer 2 - Visual Cortex Neurons (V1)
# Neurons whose receptive fields are orientation sensitive, 
# meaning that they fire when a light is in their receptive field, 
# but only if its is oriented in a specific direction. 
#
# Use the array of “center on” cells to construct 3 V1 (visual cortex 1) neurons 
# which are receptive to bars in horizontal, 45°, or vertical direction respectively. 
#
# These neurons should fire at 100 Hz when responding to their best angles, 
# and less on response to neighboring angles. 
# =============================================================================

# simulation
def simulate(angle):
    '''
    uses LIF neuron to simulate neural spikes due detection
    of specific bar angle orientation
    
    Parameters
    ----------
    angle : float
            user enter an angle as input.

    Returns
    -------
    v_all : 2d numpy array
        contains evolution of membrane potential of all 3 V1 neurons.
    spikes : 2d numpy array
        stores the time stamp at which each V1 neuron spikes.
    '''
    # find the corresponding index in list given angle
    idx = np.where(np.isclose(angles, angle))[0]
    assert len(idx) != 0,f'angle must be one of {angles.tolist()}'
    k = idx[0]
    
    # reshape the 2d bar image into a 1d 
    bar_vector = bars[k].reshape(-1) # shape is 15*1
    
    # reshape the v1 3d neural filters (3, 3, 5) best_resp into a 2d
    filters_matrix = best_resp.reshape(3, 15) # shape is 3*15

    # calculate ththee 3 neurons responses
    response = filters_matrix @ bar_vector # shape is 3*1 (3*15 @ 15*1)
    print(response)

    #calculate inputs for each V1 neuron
    inp = []
    for i in range (len(response)):
        inp.append(response[i] * pulse)
    inp = np.array(inp)
    
    v_all = np.zeros((3,total_t))
    spikes = np.zeros((3,total_t))
    v_m = np.zeros(3)
    
    # for all V1 neurons, compute voltage & spikes given inputs
    for t in range(total_t):
        v_m = v_m + dt/tau*((-v_m + w*inp[:,t])) # euler's method
        v_all[:, t] = v_m
        
        # loop through 3 V1 neurons - if they pass theta, then fire & reset
        for neuron_index in range(3):
            if v_m[neuron_index] >= theta:
                spikes[neuron_index,t] = 1
                v_m[neuron_index] = 0
                
    return v_all,spikes

if __name__ == "__main__":
    
    # test one specific angle
    angle = 90 ### change this if want to try a different angle
    
    v1_all = simulate(angle)[0]
    
    plt.figure()
    plt.title(f'V1 horizontal neuron, given {angle}°')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.plot(v1_all[0])
    
    plt.figure()
    plt.title(f'V1 vertical neuron, given {angle}°')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.plot(v1_all[1])
    
    plt.figure()
    plt.title(f'V1 slant/45° neuron, given {angle}°')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.plot(v1_all[2])
    
    # testing all angles
    '''
    for angle in angles:
        
        v1_all = simulate(angle)[0]
        
        plt.figure()
        plt.title(f'V1 horizontal neuron, Given {angle}°')
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential")
        plt.plot(v1_all[0])
        
        plt.figure()
        plt.title(f'V1 vertical neuron, given {angle}°')
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential")
        plt.plot(v1_all[1])
        
        plt.figure()
        plt.title(f'V1 slant/45° neuron, given {angle}°')
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential")
        plt.plot(v1_all[2])
    '''