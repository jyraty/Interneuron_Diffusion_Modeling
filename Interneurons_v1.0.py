# -*- coding: utf-8 -*-
"""
@author: jyraty
Copyright (C) 2026 Jean-Yves Raty

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but **without any warranty**; without even the implied warranty of
**merchantability** or **fitness for a particular purpose**.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import classes
import routines
import jump
import move_neurons
import compute_force
import tails
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mytk
import threading
import cv2





### global parameters
maxpoints = 10
maxneurons= 400 
maxpatch=100
maxmask=10
intervalle_plot=5000
nmaxtails=8
nmaxlevels=3
n_batches=0
delay_batch=0
print(' let s start')

### data   box ###
box_x0 = 0.0
box_y0 = 0.0
box_n_neurons=0
box_friction=0
box_friction_min=0.0
box_friction_max=0.0
box_slipstick_profile=""
box_slipstick_min=0.0
box_slipstick_max=0.0
box_slipstick_pause=0
box_friction_profile=""
box_jumprate=1.0
box_jumprate_min=0.0
box_jumprate_max=0.0
box_jumprate_profile=""
box_jumpamp=1.0
box_jumpamp_min=0.0
box_jumpamp_max=0.0
box_jumpamp_profile=""
box_tailrate=1.0
box_tailrate_min=0.0
box_tailrate_max=0.0
box_tailrate_profile=""
box_branchrate=1.0
box_branchrate_min=0.0
box_branchrate_max=0.0
box_branchrate_profile=""
box_MCFIELD=""
box_const_force=0.0
box_reset_vel=0.0
box_timestep=0
box_plotfield="pot"
box_temperature=1.0
box_accept=1.0
box_reject=1.0
gc_split=0.0
L2=0.0 
L3=0.0 
NK_freq=0.0

box_PBC=False
box_n_neurons=0
neighbours_list_time=5000
neighbours_list_rcut=20.0
mot=""





### data   neurons ###

neurons_maxpoints=np.zeros((maxneurons), dtype=int)
neurons_maxpoints[:]=maxpoints
neurons_activation = np.zeros((maxneurons), dtype=float)  # default activation level
neurons_x=np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_y=np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_x_ini=np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_y_ini=np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_xprev=np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_yprev=np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_npoints=np.zeros((maxneurons), dtype=int)
neurons_npoints_ini=np.zeros((maxneurons), dtype=int)
neurons_nneighbours=np.zeros((maxneurons), dtype=int)
neurons_neighbours_id=np.zeros((maxneurons,maxneurons), dtype=int)
neurons_tail_ntot=np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_nbranches=np.zeros((maxneurons), dtype=float)
neurons_ncontacts=np.zeros((maxneurons), dtype=float)
active_contacts=np.zeros((maxneurons,maxneurons), dtype=int)
neurons_gc_split=np.zeros((maxneurons), dtype=float)
neurons_NK_freq=np.zeros((maxneurons), dtype=float)
neurons_L2=np.zeros((maxneurons), dtype=float)
neurons_L3=np.zeros((maxneurons), dtype=float)






neurons_size           = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_size[:]        = 1.0
neurons_size_ini       = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_size_ini[:]    = 1.0
neurons_nk2            = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_nk2[:]         = 0
neurons_nk2_ini        = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_nk2_ini[:]     = 0.0
neurons_nk3            = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_nk3[:]         = 0.0
neurons_nk3_ini        = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_nk3_ini[:]     = 0.0
neurons_k2             = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_k2_ini         = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_k3             = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_k3_ini         = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_k2_id          = np.zeros((maxneurons,nmaxtails,maxpoints, 2), dtype=int)
neurons_k2_id_ini      = np.zeros((maxneurons,nmaxtails,maxpoints, 2), dtype=int)
neurons_k3_id          = np.zeros((maxneurons,nmaxtails,maxpoints, 3), dtype=int)
neurons_k3_id_ini      = np.zeros((maxneurons,nmaxtails,maxpoints, 3), dtype=int)
neurons_length         = np.zeros((maxneurons,nmaxtails,maxpoints, maxpoints), dtype=float)
neurons_length_ini     = np.zeros((maxneurons,nmaxtails,maxpoints, maxpoints), dtype=float)
neurons_cosangle       = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_cosangle_ini   = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_mass           = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_mass_ini       = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_force          = np.zeros((maxneurons,nmaxtails,maxpoints,2), dtype=float)
neurons_velocity       = np.zeros((maxneurons,nmaxtails,maxpoints,2), dtype=float)
neurons_repulsion      = np.zeros((maxneurons), dtype=float)
neurons_r_width        = np.zeros((maxneurons), dtype=float)
neurons_r_cut          = np.zeros((maxneurons), dtype=float)
neurons_is_repulsive   = np.zeros((maxneurons), dtype=int)
neurons_pause          = np.zeros((maxneurons), dtype=int)

neurons_nr_activetails = np.zeros((maxneurons), dtype=int)
neurons_id_activetail  = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_active    = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_level     = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_number    = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_ntot      = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_nchildren  = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_hasparent = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_node      = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_node[:,:]=3

neurons_jmp_dyn         = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=int)
neurons_jmp_dyn_ini     = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=int)
neurons_jmp_rate        = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_jmp_rate_ini    = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_jmp_amp         = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_jmp_amp_ini     = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_jmp_ang         = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float) 
neurons_jmp_ang_ini     = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float) 
neurons_tail_angle_start_min =np.zeros((maxneurons), dtype=float)
neurons_tail_angle_start_max =np.zeros((maxneurons), dtype=float)
neurons_tail_x          = np.zeros((maxneurons,maxpoints), dtype=float)
neurons_tail_y          = np.zeros((maxneurons,maxpoints), dtype=float)
neurons_tail_xprev      = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_tail_yprev      = np.zeros((maxneurons,nmaxtails,maxpoints), dtype=float)
neurons_tail_velocity   = np.zeros((maxneurons,nmaxtails,maxpoints,2), dtype=float)

neurons_tail_nmax       = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_nmax[:]    = maxpoints
neurons_tail_l          = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_l[:]       = 0.0   
neurons_tail_l0         = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_l0[:]      = 0.0
neurons_tail_angle_start_min = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_angle_start_min[:] = 0.0
neurons_tail_angle_start_max = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_angle_start_max[:] = 0.0



neurons_tail_angle      = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_angle[:]   = 0.0
neurons_tail_mass_max   = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_mass_max[:]=0.0
neurons_tail_active     = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_tail_active[:]  = 0
neurons_tail_create_rate       = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_create_rate[:]    = 0.0
neurons_tail_delete_rate       = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_delete_rate[:]    = 0.0
neurons_tail_create_rate_ini       = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_create_rate_ini[:]    = 0.0
neurons_tail_delete_rate_ini       = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_delete_rate_ini[:]    = 0.0
neurons_tail_create_multfactor = np.zeros((maxneurons,nmaxtails), dtype=float)
neurons_tail_delete_multfactor = np.zeros((maxneurons,nmaxtails), dtype=float)






neurons_bulge_active = np.zeros((maxneurons,nmaxtails), dtype=int)
neurons_bulge_pc=np.zeros((maxneurons), dtype=float)
neurons_bulge_minpoints=np.zeros((maxneurons), dtype=int)
neurons_bulge_create_rate=np.zeros((maxneurons), dtype=float)
neurons_bulge_delete_rate=np.zeros((maxneurons), dtype=float)
neurons_bulge_mass=np.zeros((maxneurons,3), dtype=float)
neurons_bulge_size=np.zeros((maxneurons,3), dtype=float)
neurons_old_mass = np.zeros((maxneurons,3), dtype=float)
neurons_old_size = np.zeros((maxneurons,3), dtype=float)

neurons_times = np.zeros((maxneurons,3), dtype=float)


tail_nmax = np.zeros((nmaxlevels), dtype=int)
tail_l = np.zeros((nmaxlevels), dtype=float)                            
tail_l0 = np.zeros((nmaxlevels), dtype=float)
tail_k2 = np.zeros((nmaxlevels), dtype=float)
tail_k3 = np.zeros((nmaxlevels), dtype=float)
tail_angle_start_min = np.zeros((nmaxlevels), dtype=float)
tail_angle_start_max = np.zeros((nmaxlevels), dtype=float)
tail_angle= np.zeros((nmaxlevels), dtype=float)
tail_mass_max= np.zeros((nmaxlevels), dtype=float)
tail_create_rate= np.zeros((nmaxlevels), dtype=float)
tail_delete_rate= np.zeros((nmaxlevels), dtype=float)
tail_create_multfactor= np.zeros((nmaxlevels), dtype=float)
tail_delete_multfactor= np.zeros((nmaxlevels), dtype=float)




global fig
global canvas




n_glials=0
n_patches=0
n_masks=0
n_levels=0

patch_params = np.zeros((maxpatch,100), dtype=float)
patch_type   = list(str('none ') * maxpatch)  
patch_shape  = list(str('none ') * maxpatch)  

mask_params = np.zeros((maxmask,100), dtype=float)

mask_type   = list(str('none ') * maxpatch)  
mask_shape  = list(str('none ') * maxpatch)  
mask_file   = list(str('none ') * maxpatch)   

with open("default_data.txt", "r", encoding="utf-8") as f:
    lines=f.readlines()
    #print(" nlines "+str(len(lines)))
    il = 0

    while il < len(lines):    
        parts = lines[il].split()
        if "#" in lines[il]:
            il+=1
            continue
        
        #print(" il "+str(il)+" "+str(lines[il]))
        parts = lines[il].split()

        
        # Vérifier si la ligne contient "mur"
        if "universe_dimension" in lines[il]:
            try:
                # Récupérer les deux premières valeurs après "mur"
                
                xf = float(parts[1])
                yf = float(parts[2])
                nx = int(xf)
                ny =  int(yf)
                print(f"xf, yf ="+str(xf)+" "+str(yf))
                # create universe
                box_xf=xf
                box_yf=yf
                box_nx=nx
                box_ny=ny
                box_dx=box_xf/box_nx
                box_dy=box_yf/box_ny
                ymin_time=1.0
                ymax_time=yf
                

                #print(" dx "+str(dx))
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
        if "timestep" in lines[il]:
            try:              
                box_timestep = float(parts[1])
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")    
        if "movie" in lines[il]:
            try:              
                movie_fps = int(parts[1])
                intervalle_plot = int(parts[2])

                il+=1
                print(f'Movie :every {intervalle_plot} with {movie_fps} fps ')
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")    
        if "potential" in lines[il]:
            try:
                # Récupérer les deux premières valeurs après "mur"
                
                vi = float(parts[1])
                vf = float(parts[2])
                pot_field = np.zeros((nx, ny), dtype=float)
                if "const_force" in lines[il]:
                    try:
                        box_const_force=float(parts[4])
                    except (ValueError, IndexError):
                        print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")    
                for i in range(ny):
                    pot_field[:,i]=vf+(i)*(vi-vf)/(1.0*(ny-1))


                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")

        if "walls" in lines[il]:
            try:
                # Récupérer les deux premières valeurs après "mur"
                
                wheight = float(parts[1])
                wsigma =  float(parts[2])                
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
        if "friction" in lines[il]:
            try:
                fr_params = np.zeros((10), dtype=float)
                fr_type= str(parts[1])
                print(f' friction {fr_type}')
                n=len(parts)
                for j in range (2,n):                    
                    fr_params[j-2]=float(parts[j])
                    print(f' j param {j} {fr_params[j-2]}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}") 
        if "slipstick" in lines[il]:
            try:
                slipstick_params = np.zeros((10), dtype=float)
                slipstick_type= str(parts[1])
                box_slipstick_pause= int(parts[2])
                box_slipstick_amplitude=float(parts[3])
                print(f' slipstick {slipstick_type}')
                n=len(parts)
                for j in range (4,n):                    
                    slipstick_params[j-4]=float(parts[j])
                    print(f' {n} j slipstickparam {j} {slipstick_params[j-4]}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
                
        if "jumpamp" in lines[il]:
            try:
                jumpamp_params = np.zeros((10), dtype=float)
                jumpamp_type= str(parts[1])
                print(f' jumpamp {jumpamp_type}')
                n=len(parts)
                for j in range (2,n):                    
                    jumpamp_params[j-2]=float(parts[j])
                    print(f' jumpamp j param {j} {jumpamp_params[j-2]}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
                
        if "jumprate" in lines[il]:
            try:
                jumprate_params = np.zeros((10), dtype=float)
                jumprate_type= str(parts[1])
                print(f'jumprate {jumprate_type}')
                n=len(parts)
                for j in range (2,n):                    
                    jumprate_params[j-2]=float(parts[j])
                    print(f' jumprate j param {j} {jumprate_params[j-2]}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
                
        if "MCFIELD" in lines[il]:
            try:
                mcfield=str(parts[1])
                box_mcfield=mcfield
                mcfield_sign=str(parts[2])
                print(f'MC moves based on field :{mcfield} {mcfield_sign}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
        
        
        if "temperature" in lines[il]:
            try:
                box_temperature = float(parts[1])
                print(f' T {box_temperature}')
                n=len(parts)
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
        if "PBC" in lines[il]:
            try:
                intin=int(parts[1])
                if (intin ==0 ):
                    box_PBC=False
                else:
                    box_PBC=True
               
                print(f' PBC {box_PBC}')
                n=len(parts)
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
        if "MASK" in lines[il]:
             try:
                 n=len(parts)
                 n_masks += 1
                 mask_type[n_masks-1]=str(parts[1])
                 mask_file[n_masks-1]=str(parts[2])
                 mask_shape[n_masks-1]=str(parts[3])
                 print(f' MASK nr {n_masks} {mask_type[n_masks-1]} \
                       {mask_file[n_masks-1]}')

                 for j in range (4,n):                    
                     mask_params[n_masks-1,j-4]=float(parts[j])
                     print(f' masks param {j} {mask_params[n_masks-1,j-2]}')
                 il+=1
             except (ValueError, IndexError):      
                 print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
              
        if "n_neurons" in lines[il]:
            try:
                n_neurons = int(parts[1])
                n_neurons_ini=n_neurons
                box_n_neurons=n_neurons
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")    
        if "batches" in lines[il]:
            try:
                n_batches = int(parts[1])
                delay_batch=int(parts[2])
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")         

        if "neuron_data" in lines[il]:
            try:
                nshape=str(parts[1])
                xmin = float(parts[2])
                xmax = float(parts[3])                            
                ymin = float(parts[4])
                ymax = float(parts[5])
                npoints = int(parts[6])
                anisotropy = float(parts[7])
                global length
                length = np.zeros((npoints, npoints), dtype=float)
                angle = np.zeros(npoints, dtype=float)
                mass = np.zeros(npoints, dtype=float)
                jmp_dyn  =np.zeros((npoints), dtype=int)
                jmp_rate =np.zeros((npoints), dtype=float)
                jmp_amp =np.zeros((npoints), dtype=float)
                jmp_ang =np.zeros((npoints), dtype=float)
                size =np.zeros((npoints), dtype=float)
                k2 = np.zeros(npoints, dtype=float)
                k2_id = np.zeros((npoints, 2), dtype=int)
                k3 = np.zeros(npoints, dtype=float)
                k3_id = np.zeros((npoints, 3), dtype=int) 
                il+=1
                
                for i in range(npoints):
                    parts = lines[il].split()        
                    mass[i]=float(parts[1])
                    jmp_dyn[i]=int(parts[2])
                    jmp_rate[i]=float(parts[3])
                    jmp_amp[i]=float(parts[4])
                    jmp_ang[i]=float(parts[5])
                    size[i]=float(parts[6])
                    print(f"mass jmp jmpr jmpa {mass[i]} {jmp_dyn[i]} {jmp_rate[i]} {jmp_amp[i]} {jmp_ang[i]} {size[i]}")
                    il+=1
                    #print(f"masses "+str(mass[:]))
                   
                nk2=0
                if "nk2" in lines[il]:
                    parts = lines[il].split()
                    nk2=int(parts[1])
                    il+=1

                    #print(f"nk2 "+str(nk2))
                    for ik2 in range(nk2):
                        #print(f"il "+str(il))
                        #print(f"ik2 "+str(ik2))
                        parts = lines[il].split()     
                        #print(f"parts "+str(parts[:]))
                        i1=int(parts[0])
                        i2=int(parts[1])
                        k2_id[ik2,0]=i1
                        k2_id[ik2,1]=i2
                        #print(f"i1 i2 "+str(i1)+" "+str(i2))
                        length[i1-1,i2-1]=float(parts[2])
                        k2[ik2]=float(parts[3])
                        il+=1

                nk3=0
                if "nk3" in lines[il]:
                    #print(f"in nk3")
                    parts = lines[il].split()  
                    nk3 = int(parts[1])
                    #print(f"nk3 "+str(nk3))
                    il+=1
                    for ik3 in range (nk3):
                        parts = lines[il].split() 
                        i1=int(parts[0])
                        i2=int(parts[1])
                        i3=int(parts[2])  
                        k3_id[ik3,0]=i1
                        k3_id[ik3,1]=i2
                        k3_id[ik3,2]=i3
                        angle[ik3]=float(parts[3])
                        k3[ik3]=float(parts[4])
                        #print(f"parts "+str(parts[:]))  
                        il+=1
            except (ValueError, IndexError):
                print(f" last OK line :  {lines[il-1]}")
                print(f"⚠️ Ligne mal formée ignorée : {il} {lines[il]}") 
        if "neighbours" in lines[il]:
            try:
                n=len(parts)
                neighbours_list_time=int(parts[1])
                neighbours_list_rcut=float(parts[2])
                print(f' neighbours list every {parts[1]} step with rcut  {parts[2]}  ')
                il+=1
            except(ValueError, IndexError):
                print(f" last OK line :  {lines[il-1]}")
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")

        if "tail_data" in lines[il]:
            try:
                # TAIL_DATA level nmax, %l,    l0  k2,  k3    angstartmin, angstartmax, angle,    mass_max, createrate  deleterate  create_multfactor delete_multfcator
                #tail_data   1     4     0.5   3.0  3   7.0     40.0        40.0         40.0    3.0      0.001      0.001    1.0  1.0
                level=int(parts[1])
                if (level > n_levels): # initially n_levels=0
                    n_levels=level
                    n_tails = int(2**(level-1)) # at each level one branch can subdivide into two here level 1 is actually 2 branches, level 2 is 4 branches etc


                tail_nmax[level-1] = int(parts[2])
                tail_l[level-1] = float(parts[3])                            
                tail_l0[level-1] = float(parts[4])
                tail_k2[level-1] = float(parts[5])
                tail_k3[level-1] = float(parts[6])
                tail_angle_start_min[level-1] = float(parts[7])
                tail_angle_start_max[level-1] = float(parts[8])
                tail_angle[level-1] = float(parts[9]) 
                tail_mass_max[level-1] = float(parts[10])
                tail_create_rate[level-1] = float(parts[11])
                tail_delete_rate[level-1] = float(parts[12])
                tail_create_multfactor[level-1] = float(parts[13])
                tail_delete_multfactor[level-1] = float(parts[14])
                print(f"tail level {level} { tail_nmax[level-1]} { tail_l[level-1]}")
                il+=1
            except(ValueError, IndexError):
                print(f" last OK line :  {lines[il-1]}")
                print(f"⚠️ Ligne mal formée ignorée : {il} {lines[il]}")
        if "tailrate" in lines[il]:
            try:
                tailrate_params = np.zeros((10), dtype=float)
                tailrate_type= str(parts[1])
                print(f' tailrate {tailrate_type}')
                n=len(parts)
                for j in range (2,n):                    
                    tailrate_params[j-2]=float(parts[j])
                    print(f' tailrate j param {j} {tailrate_params[j-2]}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
       
        if "tail_branch_interaction" in lines[il]:
            try:
                tail_branch_interaction_params = np.zeros((10), dtype=float)
                n=len(parts)
                for j in range (1,n):                    
                    tail_branch_interaction_params[j-1]=float(parts[j])
                    print(f' tb interactions {parts[1]} {parts[2]} {parts[3]} {parts[4]}  ')
                il+=1
            except(ValueError, IndexError):
                print(f" last OK line :  {lines[il-1]}")
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
                
        if "branchrate" in lines[il]:
            try:
                branchrate_params = np.zeros((10), dtype=float)
                branchrate_type= str(parts[1])
                print(f' {branchrate_type}')
                n=len(parts)
                for j in range (2,n):                    
                    branchrate_params[j-2]=float(parts[j])
                    print(f' branchrate j param {j} {branchrate_params[j-2]}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")
        if "bulge_data" in lines[il]:
              try:
                  #print(f"bulge")
                  #pc is pc of soma going into bulge
                  bulge_pc   = float(parts[1]) 
                  bulge_minpoints =int(parts[2])
                  bulge_create_rate = float(parts[3])
                  bulge_delete_rate = float(parts[4])
                  bulge_mult_factor = float(parts[4])
                  #bulge_amp  # for this moment leave out this possibilioty

                  print(f"bulge {bulge_pc} {bulge_create_rate} {bulge_delete_rate}")
                  il+=1
              except(ValueError, IndexError):
                  print(f" last OK line :  {lines[il-1]}")
                  print(f"⚠️ Ligne mal formée ignorée : {il} {lines[il]}")                      
        if "interaction_att" in lines[il]:
            try:
                parts = lines[il].split() 
                fraction_att = float(parts[1])
                interaction_att = float(parts[2])
                r_width_att = float(parts[3])
                r_cut_att = float(parts[4])
                #print(f'att widtch cut {interaction_att} {r_width_att} {r_cut_att}')
                il+=1
                #time.sleep(10)
            except (ValueError, IndexError):
                print(f" last OK line :  {lines[il-1]}")
                print(f"⚠️ Ligne mal formée ignorée : {il} {lines[il]}") 
        if "interaction_rep" in lines[il]:
            try:
                parts = lines[il].split() 
                fraction_rep = float(parts[1])
                interaction_rep = float(parts[2])
                r_width_rep = float(parts[3])
                r_cut_rep = float(parts[4])
                #print(f'rep widtch cut {interaction_rep} {r_width_rep} {r_cut_rep}')
                il+=1
                #time.sleep(10)
            except (ValueError, IndexError):
                print(f" last OK line :  {lines[il-1]}")
                print(f"⚠️ Ligne mal formée ignorée : {il} {lines[il]}")
                

        if "n_line_pert" in lines[il]:
            try:
                parts = lines[il].split() 
                n_line_pert = int(parts[1])
                #print(f"nr line pert  "+str(n_line_pert))
                il+=1

                parts = lines[il].split()
                l_anisotropy=float(parts[0])
                l_length=float(parts[1])
                l_height=float(parts [2])
                l_width=float(parts[3])
                l_slength=float(parts[4])
                l_sheight=float(parts[5])
                l_swidth=float(parts[6])
                #print(f"width, swidth "+str(l_width)+str(l_swidth))
                il+=1
                if "biased" in lines[il]:
                    l_biased=True
                else:
                    l_biased=False
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}") 
        if "line_pert_profile" in lines[il]:
            try:
                parts = lines[il].split() 
                line_pert_profile= str(parts[1])
                line_pert_params = np.zeros((10), dtype=float)
                print(f' {line_pert_profile}')
                n=len(parts)
                for j in range (2,n):                    
                    line_pert_params[j-2]=float(parts[j])
                    print(f' line_pert_params j param {j} {line_pert_params[j-2]}')
                print(f' {line_pert_profile}')
                il+=1
            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}")                     
        if "n_glials" in lines[il]:
            try:
                parts = lines[il].split() 
                n_glials = int(parts[1])
                #print(f"nr line pert  "+str(n_line_pert))     
                gl_length=float(parts[2])
                gl_height=float(parts[3])
                gl_width=float(parts[4])
                #print(f"width, swidth "+str(l_width)+str(l_swidth))
                il+=1

            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}") 
        
        if "SPEEDMEASURE" in lines[il]:
            try:
                parts = lines[il].split() 
                ymin_time = float(parts[1])
                ymax_time = float(parts[2])
                print(f"SPEEDMEASURE {ymin_time} {ymax_time} " )
                il+=1       

            except (ValueError, IndexError):
                print(f"⚠️ Ligne mal formée ignorée : {lines[il]}") 
    il+=1                

#create perturbation field
pert_field= np.zeros((nx, ny), dtype=float)

rng=np.random.default_rng()
rng_vals=rng.random(10*10000)
  
routines.create_l_perturbation(n_line_pert,l_anisotropy,l_length,l_height,\
                               l_width,l_slength,l_sheight,l_swidth, l_biased, \
                               nx,ny,\
                               pert_field,box_dx, box_dy, \
                               box_xf, box_yf, box_PBC, line_pert_profile,line_pert_params, rng_vals)


walls_field= np.zeros((nx, ny), dtype=float)      
routines.create_wall_field(wheight, wsigma, walls_field, box_dx, box_dy, box_nx, box_ny, box_PBC ) 

affinity_field= np.zeros((nx, ny), dtype=float)      
routines.create_affinity_field(affinity_field, box_dx, box_dy, box_nx, box_ny, box_PBC )

glials_field= np.zeros((nx, ny), dtype=float)      
routines.glials_field(n_glials,gl_length,gl_height, gl_width, glials_field,box_dx, box_dy, box_nx, box_ny)


friction_field=np.zeros((nx, ny), dtype=float) 
routines.create_friction_field(fr_type, fr_params, friction_field,box_dx, box_dy, box_nx, box_ny, box_friction_profile, \
                          box_friction_min,box_friction_max) 

slipstick_field=np.zeros((nx, ny), dtype=float) 
routines.create_slipstick_field(slipstick_type, slipstick_params, slipstick_field,box_dx, box_dy, box_nx, box_ny, box_slipstick_profile)

jumpamp_field=np.zeros((nx, ny), dtype=float) 
routines.create_jumpamp_field(jumpamp_type, jumpamp_params,jumpamp_field,box_dx, box_dy, box_nx, box_ny, box_jumpamp_profile, \
                              box_jumpamp_min, box_jumpamp_max) 

                            
jumprate_field=np.zeros((nx, ny), dtype=float) 
routines.create_jumprate_field(jumprate_type, jumprate_params, jumprate_field, box_dx, box_dy, box_nx, box_ny, box_jumprate_profile)

tailrate_field=np.zeros((nx, ny), dtype=float) 
routines.create_tailrate_field(tailrate_type, tailrate_params, tailrate_field, box_dx, box_dy, box_nx, box_ny, box_tailrate_profile, \
                              box_tailrate_min, box_tailrate_max) 


branchrate_field=np.zeros((nx, ny), dtype=float) 
routines.create_branchrate_field(branchrate_type, branchrate_params, branchrate_field,box_dx, box_dy, box_nx, box_ny, box_branchrate_profile, \
                              box_branchrate_min, box_branchrate_max) 


for i in range (n_masks):
    if (mask_type[i] == 'Potential' ):
        routines.add_mask(pot_field, mask_type[i], mask_file[i], mask_shape[i], mask_params[i,:],box_nx, box_ny)
    if (mask_type[i] == 'Friction' ):
        routines.add_mask(friction_field, mask_type[i], mask_file[i],mask_shape[i],mask_params[i,:],box_nx, box_ny)
    if (mask_type[i] == 'Affinity' ):
        routines.add_mask(affinity_field, mask_type[i], mask_file[i],mask_shape[i],mask_params[i,:],box_nx, box_ny)      
    if (mask_type[i]=='Potential_on_Affinity'):
        routines.mask_potentialOnAffinity(affinity_field, mask_file[i],box_nx, box_ny)
   
#create our neurons
#before calling the routine create N_Neuorn instances of class neuron

neurons = [classes.Neuron(i) for i in range(n_neurons)]



#initiate trajectories
tx=classes.Trajectory(500, maxneurons)  #max time maxneurons
tx.index=0
ty=classes.Trajectory(500, maxneurons) 
ty.index=0
tx.plot1=0
tx.plot2=0

diffx=classes.Trajectory(500, maxneurons) 
diffx.index=0
diffy=classes.Trajectory(500, maxneurons) 
diffy.index=0
difftot=classes.Trajectory(500, 1) 
difftot.index=0
diffx.plot1=0
diffx.plot2=0
diffy.plot1=0
diffy.plot2=0
difftot.plot1=0
difftot.plot2=0


# Example: Activate each neuron with input = index value
#for i, neuron in enumerate(neurons):
#    neuron.activate(i)
# Print neurons
#for neuron in neurons:
#    print(neuron)
    
############################################################################

#note V4 : all neurons are created, but may appear in batches
routines.create_neurons(n_neurons,nshape,xmin, xmax, ymin, ymax,n_levels, n_tails, npoints, anisotropy, nk2, nk3, \
                        interaction_att, fraction_att, r_width_att, r_cut_att,\
                        interaction_rep, fraction_rep, r_width_rep, r_cut_rep, box_yf, length, angle, mass, size, \
                        jmp_dyn, jmp_rate, jmp_amp, jmp_ang, k2, k2_id,k3,k3_id,  \
                        neurons_maxpoints,neurons_x,neurons_y,neurons_x_ini, neurons_y_ini,neurons_npoints,neurons_npoints_ini, \
                        neurons_tail_active,neurons_tail_ntot, neurons_tail_level,  neurons_tail_number, neurons_tail_nchildren, neurons_tail_hasparent,\
                        neurons_repulsion,neurons_r_width,neurons_r_cut, \
                        neurons_nk2, neurons_nk2_ini, neurons_nk3, neurons_nk3_ini, neurons_mass, neurons_mass_ini, neurons_size, neurons_size_ini, \
                        neurons_jmp_dyn, neurons_jmp_dyn_ini ,neurons_jmp_rate,neurons_jmp_rate_ini,neurons_jmp_amp,neurons_jmp_amp_ini, \
                        neurons_jmp_ang, neurons_jmp_ang_ini, neurons_k2_id, neurons_k2_id_ini, neurons_k2, neurons_k2_ini, \
                        neurons_length, neurons_length_ini,neurons_k3_id, neurons_k3_id_ini, neurons_k3, neurons_k3_ini, \
                        neurons_cosangle, neurons_cosangle_ini,neurons_velocity,  neurons_is_repulsive)

routines.add_tail_data(n_neurons, n_tails, neurons_tail_active,neurons_tail_ntot,neurons_npoints_ini,                 
                        neurons_tail_hasparent, neurons_tail_nchildren, neurons_tail_level)
                       

    
routines.add_bulge_data(n_neurons, bulge_pc, bulge_minpoints,bulge_create_rate, bulge_delete_rate,\
                   neurons_bulge_pc, neurons_bulge_minpoints, neurons_bulge_create_rate,neurons_bulge_delete_rate, neurons_bulge_mass,\
                   neurons_mass_ini, neurons_bulge_size,neurons_size_ini)
            

############################################################################  
    
#time.sleep(10)

     
# plot potential in mp window
potvalue= np.zeros((nx, ny), dtype=float)
potvalue[:,:]=pot_field[:,:]
potvalue[:,:]+=walls_field[:,:]
potvalue[:,:]+=pert_field[:,:]
potvalue[:,:]+=glials_field[:,:]
potvalue[0,:]=potvalue[nx-1,:]
force = np.zeros((nx, ny), dtype=float)
accept_field= np.zeros((nx, ny), dtype=float) 
accept_field[:,:]=jumpamp_field[:,:]
potmax=np.max(potvalue)


if (box_mcfield=='Friction'):
    accept_field[:,:]=friction_field[:,:]    
if (box_mcfield=='Potential'):
    accept_field[:,:]=potvalue[:,:]
if (box_mcfield=='Jumpamp'):
    accept_field[:,:]=jumpamp_field[:,:]
if (box_mcfield=='Jumprate'):
    accept_field[:,:]=jumprate_field[:,:]
if (box_mcfield=='Tailrate'):
    accept_field[:,:]=tailrate_field[:,:]
if (box_mcfield=='Branchrate'):
    accept_field[:,:]=branchrate_field[:,:] 
if (box_mcfield=='Affinity'):
    accept_field[:,:]=affinity_field[:,:] 
print(f'MCFIELD = {box_mcfield}')
for i in range (nx):
    for j in range(ny):
        if (accept_field[i,j] <=0 ):
            accept_field[i,j]=0                    
    
###########################################################################
#routines.create_plot(potvalue, neurons,box)
###########################################################################
xmin=box_x0
xmax=box_xf
nx=box_nx
dx=xmax/nx
ymin=box_y0
ymax=box_yf
ny=box_ny
dy=ymax/ny
    #print(f'xmin, xmax, nx {xmin}, {xmax}, {nx}')
    #print(f'ymin, ymax, ny {ymin}, {ymax}, {ny}')
xpoints = np.linspace(xmin, xmax, nx)
ypoints = np.linspace(ymin, ymax, ny)

print(f'{len(xpoints)} {len(ypoints)} {len(potvalue[:,1])}')
gx,gy = np.gradient(potvalue, xpoints,ypoints, edge_order=1)
print("grad_x shape =", gx.shape)
print("grad_y shape =", gy.shape)

shapes = []


def get_unique_filename(filename):
    """
    Returns a filename that does not already exist.
    Example:
        movie.mp4      -> movie.mp4
        movie.mp4      -> movie_1.mp4 (if movie.mp4 exists)
        movie.mp4      -> movie_2.mp4 (if movie.mp4 and movie_1.mp4 exist)
    """
    path = Path(filename)

    if not path.exists():
        return str(path)

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return str(candidate)
        i += 1


def startsim(root2,fig, ax,canvas,stop_event, win_visible):
    accept_arr=np.zeros(maxneurons, dtype=int)
    reject_arr=np.zeros(maxneurons, dtype=int)
    box_accept=1.0
    box_reject=1.0

    gc_split=0.0
    L2=0.0  
    L3=0.0
    NK_freq=0.0


    pmax=2.0
    pmin=0.0
    print(f' pmax pmin forced {pmax} {pmin}')
    neurons_xprev[:,:,:]=neurons_x[:,:,:]
    neurons_yprev[:,:,:]=neurons_y[:,:,:]

  

    VIDEO_FILENAME =  get_unique_filename("simulation.mp4") 
    writer = imageio.get_writer(VIDEO_FILENAME, fps=movie_fps)


    n_neurons=0 

    for itime in range(50000000):
        #this is where the number pf neurons can be increased in batches
        if (np.mod(itime,delay_batch) == 0):
            n_neurons=n_neurons+n_batches
            if (n_neurons > n_neurons_ini):
                n_neurons=n_neurons_ini
        if (stop_event.is_set()):
            ##writer.release()
            writer.close()
            break 
        if (np.mod(itime,neighbours_list_time) == 0):
            routines.compute_neighbours(n_neurons, neurons_x, neurons_y, box_xf, box_yf, box_PBC, neighbours_list_rcut, neurons_nneighbours, neurons_neighbours_id)



        if (np.mod(itime,intervalle_plot) == 0):
            acc=box_accept/(box_accept+box_reject)
            gc_split=sum(neurons_gc_split) /(1.0*n_neurons)
            L2=sum(neurons_L2)  /(1.0*n_neurons)
            L3=sum(neurons_L3) /(1.0*n_neurons)
            NK_freq=sum(neurons_NK_freq)  /(1.0*n_neurons)
         
            print(f' {itime} {n_neurons} rates {box_accept} {box_reject} {acc:.3f} NK {NK_freq:9.7f} GC {gc_split:10.7f} L2 {L2:10.7f} L3 {L3:10.7f}    ')
            #print(f'visible {win_visible} ')

            mytk.erase_and_replot_neuron(n_neurons, n_tails, n_levels, fig, ax,canvas, neurons,box_nx,box_ny,box_xf, shapes,\
                                        neurons_tail_ntot,neurons_x,neurons_y, neurons_size,neurons_bulge_active,\
                                        neurons_bulge_size, neurons_tail_active,neurons_tail_node, neurons_tail_level, win_visible )

            img=mytk.fig_to_rgb(fig)

            root2.update_idletasks() 
            time.sleep(0.1)            

           
            ##writer.append_data(img)
            # Write frame to video
            writer.append_data(img)

            if (win_visible):
                canvas.draw()


  
            
    # write pos to trajectory
            routines.update_trajectories(n_neurons, tx, ty, itime, box_timestep,neurons_x,neurons_y,neurons_xprev, neurons_yprev,\
                                    neurons_velocity,box_xf,box_yf, neurons_tail_active,neurons_nbranches, neurons_ncontacts,gc_split, L2,L3,NK_freq)

            neurons_nbranches[:]=0.0
            neurons_ncontacts[:]=0.0
            neurons_gc_split[:]=0.0
            neurons_L2[:]=0.0
            neurons_L3[:]=0.0
            neurons_NK_freq[:]=0.0
            
            routines.update_diffusion(n_neurons, diffx, diffy, difftot, itime, box_timestep,\
                                 neurons_x, neurons_y,neurons_x_ini,neurons_y_ini,box_xf, box_yf,\
                                     neurons_times,ymin_time,ymax_time, neurons_tail_active, neurons_tail_ntot, neurons_tail_node,nmaxtails)


        rng=np.random.default_rng()
        rng_vals=rng.random(10*10000)
        
        accept_arr[:]=0
        reject_arr[:]=0
        jump.jump(neurons_pause,n_tails, pmax, pmin, rng_vals, n_neurons, potvalue, jumpamp_field, jumprate_field, accept_field,mcfield_sign,\
                      box_x0,box_xf,box_y0,box_yf,box_dx,box_dy,box_nx,box_ny,box_timestep, box_temperature, box_PBC,neurons_tail_ntot,\
                     neurons_tail_active,neurons_tail_node,neurons_jmp_dyn,neurons_x,neurons_y,\
                     neurons_jmp_rate, neurons_jmp_amp,neurons_jmp_ang,neurons_velocity,\
                     accept_arr,reject_arr,neurons_tail_hasparent,neurons_npoints_ini ) 


        box_accept+=sum(accept_arr)
        box_reject+=sum(reject_arr)
    ###########################################################################
        compute_force.compute_force(rng_vals,potvalue, n_neurons, n_tails, gx, gy,\
                          box_x0,box_xf,box_y0,box_yf,box_dx,box_dy,box_nx, box_ny,box_const_force, box_PBC,\
                          tail_branch_interaction_params,
                          neurons_force, neurons_repulsion, neurons_r_cut, neurons_r_width,\
                          neurons_tail_active, neurons_tail_ntot, neurons_tail_node,neurons_tail_hasparent,\
                          neurons_x, neurons_y,neurons_nk2,neurons_k2_id, neurons_length, neurons_k2,\
                          neurons_nk3, neurons_k3_id,neurons_cosangle,neurons_k3,neurons_is_repulsive,\
                          neurons_nneighbours, neurons_neighbours_id, neurons_ncontacts, active_contacts)
    ###########################################################################   
    ###########################################################################
        move_neurons.move_neurons(neurons_pause, n_neurons, n_tails, friction_field, box_x0, box_xf, box_y0, box_yf, box_dx, box_dy, box_nx, box_ny,box_PBC,\
                         box_timestep, neurons_tail_ntot, neurons_mass,neurons_bulge_active, neurons_bulge_mass,\
                         neurons_velocity,neurons_x,neurons_y, neurons_force,neurons_tail_node,neurons_tail_hasparent, neurons_tail_ntot,potvalue, potmax)

    ###########################################################################
    ###########################################################################
        tails.create_or_erase_tail(n_neurons, n_tails, n_levels, potvalue, tailrate_field, pert_field,slipstick_field, accept_field,mcfield_sign,pmax, pmin, box_temperature,box_timestep,\
                                 box_x0, box_xf, box_y0, box_yf, box_nx, box_ny, box_dx, box_dy, box_PBC, box_slipstick_pause,neurons_pause,\
                                 neurons_x, neurons_y, neurons_npoints_ini,neurons_length_ini,\
                                 neurons_tail_active, neurons_tail_ntot,neurons_bulge_active,\
                                 neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
                                 neurons_jmp_dyn_ini,neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini,\
                                 neurons_nk2, neurons_nk3, neurons_velocity,neurons_tail_node, neurons_tail_angle, neurons_mass, neurons_mass_ini, neurons_size, neurons_size_ini,\
                                 neurons_k2, neurons_k2_id, neurons_k3, neurons_k3_id, neurons_length,\
                                 neurons_cosangle, tail_nmax, tail_l, tail_l0, tail_k2, tail_k3,\
                                 tail_angle_start_min, tail_angle_start_max, tail_angle,\
                                 tail_mass_max,tail_create_rate, tail_delete_rate,\
                                 tail_create_multfactor, tail_delete_multfactor, neurons_tail_level, neurons_tail_hasparent, neurons_tail_nchildren, neurons_nbranches,friction_field,\
                                 box_slipstick_amplitude,neurons_gc_split, neurons_L2,neurons_L3,neurons_NK_freq)

    
    ###########################################################################
    ###########################################################################
        routines.create_or_remove_bulge(n_neurons, n_tails, pert_field, box_dx, box_dy, box_nx, box_ny,\
                                    neurons_bulge_active,neurons_bulge_minpoints,neurons_x,neurons_y,\
                                    neurons_bulge_create_rate,neurons_tail_active,neurons_tail_ntot,\
                                    neurons_tail_node,neurons_bulge_delete_rate, bulge_mult_factor)
    ###########################################################################
            


#main loop here    

def main():
    stop_event=threading.Event()

    win_visible=False

    root = tk.Tk()
    root.title("INM_SIM_v1")
    root.geometry('100x300+50+50')
    global root2
    root.title("Plot")
    root2=tk.Toplevel()
    root2.geometry('1000x1200+200+00')

    
    xmin=box_x0
    xmax=box_xf
    nx=box_nx
    ymin=box_y0
    ymax=box_yf
    ny=box_ny
    plotvalue = np.zeros((nx, ny), dtype=float)    
# Button to open plot window
    xpoints = np.linspace(xmin, xmax, nx)
    ypoints = np.linspace(ymin, ymax, ny)
    print("len x =", len(xpoints))
    print("len y =", len(ypoints))
    print("potvalue shape =", potvalue.shape)
    print("win_visible before startsim =", win_visible)

# Create matplotlib figure
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(1000*px, 1200*px))

    swap_pot=np.swapaxes(potvalue,0,1)
    vmmax=np.max(potvalue)
    #c = ax.pcolormesh(xpoints, ypoints, potvalue, vmin=0.0, vmax=15.0, shading='auto', cmap='viridis')
    c = ax.pcolormesh(xpoints, ypoints, swap_pot, vmin=-1, vmax=40.0,shading='auto', cmap='pink')
    #fig.colorbar(c)
    ax.set_title('Potential')
    xax = ax.axes.get_xaxis()
    xax = xax.set_visible(False)
    yax = ax.axes.get_yaxis()
    yax = yax.set_visible(False)


# Embed in tkinter window
    

    
    canvas = FigureCanvasTkAgg(fig, master=root2)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    ##    canvas.draw()

    
    btn2 =  tk.Button(root, text="Start Sim", command=lambda:start(root2,fig,ax,canvas,stop_event, win_visible))
    btn2.pack(pady=10)
    btn3 =  tk.Button(root, text="Stop Sim", command=lambda:stop_event.set())
    btn3.pack(pady=10)
    btn4 =  tk.Button(root, text="Field to plot", command=lambda:ouvrir_fenetre3(fig, ax,canvas,plotvalue, win_visible))
    btn4.pack(pady=10) 
    root.mainloop()

def toggle_show():
    window_name = "Plot"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    isopen=cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE)
    if isopen == -1.0:
        root2.deiconify()
    else:
        root2.withdraw()


def start(root2,fig, ax,canvas,stop_event, win_visible):
    print(f' in start win_visible {win_visible}')
    stop_event.clear()
    x=threading.Thread(target=startsim, args=(root2, fig, ax,canvas,stop_event,win_visible), daemon=True)
    x.start()

def ouvrir_fenetre3(fig, ax,canvas, plotvalue, win_visible):
           
    fenetre_var = tk.Toplevel()
    fenetre_var.geometry("200x200+50+370")
    fenetre_var.resizable(False, False)
    fenetre_var.title("Modify plotfield")
    label = tk.Label(fenetre_var, text="Choose option :", font=("Arial", 12))
    label.pack(pady=10)
    options = ["potential", "friction", "jumpamp", "jumprate", "tailrate", "branchrate", "slipstick", "affinity"]
    combo = ttk.Combobox(fenetre_var, values=options, state="readonly")  # readonly = empêche la saisie manuelle
    combo.current(0)  # Définit la valeur par défaut
    combo.pack(pady=5)

# Fonction appelée lorsqu’une option est sélectionnée
    def selection(event):
        selection = combo.get()
        result_label.config(text=f"Vous avez choisi : {selection}")
        box_plotfield = selection
        print(f' box plotfield {box_plotfield}')
        plotvalue[:,:]=pot_field[:,:]
        plotvalue[:,:]+=walls_field[:,:]
        plotvalue[:,:]+=pert_field[:,:]
        plotvalue[:,:]+=glials_field[:,:]

          
        if (box_plotfield =='potential'):
            plotvalue[:,:]=potvalue[:,:]          
        if (box_plotfield =='friction'):
            plotvalue[:,:]=friction_field[:,:]
        if (box_plotfield =='jumpamp'):
            plotvalue[:,:]=jumpamp_field[:,:]        
        if (box_plotfield =='jumprate'):
            plotvalue[:,:]=jumprate_field[:,:] 
        if (box_plotfield =='tailrate'):
            plotvalue[:,:]=tailrate_field[:,:]
        if (box_plotfield =='branchrate'):
            plotvalue[:,:]=branchrate_field[:,:]    
        if (box_plotfield =='slipstick'):
            plotvalue[:,:]=slipstick_field[:,:]   
        if (box_plotfield =='affinity'):
            plotvalue[:,:]=affinity_field[:,:] 
        mytk.updatecanvas(fig, ax,canvas,plotvalue,box_x0, box_xf,box_nx,box_y0,box_yf,box_ny,box_plotfield, win_visible)   
        print(f' box plotfield {box_plotfield}')


        
    combo.bind("<<ComboboxSelected>>", selection)

    # Label pour afficher le résultat
    result_label = tk.Label(fenetre_var, text="", font=("Arial", 11))
    result_label.pack(pady=10) 

   



    
if __name__ == "__main__":
    main()


