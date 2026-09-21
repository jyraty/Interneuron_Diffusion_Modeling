# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:42:07 2025
in this version 30, when a jump is rejected, there is a second attemp in opposite directions
@author: jyrat
"""
import numpy as np

import classes
from numba import njit, prange
import math

global pi
pi = np.float64(np.pi)
global twopi
twopi = np.float64(2.0*np.pi)

###############################################################################
#  make jumps
###############################################################################
#
#
@njit
def _bilinear_interp(field, x, y, xmin, ymin, deltax, deltay, nx, ny):
    """
    Bilinear interpolation inside a rectangular grid.
    field shape: (nx, ny)
    x,y: coordinates in same units as xmin/xmax etc.
    """
    # compute continuous indices
    fx = (x - xmin) / deltax
    fy = (y - ymin) / deltay

    ix = int(fx)
    iy = int(fy)

    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if ix >= nx - 1:
        ix = nx - 2
    if iy >= ny - 1:
        iy = ny - 2

    dx = fx - ix
    dy = fy - iy

    v00 = field[ix,   iy]
    v10 = field[ix+1, iy]
    v01 = field[ix,   iy+1]
    v11 = field[ix+1, iy+1]

    # bilinear combine
    v0 = v00*(1.0-dx) + v10*dx
    v1 = v01*(1.0-dx) + v11*dx
    v = v0*(1.0-dy) + v1*dy
    return v

@njit(parallel=True)
def jump(neurons_pause,n_tails, pmax, pmin, rng_vals,  n_neurons: int, pot_field, jumpamp_field, jumprate_field, interp_acceptance,mcfield_sign,
         box_x0, box_xf, box_y0, box_yf, box_dx, box_dy, box_nx, box_ny, box_timestep, box_temperature, box_PBC, neurons_tail_ntot,
         neurons_tail_active, neurons_tail_node, neurons_jmp_dyn, neurons_x, neurons_y,
         neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang, neurons_velocity,
         accept_arr,reject_arr, neurons_tail_hasparent, neurons_npoints_ini):

    xmin = box_x0
    xmax = box_xf
    ymin = box_y0
    ymax = box_yf
    deltax = box_dx
    deltay = box_dy
    nx = box_nx
    ny = box_ny
    mult_vel=10.0  # was 10 CHECK  
    if (pmax != pmin):
        Boltzmann = 1.0/(pmax-pmin)/box_temperature * deltax * deltay / box_timestep /box_timestep
    ##Boltzmann = 1.0/box_temperature * deltax * deltay / box_timestep /box_timestep
        
    # make jump for any point with jum_dyn=1
    for i in prange(n_neurons):
        if (neurons_pause[i] ==0):
            r_idx = i*10  # consume random numbers deterministically
            first_active=-1
            n_active_tails=sum(neurons_tail_active[i, :])
            for itail in range(n_tails):
    
                npoints = neurons_tail_ntot[i, itail]
                if (neurons_tail_active[i, itail] == 1):
                    if (neurons_tail_hasparent[i, itail] == 0 ):
                        main_tail=itail

                    
                    ### DEBUG correct here : ALSO jump trailing process if branched
                    ## change in V48  JUMP
                    #if ((itail == 0) and (neurons_tail_active[i, 1] ==0)):
                    if (itail == main_tail ):
                        startpoint = 0
                    else:
                        startpoint = neurons_tail_node[i, itail]
                        # node belongs to tail 0


                    # for the first npoints_ini points, only one tail should make jumps, then x and y should be copied to all others
                    # let's take the first active tail


    
                    for j in range(startpoint, neurons_tail_ntot[i, itail]):
                        myx=neurons_x[i, itail, j]
                        myy=neurons_y[i, itail, j]
                        if (neurons_jmp_dyn[i, itail, j] == 1):
                            ix = int(myx/deltax)
                            if (ix < 0):
                                ix = 0
                            if (ix > nx-1):
                                ix = nx-1
                            iy = int(myy/deltay)
                            if (iy < 0):
                                iy = 0
                            if (iy > ny-1):
                                iy = ny-1
                            #jumpamp_fac = _bilinear_interp(jumpamp_field, myx, myy, xmin, ymin, deltax, deltay, nx, ny)
                            #jumprate_fac = _bilinear_interp(jumprate_field, myx, myy, xmin, ymin, deltax, deltay, nx, ny)
                            jumpamp_fac = jumpamp_field[ix, iy]
                            jumprate_fac = jumprate_field[ix, iy]
                            # print(f" testjump i j {i} {j}")
    
                            prob_jump = rng_vals[r_idx]
                            r_idx += 1
    
                            if (prob_jump < neurons_jmp_rate[i, itail, j]*jumprate_fac):
                                # make a jump
                                # print(f" jump {i} {j}")
                                # for direction pt 1 in dir 12
                                #              pt 2 in dir 23
                                #              pt npoints in dir mpoints-1 npoints
                                x = neurons_x[i, itail, j]
                                y = neurons_y[i, itail, j]
                                
                                if (j == 0):
                                    xp0=neurons_x[i, itail, 0]
                                    yp0=neurons_y[i, itail, 0]
                                    xp1 = neurons_x[i, itail, 1]
                                    yp1 = neurons_y[i, itail, 1]
                                else:
                                    xp0=neurons_x[i, itail, j-1]
                                    yp0=neurons_y[i, itail, j-1]
                                    xp1 = neurons_x[i, itail, j]
                                    yp1 = neurons_y[i, itail, j]
    
                                dx = xp1-xp0
                                dy = yp1-yp0

                              

                                if box_PBC:
                                    dx = dx- xmax * round(dx / xmax)                                   
                                norm = math.sqrt(dx*dx+dy*dy)
                                ang1=np.arctan2(dy, dx)
    
    
                                
    
    
                                amplitude = rng_vals[r_idx] * neurons_jmp_amp[i, itail, j]*jumpamp_fac
                                r_idx += 1
                                delta_ang=  2.0 * (rng_vals[r_idx]-0.5)*neurons_jmp_ang[i,itail, j]*pi/180.0 
                                ang = ang1 +delta_ang
                                ang_revdir= ang1 -delta_ang
                                r_idx += 1
                                
                                ddx = amplitude*np.cos(ang)
                                ddy = amplitude*np.sin(ang)
    
                                OK_ACC = True
                                if (amplitude < norm):
                                    xp=neurons_x[i, itail, j]
                                    yp=neurons_y[i, itail, j]
                                    vxp=neurons_velocity[i, itail, j, 0]
                                    vyp=neurons_velocity[i, itail, j, 1] 
                                    neurons_x[i, itail, j] += ddx
                                    newposx= neurons_x[i, itail, j] 
                                    neurons_y[i, itail, j] += ddy
                                    newposy= neurons_y[i, itail, j] 
                                    neurons_velocity[i, itail, j, 0] += ddx \
                                        / box_timestep * neurons_jmp_rate[i, itail, j]*mult_vel
                                    neurons_velocity[i, itail, j, 1] += ddy \
                                        / box_timestep * neurons_jmp_rate[i, itail, j]*mult_vel
                                        
                                    # reject if out of limits
                                    if (box_PBC == False):
                                        if ((neurons_x[i, itail, j] <= xmin) or (neurons_x[i, itail, j] >= xmax)):
                                            OK_ACC = False
                                    else:
                                        if (neurons_x[i, itail, j] <= xmin):
                                            neurons_x[i, itail, j] += xmax
                                        if (neurons_x[i, itail, j] >= xmax):
                                            neurons_x[i, itail, j] -= xmax
    
                                    if ((neurons_y[i, itail, j] <= ymin+10) or (neurons_y[i, itail, j] >= ymax-10)):      
                                        OK_ACC = False
                                        
                                    #  for exotic frames, reject if any potential value higher than 3.0
                                    ix = int(newposx/deltax)
                                    iy = int(newposy/deltay)
                                    if (pot_field[ix, iy]> 19.0):
                                        OK_ACC = False    ####DEBUG TEST IF USEFUL
                                           
                                    if not OK_ACC:
                                        neurons_x[i, itail, j]=xp
                                        neurons_velocity[i, itail, j, 0] = vxp
                                        neurons_y[i, itail, j] = yp
                                        neurons_velocity[i, itail, j, 1]  =vyp
                                        
    
                                # rejects based on acceptance rates
                                    REJECT_DIR=False
                                    if (OK_ACC == True):
                                        xp = x-ddx
                                        yp = y-ddy
        
                                        if (box_PBC == True):
                                            if (xp >= xmax):
                                                xp -= xmax
                                            if (xp <= xmin):
                                                xp += xmax
                                        p2 = _bilinear_interp(interp_acceptance, x, y, xmin, ymin, deltax, deltay, nx, ny)
                                        p1 = _bilinear_interp(interp_acceptance, xp, yp,   xmin, ymin, deltax, deltay, nx, ny)
                                        
                                        if (pmax !=pmin):
                                            delta = (p2-p1)*Boltzmann #/(pmax-pmin)/box_temperature * xmax * ymax / box_timestep /box_timestep
                                        ## here wed have to do something about normalization
                                        ## E/kT  E either pot or kin
                                        ## kin m*v**2/2
                                        else:
                                            delta=0.0
                                            
                                        if (mcfield_sign=='Positive'):
                                            prob_acc = np.exp(delta)
                                        else:
                                            prob_acc = np.exp(-delta)
                                       ### print(f' {box_temperature} {box_timestep} {p1} {p2} {delta} {prob_acc}')
        
                                        if (rng_vals[r_idx] > prob_acc):
                                            
                                            r_idx += 1
                                            neurons_x[i, itail, j] =xp
                                            neurons_y[i, itail, j] =yp
                                            neurons_velocity[i, itail, j, 0]=vxp
                                            neurons_velocity[i, itail, j, 1]=vyp
                                            
                                            if (box_PBC == True):
                                                if (neurons_x[i, itail, j] >= xmax):
                                                    neurons_x[i, itail, j] -= xmax
                                                if (neurons_x[i, itail, j] <= xmin):
                                                    neurons_x[i, itail, j] += xmax
                                            REJECT_DIR=True
                                            
                                        if REJECT_DIR:  #we try other direction
                                            ###IF main tail and 3 points, inverse polarity
                                            ix = int(x/deltax)
                                            iy = int(y/deltay)
                                            
                                            #if (pot_field[ix, iy]> 1.0) and (n_active_tails==1 ) and (neurons_tail_ntot[i, itail]==neurons_npoints_ini[i] and (j == 2)):
                                            if (pot_field[ix, iy]> 1.0) and (n_active_tails==1 ) and (j == 2):
                                                #let s inverse positions and speed for all points. the j = 2 condition is there to avoid double inversion
                                                endpoint=neurons_npoints_ini[i]
                                                for j in range(endpoint):
                                                    jnew=endpoint-1-j
                                                    tx = neurons_x[i, itail-1, j]
                                                    neurons_x[i, itail-1, j]=neurons_x[i, itail-1, jnew]
                                                    neurons_x[i, itail-1, jnew]=tx
                                                    tx = neurons_y[i, itail-1, j]
                                                    neurons_y[i, itail-1, j]=neurons_y[i, itail-1, jnew]
                                                    neurons_y[i, itail-1, jnew]=tx
                                                    tx = -neurons_velocity[i, itail-1, j,0]
                                                    neurons_velocity[i, itail-1, j,0]=-neurons_velocity[i, itail-1, jnew,0]
                                                    neurons_velocity[i, itail-1, jnew,0]=tx
                                                    tx = -neurons_velocity[i, itail-1, j,1]
                                                    neurons_velocity[i, itail-1, j,1]=-neurons_velocity[i, itail-1, jnew,1]
                                                    neurons_velocity[i, itail-1, jnew,1]=tx
                                                    OK_ACC=True
                                                #print('reversed on high pot')
                                            else:
                                                ##DEBUG REMOVE REVERSE DIT FOR MOMENT
                                                ang = ang_revdir  
                                            
                                                ddx = amplitude*np.cos(ang)
                                                ddy = amplitude*np.sin(ang)
            
                                                OK_ACC = True
                                                if (amplitude < norm):
                                                    xp=neurons_x[i, itail, j]
                                                    yp=neurons_y[i, itail, j]
                                                    vxp=neurons_velocity[i, itail, j, 0]
                                                    vyp=neurons_velocity[i, itail, j, 1] 
                                                    neurons_x[i, itail, j] += ddx
                                                    neurons_y[i, itail, j] += ddy
                                                    x=neurons_x[i, itail, j]
                                                    y=neurons_y[i, itail, j]
                                                    neurons_velocity[i, itail, j, 0] += ddx \
                                                        / box_timestep * neurons_jmp_rate[i, itail, j]*mult_vel
                                                    neurons_velocity[i, itail, j, 1] += ddy \
                                                        / box_timestep * neurons_jmp_rate[i, itail, j]*mult_vel
                                                    
                                                    # reject if out of limits
                                                    if (box_PBC == False):
                                                        if ((neurons_x[i, itail, j] <= xmin) or (neurons_x[i, itail, j] >= xmax)):                               
                                                            OK_ACC = False
                                                    else:
                                                        if (neurons_x[i, itail, j] <= xmin):
                                                            neurons_x[i, itail, j] += xmax
                                                        if (neurons_x[i, itail, j] >= xmax):
                                                            neurons_x[i, itail, j] -= xmax
            
                                                    if ((neurons_y[i, itail, j] <= ymin+0*deltay) or (neurons_y[i, itail, j] >= ymax-0*deltay)):
                                                        OK_ACC = False
                                                  
                                                    # for exotic masks reject if too high pot
                                                    ix = int(x/deltax)
                                                    iy = int(y/deltay)
                                                    if (pot_field[ix, iy]> 3.0):
                                                        OK_ACC = False
                                                ###
                                                
                                                
                                                if not OK_ACC:
                                                    neurons_x[i, itail, j]=xp
                                                    neurons_velocity[i, itail, j, 0] = vxp
                                                    neurons_y[i, itail, j] = yp
                                                    neurons_velocity[i, itail, j, 1]  =vyp
                                                    
                                            # rejects based on acceptance rates
        
                                                
                                                if (OK_ACC == True):                
                                                    if (box_PBC == True):
                                                        if (xp >= xmax):
                                                            xp -= xmax
                                                        if (xp <= xmin):
                                                            xp += xmax
                                                            
                                                    p2 = _bilinear_interp(interp_acceptance, x, y, xmin, ymin, deltax, deltay, nx, ny)
                                                    p1 = _bilinear_interp(interp_acceptance, xp, yp,   xmin, ymin, deltax, deltay, nx, ny)
                                                   
                   
                                                    delta = (p2-p1)*Boltzmann
                                                    if (mcfield_sign=='Positive'):
                                                        prob_acc = np.exp(delta)
                                                    else:
                                                        prob_acc = np.exp(-delta)
                                                   ### print(f' {box_temperature} {box_timestep} {p1} {p2} {delta} {prob_acc}')
                    
                                                    if (rng_vals[r_idx] > prob_acc):
                                                        
                                                        r_idx += 1
                                                        neurons_x[i, itail, j] =xp
                                                        neurons_y[i, itail, j] =yp
                                                        neurons_velocity[i, itail, j, 0]=vxp
                                                        neurons_velocity[i, itail, j, 1]=vyp
                                                        
                                                        if (box_PBC == True):
                                                            if (neurons_x[i, itail, j] >= xmax):
                                                                neurons_x[i, itail, j] -= xmax
                                                            if (neurons_x[i, itail, j] <= xmin):
                                                                neurons_x[i, itail, j] += xmax
                                                        reject_arr[i] += 1
                                        
                                        
                                            
                                        else:
                                            accept_arr[i] += 1



    return
