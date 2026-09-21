# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:49:18 2025

@author: jyrat
"""
import math
from multiprocessing.spawn import old_main_modules
from tkinter import ACTIVE
from numba import njit, prange
##
#
###############################################################################
#  compute force TODO
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
    '''if ix >= nx - 1:
        ix = nx - 2
    if iy >= ny - 1:
        iy = ny - 2
    '''
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
    v =   v0*(1.0-dy) + v1*dy
    return v

@njit(parallel=True, fastmath=True)
def compute_force(rng_vals,potvalue, n_neurons, n_tails, gx, gy,
                  box_x0, box_xf, box_y0, box_yf, box_dx, box_dy, box_nx, box_ny,box_const_force, box_PBC,
                  tail_branch_interaction_params,
                  neurons_force, neurons_repulsion, neurons_r_cut, neurons_r_width,
                  neurons_tail_active, neurons_tail_ntot, neurons_tail_node, neurons_tail_hasparent, 
                  neurons_x, neurons_y, neurons_nk2, neurons_k2_id, neurons_length, neurons_k2,
                  neurons_nk3, neurons_k3_id, neurons_cosangle, neurons_k3,neurons_is_repulsive,\
                  neurons_nneighbours, neurons_neighbour_id, neurons_ncontacts, active_contacts):
    # in this version we cancel the possibility for divergence (distances smaller than 1.0 are set to 1.0)
    xmin = box_x0
    xmax = box_xf
    ymin = box_y0
    ymax = box_yf
    deltax = box_dx
    deltay = box_dy
    nx=box_nx
    ny=box_ny
    l0_tb=tail_branch_interaction_params[0]
    k2_tb=tail_branch_interaction_params[1]

    dmin=1.0
    

    

    # first, external force is computed on each point of each neuron

    const_force = box_const_force

    for i in prange(n_neurons):
        r_idx=i*10
        #print(f' check3 {i} { neurons_x[i,0,2]} { neurons_x[i,0,1]} { neurons_x[i,0,0]}')
        neurons_force[i, :, :, :] = 0.0   # [itail, point, dir]
        rep = neurons_repulsion[i]
        r_width = neurons_r_width[i]
        r_cut = neurons_r_cut[i]


        for itail in range(n_tails):

            if (neurons_tail_active[i, itail] == 1):

                #if (itail == 0):
                if (neurons_tail_hasparent[i,itail]==0):
                    startpoint = 0
                else:
                    startpoint = neurons_tail_node[i, itail]-1
                endpoint = neurons_tail_ntot[i, itail]
                for j in range(startpoint, endpoint):

                    x = neurons_x[i, itail, j]
                    y = neurons_y[i, itail, j]

                    if (box_PBC == False):
                        if (x < xmin+deltax/2.0):
                            x = xmin+deltax/2.0
                        if (x > xmax-deltax/2.0):
                            x = xmax-deltax/2.0
                    else:
                        if (x < xmin):
                            x = x + xmax
                        if (x > xmax):
                            x = x - xmax

                    if (y < ymin+1*deltay/5.0):
                        y = ymin+1*deltay/5.0
                    if (y > ymax-1*deltay/5.0):
                        y = ymax-1*deltay/5.0

                    # print(f' i itail j x y {i} {itail} {j} {x} {y}')

                    
                    gx_val = _bilinear_interp(gx, x, y, xmin, ymin, deltax, deltay, nx, ny)
                    gy_val = _bilinear_interp(gy, x, y, xmin, ymin, deltax, deltay, nx, ny)
                    # ­ JY CHECK DEBUG

                    neurons_force[i, itail, j, 0] -= gx_val
                    neurons_force[i, itail, j, 1] -= gy_val
                    neurons_force[i, itail, j, 1] += const_force

                # add internal pair force
                for k in range(neurons_nk2[i, itail]):

                    # only add this force if after node ( = startpoint+1)  !!!!!

                    i1 = neurons_k2_id[i, itail, k, 0]

                    if (i1 >= startpoint+1):

                        i2 = neurons_k2_id[i, itail, k, 1]
                        l0 = neurons_length[i, itail, i1-1, i2-1]
                        kk = neurons_k2[i, itail, k]

                        dx = neurons_x[i, itail, i2-1] - neurons_x[i, itail, i1-1]  # PBC
                        dx -= xmax * round(dx / xmax)

                        dy = neurons_y[i, itail, i2-1]-neurons_y[i, itail, i1-1]
                        l = math.sqrt(dx*dx+dy*dy)


                        costhetax = dx/l                 
                        sinthetay = dy/l  # debug
                  

                        if (l < dmin):
                            l = dmin
                        if (l > l0+dmin):
                            l = l0+dmin
                        fx = kk*(l-l0)*costhetax
                        fy = kk*(l-l0)*sinthetay

                        neurons_force[i, itail, i2-1, 0] -= fx
                        neurons_force[i, itail, i2-1, 1] -= fy
                        neurons_force[i, itail, i1-1, 0] += fx
                        neurons_force[i, itail, i1-1, 1] += fy
                        if (fx > 100):
                            print(f'i1 i2 force kk l l0', i1, i2, fx, fy, kk, l, l0)
                # add internal angular force
                for k in range(neurons_nk3[i, itail]):
                    i1 = neurons_k3_id[i, itail, k, 0]


                    i2 = neurons_k3_id[i, itail, k, 1]
                    # only add if i2 >= node ( or = startpoint+1)!!!
                    if (i2 >= startpoint+1):

                        i3 = neurons_k3_id[i, itail, k, 2]
                        # careful if index > 100, the point is on the PARENT tail
                        i_parent = neurons_tail_hasparent[i, itail]   
                   

                        a0 = neurons_cosangle[i, itail, k]
                        k3 = neurons_k3[i, itail, k]

                        tailref1 = itail
                        iref1 = i1

                        if (i1 > 100):
                            tailref1 = i_parent-1
                            iref1 = i1-100
                        tailref2 = itail
                        iref2 = i2

                        if (i2 > 100):
                            tailref2 = i_parent-1
                            iref2 = i2-100
                        tailref3 = itail
                        iref3 = i3

                        if (i3 > 100):
                            tailref3 = i_parent-1
                            iref3 = i3-100

                        dx1 = neurons_x[i, tailref1, iref1-1] - neurons_x[i, tailref2, iref2-1]  # PBC
                        dx1 -= xmax * round(dx1 / xmax)
                    
                        dy1 = neurons_y[i, tailref1, iref1-1] - neurons_y[i, tailref2, iref2-1]
                    
                        dx2 = neurons_x[i, tailref3, iref3-1] - neurons_x[i, tailref2, iref2-1]  # PBC
                        dx2 -= xmax * round(dx2 / xmax)
                    
                        dy2 = neurons_y[i, tailref3, iref3-1] - neurons_y[i, tailref2, iref2-1]

                        l1 = math.sqrt(dx1*dx1+dy1*dy1)
                        l2 = math.sqrt(dx2*dx2+dy2*dy2)
                        '''if (l1 < dmin): 
                            l1=dmin

                        if (l2 < dmin):
                            l2=dmin
                        '''
                        

                        costheta = (dx1*dx2+dy1*dy2)/l1/l2

                        dpot = 2*k3*(costheta-a0)
                        ### here is (theta-theta0)**2
                        '''fx1=-dpot*(dx2/l1/l2 - costheta/l1)
                        fy1=-dpot*(dy2/l1/l2 - costheta/l1)

                        fx3=-dpot*(dx1/l1/l2 - costheta/l2)
                        fy3=-dpot*(dy1/l1/l2 - costheta/l2)
                        '''
                        ## end theta -theta0)**2

                        ### here is for costheta -costheta0
                        fx1 = -dpot*(dx2/l1/l2 - costheta*dx1/l1/l1)
                        fy1 = -dpot*(dy2/l1/l2 - costheta*dy1/l1/l1)

                        fx3 = -dpot*(dx1/l1/l2 - costheta*dx2/l2/l2)
                        fy3 = -dpot*(dy1/l1/l2 - costheta*dy2/l2/l2)
                        
                        ### end costheta-costeheta°

                        fx2 = -1.0*(fx1+fx3)
                        fy2 = -1.0*(fy1+fy3)


                        neurons_force[i, tailref1, iref1-1, 0] += fx1
                        neurons_force[i, tailref1, iref1-1, 1] += fy1
                        neurons_force[i, tailref2, iref2-1, 0] += fx2
                        neurons_force[i, tailref2, iref2-1, 1] += fy2
                        neurons_force[i, tailref3, iref3-1, 0] += fx3
                        neurons_force[i, tailref3, iref3-1, 1] += fy3

                        # sum of internal forces should be 0
                        # print(f"i1 i2 i3 angle f {i1} {i2} {i3} {ang} {f}")
                        # add repulsion between neurites or branches heads

                        # problem for node forces : need to sum in first channel

                if (itail != 0):
                    left_node = neurons_tail_node[i, itail]
                    neurons_force[i, 0, left_node,
                                  0] += neurons_force[i, itail, left_node, 0]
                    neurons_force[i, 0, left_node,
                                  1] += neurons_force[i, itail, left_node, 1]
                    

        for k_index in range(neurons_nneighbours[i]):
            newcontact=False 
            contact=False
            old_contact=False
            k = neurons_neighbour_id[i, k_index]
            ##repk = neurons_repulsion[k]
                    
            if (active_contacts[i,k] != 0):
                old_contact=True
            

               

            for itail in range(n_tails):
                if ((neurons_tail_active[i, itail] == 1) and (neurons_is_repulsive[i]!=0)):
                    # last point on tail1 and 2 (if any)
                    j1 = neurons_tail_ntot[i, itail]

                    #in this version we loop only on neighbours

                    for ktail in range(n_tails):
                        if ((neurons_tail_active[k, ktail] == 1) and (neurons_is_repulsive[k]!=0)):
                            
                            # last point on tail neuron i
                            x1 = neurons_x[i, itail, j1-1]
                            y1 = neurons_y[i, itail, j1-1]
                            # last point on tail neuron k
                            ##all tail points
                            for k1 in range (1,neurons_tail_ntot[k, ktail]):
                                x2 = neurons_x[k, ktail, k1-1]
                                y2 = neurons_y[k, ktail, k1-1]
                                dx = x2-x1
                                if box_PBC:
                                    dx -= xmax * round(dx / xmax)
                                dy = y2-y1  # PBC
                                d = (dx*dx+dy*dy)
                                if (d < r_cut*r_cut):
                                        dist=math.sqrt(d)                               
                                        ##r = -(rep*repk)/(((dist+r_width)/r_width)**2.0)
                                        r = rep/(((dist+r_width)/r_width)**3.0)
                                        if (dist < dmin): 
                                            dist=dmin
                                        
                                        fx = r*dx/dist  # costheta
                                        fy = r*dy/dist  # sintheta

                                        #rescale if not goof

                                        # sign depents on the side !
                                        neurons_force[i, itail, j1-1, 0] -= fx
                                        neurons_force[i, itail, j1-1, 1] -= fy
                                        # JY debug here can't have same sign
                                        neurons_force[k, ktail, k1-1, 0] += fx
                                        neurons_force[k, ktail, k1-1, 1] += fy
                                        contact=True
                                        active_contacts[i, k] += 1
                                        
                                        if (old_contact==False):
                                            newcontact=True
                                            old_contact==True
                                            #print('newcontact')
                                          
            if (contact==False):
                if (active_contacts[i, k] > 1000):
                    active_contacts[i, k] = 0
                
                
            if (newcontact==True):
                neurons_ncontacts[i] += 1
                #print('new contact')
              
    return

