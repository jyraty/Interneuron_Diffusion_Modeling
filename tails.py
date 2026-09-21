# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:56:01 2025

@author: jyrat
"""


from string import printable
import numpy as np
from numba import njit, prange

from classes import Neuron

global pi
pi = np.float64(np.pi)
global twopi
twopi = np.float64(2.0*np.pi)
import math


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
#
#
###############################################################################
#  create or erase tail - 1 or 2 !!!
###############################################################################
#
#


def slipstick_itail(rng_vals,itail, taillevel,neurons_x,neurons_y,i,  n_neurons, pmax, pmin, box_temperature, box_timestep, accept_field,mcfield_sign, \
               box_x0, box_xf, box_y0, box_yf,box_nx, box_ny,box_PBC,box_slipstick_pause,neurons_pause,neurons_length_ini,\
               neurons_bulge_active, \
               neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
               neurons_mass,\
               neurons_jmp_dyn_ini,neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini, neurons_mass_ini,\
               neurons_velocity,neurons_nk2, neurons_nk3,\
               neurons_tail_ntot, potfield,friction_field, box_slipstick_amplitude):

    xmin = box_x0
    xmax = box_xf
    ymin = box_y0
    ymax = box_yf
    nx=box_nx
    ny=box_ny
    deltax=xmax/nx
    deltay=ymax/ny
    mult_vel=0.1
## in this V42 it my be less than endpoint 
    #pick random end
    r_idx=i
    
    if (pmax != pmin):
        Boltzmann = 1.0/(pmax-pmin)/box_temperature * deltax * deltay / box_timestep /box_timestep
        

    # OK tere should be only one tail. it is itail
    endpoint=neurons_tail_ntot[i,itail-1]
    # in this version we use the amplitude of shift to determine nr shifting points
    #nr_shifting_points=3+int(rng_vals[r_idx]*(endpoint-3)*(1-box_slipstick_amplitude) )
    nr_shifting_points=endpoint-int(box_slipstick_amplitude)
    #was     nr_shifting_points=3+int(box_slipstick_amplitude)
    # this is new in V4 12-16-26



   ## nr_shifting_points=3
    if (nr_shifting_points > endpoint-1) or (nr_shifting_points < 3):
        return
        

    if ((endpoint > 3) and (neurons_bulge_active[i,itail-1]==0) ) :
        #print(f' slip {nr_shifting_points} on {endpoint}')
        ##in V42 we can have more than 3 points shifting to last
        # here, we change coordinates of point 3 to those of endpoint
        xn2=neurons_x[i,itail-1,endpoint-1]
        yn2=neurons_y[i,itail-1,endpoint-1]
        
        xo=np.zeros(endpoint, dtype=float)
        yo=np.zeros(endpoint, dtype=float)
        for j in range (endpoint):
            xo[j]=neurons_x[i,itail-1,j]
            yo[j]=neurons_y[i,itail-1,j]
        xp=xo[nr_shifting_points-1]
        dx=xn2-xo[nr_shifting_points-1]
        if box_PBC:
            dx = dx- xmax * round(dx / xmax)
        yp=yo[nr_shifting_points-1]
        dy=yn2-yo[nr_shifting_points-1]
        

        
        #in this version we shift according to affinity field

        # in this version, no shift if tail on high potential
        pot2=_bilinear_interp(potfield, xn2, yn2,   xmin, ymin, deltax, deltay, nx, ny)
        pot1=_bilinear_interp(potfield, xp, yp,   xmin, ymin, deltax, deltay, nx, ny)
        if ((pot2> 1.0) and (pot2 >= pot1)):  ##DEBUG : eventually inverse polarity of neuron and field, but for now we just do not move if potential is high
            # added >= bcause eventually it is on the high potential plateau
            #let s inverse positions and speed for all points
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
            xn2=neurons_x[i,itail-1,endpoint-1]
            yn2=neurons_y[i,itail-1,endpoint-1]
        
            xo=np.zeros(endpoint, dtype=float)
            yo=np.zeros(endpoint, dtype=float)
            for j in range (endpoint):
                xo[j]=neurons_x[i,itail-1,j]
                yo[j]=neurons_y[i,itail-1,j]
            xp=xo[nr_shifting_points-1]
            dx=xn2-xo[nr_shifting_points-1]
            if box_PBC:
                dx = dx- xmax * round(dx / xmax)
            yp=yo[nr_shifting_points-1]
            dy=yn2-yo[nr_shifting_points-1]

        xsoma=neurons_x[i, itail-1, 1]
        ysoma=neurons_y[i, itail-1, 1]
        friction_coeff=_bilinear_interp(friction_field, xsoma, ysoma,   xmin, ymin, deltax, deltay, nx, ny)
        if (friction_coeff < 1.0):
            friction_coeff=1.0
        p2 = _bilinear_interp(accept_field, xn2, yn2, xmin, ymin, deltax, deltay, nx, ny)
        p1 = _bilinear_interp(accept_field, xp, yp,   xmin, ymin, deltax, deltay, nx, ny)
        
        
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
        #if (rng_vals[r_idx] < prob_acc):
        if (rng_vals[r_idx] < 10.0):  #DEBUG HARDCODED
            r_idx += 1            
            neurons_pause[i]=box_slipstick_pause    
            ### NEW 08_04_27: there is an effect of friction. It slows down the movement (stops it) -> effectively decrease dx and dy accordingly

            dx=dx/(friction_coeff)
            dy=dy/(friction_coeff)

            for j in range (nr_shifting_points):
                xn1= neurons_x[i,itail-1,j]+dx
                yn1= neurons_y[i,itail-1,j]+dy
            
            
    
                if box_PBC:
                    if (xn1 > xmax):
                        xn1 -= xmax
                    if (xn1 < xmin):
                        xn1 += xmax  
                else:                
                    if (xn1 > xmax):
                        xn1 = xmax-deltax
                    if (xn1 < xmin):
                        xn1 = xmin+deltax     
                    
                    if (yn1 > ymax-0.2*deltay):
                        yn1 = ymax-0.2*deltay
                    if (yn1 < ymin+0.2*deltay):
                        yn1 = ymin+0.2*deltay           
                    
              
                neurons_x[i,itail-1,j]=xn1
                neurons_y[i,itail-1,j]=yn1
                #neurons_velocity[i, 0, j, 0] = dx \
                #    / box_timestep *mult_vel
                #neurons_velocity[i, 0, j, 1] = dy \
                #    / box_timestep *mult_vel
                    
                neurons_velocity[i,itail-1,j,0]=0.0
                neurons_velocity[i,itail-1,j,1]=0.0
            
            #print(f' slip i xn xo {i} {xn2} {xo2} {xn1} {xo1} {xn0} {xo0}')
            #print(f' slip yn yo {i} {yn2} {yo2} {yn1} {yo1} {yn0} {yo0}')
            #print(f' { neurons_x[i,0,2]} { neurons_x[i,0,1]} { neurons_x[i,0,0]}')
                       
            neurons_tail_ntot[i,itail-1]=nr_shifting_points
            neurons_nk2[i,itail-1]=nr_shifting_points-1
            neurons_nk3[i,itail-1]=nr_shifting_points-2
            ##V42 everypoint moves with its characteristics
            ### if nr_shiftingf_point=3 set back to ini values
            if (nr_shifting_points==3):
                neurons_jmp_dyn[i,:,:]=neurons_jmp_dyn_ini[i,:,:]
                neurons_jmp_rate[i,:,:]=neurons_jmp_rate_ini[i,:,:]
                neurons_jmp_amp[i,:,:]=neurons_jmp_amp_ini[i,:,:]
                neurons_jmp_ang[i,:,:]=neurons_jmp_ang_ini[i,:,:]
                neurons_mass[i,:,:]=neurons_mass_ini[i,:,:]
            ## else we keep all data but have to preserve the endpoint values (mass, dyn etc...)
            ## these are transferred to point nr_shifting_point
            else:
                j=nr_shifting_points-1
                k=endpoint-1
                # reinforce initial values at end of tail
                neurons_jmp_dyn[i,:,j]=neurons_jmp_dyn_ini[i,0,2]
                neurons_jmp_rate[i,:,j]=neurons_jmp_rate_ini[i,0,2]
                neurons_jmp_amp[i,:,j]=neurons_jmp_amp_ini[i,0,2]
                neurons_jmp_ang[i,:,j]=neurons_jmp_ang_ini[i,0,2]
                neurons_mass[i,:,j]=neurons_mass_ini[i,0,2]
                #keep velocity - eventually to remove !
                #neurons_velocity[i, 0, j, 0]=neurons_velocity[i, 0, k, 0]
                #neurons_velocity[i, 0, j, 1]=neurons_velocity[i, 0, k, 1]    
    return


def erase_pts_tail1(i,  itail, neurons_npoints_ini,\
               neurons_tail_ntot,\
               neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
               neurons_jmp_dyn_ini,neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini,\
               neurons_nk2, neurons_nk3, neurons_mass,neurons_mass_ini,\
               neurons_tail_active, neurons_tail_level, neurons_tail_hasparent, neurons_tail_nchildren,\
               neurons_size,neurons_size_ini ):



    # itail should be equal to 1
    if (itail != 1):
        print(f' bug itail should be 1 {itail}')

    # in this new version thisq procedure is clled if only 1 tail is active: itail
    #     # all points removed except first three    
    neurons_tail_active[i,itail-1]=1
    neurons_tail_nchildren[i,itail-1]=0
    neurons_nk2[i, itail-1] = neurons_npoints_ini[i]-1
    neurons_nk3[i, itail-1] = neurons_npoints_ini[i]-2



    for j in range(neurons_npoints_ini[i]-1): 
        neurons_mass[i,itail-1, j] = neurons_mass_ini[i,0,j]
        neurons_jmp_dyn[i,itail-1, j] = neurons_jmp_dyn_ini[i,0, j]
        neurons_jmp_rate[i,itail-1, j] = neurons_jmp_rate_ini[i,0, j]
        neurons_jmp_amp[i,itail-1, j] = neurons_jmp_amp_ini[i,0, j]
        neurons_jmp_ang[i,itail-1, j] = neurons_jmp_ang_ini[i,0, j]
        neurons_size[i, j] = neurons_size_ini[i,j]
    neurons_tail_active[i,itail-1] = 1
    neurons_tail_ntot[i, itail-1]=3
    return

def erasefull_tail(i,n_tails,tail_k2, tail_k3,\
                   neurons_tail_active,\
                   id_tail_to_erase, neurons_tail_nchildren, neurons_tail_hasparent, neurons_tail_level, neurons_tail_node, \
                   neurons_x, neurons_y, neurons_nk2, neurons_k2, neurons_k2_id,\
                   neurons_length, neurons_size, neurons_jmp_dyn, neurons_jmp_ang, neurons_jmp_rate, neurons_jmp_amp,\
                   neurons_nk3, neurons_k3, neurons_k3_id, neurons_tail_ntot, neurons_velocity,
                   neurons_jmp_dyn_ini, neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini, neurons_mass_ini,neurons_mass):


    # obviously if there is only one tail, one can not erase
    # normally this one has no parent

    ntails=sum(neurons_tail_active[i,:])
    itail=id_tail_to_erase

    if (itail ==0):
        return  # can t erase master tail
    

    if (ntails == 1):
        if (neurons_tail_active[i,0] !=0):
            print(' bug !! should remain only tail 1 active')
        return   # can not erase only tail

        # double check : if there is a child one can not erase
    nr_children=neurons_tail_nchildren[i,itail-1]
    parent=neurons_tail_hasparent[i,itail-1]
    if (ntails != 2):
        if (nr_children > 0):
            return   # no changes are made, there are children. 
  

    neurons_tail_active[i,itail-1]=0
    # the parent should lose this particular child 
    if (parent >0):
        neurons_tail_nchildren[i,parent-1] -=1 

    # other quantities may remain (they would be overwritten upon creation)
    
    # if there is only one tail left, copy all its data to tail # 1

    if (ntails == 2): # be sure we have not erased master

        for j in range(n_tails):  # identify which is active
            if (neurons_tail_active[i,j]==1) : 
                i_active=j

        if (i_active != 0):  # we have erased the master. Copy back to master (itail 0)
            #print('We have erased master tail, copy back to master')

            neurons_x[i,0, :] = neurons_x[i,i_active, :]
            neurons_y[i,0, :] = neurons_y[i,i_active, :]
            neurons_nk2[i,0] = neurons_nk2[i,i_active]
            neurons_k2[i,0, :] = neurons_k2[i,i_active, :]
            neurons_k2_id[i,0, :, :] = neurons_k2_id[i,i_active, :, :] 

            neurons_length[i,0,:,:] = neurons_length[i,i_active,:,:]
            ##neurons_size[i,0,:]     = neurons_size[i,i_active,:]
            neurons_jmp_dyn[i,0,:] = neurons_jmp_dyn[i,i_active,:]
            neurons_jmp_ang[i,0,:] = neurons_jmp_ang[i,i_active,:]
            neurons_jmp_rate[i,0,:] = neurons_jmp_rate[i,i_active,:]
            neurons_jmp_amp[i,0,:] = neurons_jmp_amp[i,i_active,:]
            ##neurons_nk3[i,0] = neurons_nk3[i,i_active]   ##### should be reset !!!
            ##neurons_k3[i,0, :] = neurons_k3[i,i_active, :]
            ##neurons_k3_id[i,0, :, :] = neurons_k3_id[i,i_active, :, :]
            
            # cosangle remains that of 0, k3, nk3 should be RESTET  CHECK BUG

            neurons_tail_ntot[i,0]=  neurons_tail_ntot[i,i_active]
            neurons_velocity[i,0, :, :] = neurons_velocity[i,i_active, :, :]
            neurons_tail_active[i,0]=1
            neurons_tail_active[i,i_active]=0
            neurons_tail_nchildren[i,0]=0

            ###CHECK INODE SHOULD BE SET TO 3  and ther eshould be a minimum of 3 points
            if (neurons_tail_ntot[i,0] < 3):
                print(f' we have a problem: left with less than 2 nodes -> force keeping 3rd')
                neurons_tail_ntot[i,0]=3
            if (neurons_tail_node[i,0] < 3):
                print(f' we have a problem: only one tail but node less than 3 -> force')
                neurons_tail_node[i,0]=3


             ###the k2 k3 chain on 0 should have been kept intact - no need to recopy  - just to be sure !

            nk3=0
            for k in range(neurons_tail_ntot[i,0]):
                index=k+1
                if (index > 2):
                    nk3 += 1
                    neurons_k3[i,0, nk3-1] = tail_k3[0]
                    neurons_k3_id[i,0, nk3-1, 0] = index-2
                    neurons_k3_id[i,0, nk3-1, 1] = index-1
                    neurons_k3_id[i,0, nk3-1, 2] = index
            # just to be sure...reinforce dynamics
        j=neurons_tail_ntot[i,0]-1
        neurons_jmp_dyn[i,0, j] = neurons_jmp_dyn_ini[i,0, 2]
        neurons_jmp_ang[i,0, j] = neurons_jmp_ang_ini[i,0, 2]
        neurons_jmp_amp[i,0, j] = neurons_jmp_amp_ini[i,0, 2]
        neurons_jmp_rate[i,0, j] = neurons_jmp_rate_ini[i,0, 2]
        neurons_mass[i,0, j] = neurons_mass_ini[i,0, 2]
        for j in range(2):
            neurons_jmp_dyn[i,0, j] = neurons_jmp_dyn_ini[i,0, j]
            neurons_jmp_ang[i,0, j] = neurons_jmp_ang_ini[i,0, j]
            neurons_jmp_amp[i,0, j] = neurons_jmp_amp_ini[i,0, j]
            neurons_jmp_rate[i,0, j] = neurons_jmp_rate_ini[i,0, j]
            neurons_mass[i,0, j] = neurons_mass_ini[i,0, j]



            

          
    return







def extend_onlytail(i, ntail, tail_level, box_x0, box_xf, box_y0, box_yf,box_nx, box_ny,box_dx, box_dy,box_PBC,neurons_npoints_ini,\
                        neurons_x,neurons_y, neurons_tail_active, neurons_tail_ntot,neurons_bulge_active,neurons_tail_hasparent,\
                        neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
                        neurons_nk2, neurons_nk3, neurons_velocity, neurons_tail_node,neurons_mass, neurons_size,\
                        neurons_k2, neurons_k2_id, neurons_k3, neurons_k3_id, tail_k2, neurons_length,\
                        tail_l0, tail_k3, neurons_cosangle, tail_nmax, neurons_mass_ini,\
                        tail_mass_max, tail_l, tail_angle, rng_vals, pot_field,\
                        neurons_jmp_dyn_ini, neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini):




    # this routine is called when tail has only 3 points. It add all points to nmax
                
    rng_vals = np.random.rand(100000)
    r_idx=i
    
    xmin = box_x0
    xmax = box_xf
    ymin = box_y0
    ymax = box_yf
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    # ntail should be 1
    if (ntail !=1):
        print(f'n tail not 1 ? {ntail}')

    # check we increase tail if ther is no other existing
    nr_tails=sum(neurons_tail_active[i,:])

    if (nr_tails > 1):
        return

    if (neurons_bulge_active[i,ntail-1] == 1):
        return
    #taillevel should be 1

    OK=True
    if box_PBC == False:
        if (neurons_x[i,0, 1]/dx < 10):
            OK = False
        if (neurons_x[i,0, 1]/dx > nx-10):
            OK = False
    if (neurons_y[i,0, 1]/dy < 10):
        OK = False
    if (neurons_y[i,0, 1]/dy > ny-10):
        OK = False
        
    if ((neurons_tail_ntot[i,ntail-1] == neurons_npoints_ini[i]) and (OK == True)):

        # in this version, before tring to extend, if potential is high enough, do a reversal of polarity 0-> 2  2-> 1 all velocities reversed

 
        xend = neurons_x[i, 0, 2]
        yend = neurons_y[i, 0, 2]
            
        ixend = int(round(xend/dx))
        if (ixend < 0):
            ixend = 0
        if (ixend > nx-1):
            ixend = nx-1
        iyend = int(round(yend/dy))
        if (iyend < 0):
            iyend = 0
        if (iyend > ny-1):
            iyend = ny-1
        
        x1 = neurons_x[i, 0, 0]
        y1 = neurons_y[i, 0, 0]
            
        ix1 = int(round(x1/dx))
        if (ix1 < 0):
            ix1 = 0
        if (ix1 > nx-1):
            ix1 = nx-1
        iy1 = int(round(y1/dy))
        if (iy1 < 0):
            iy1 = 0
        if (iy1 > ny-1):
            iy1 = ny-1
        pot1=_bilinear_interp(pot_field, xend, yend, xmin, ymin, dx, dy, nx, ny)
        pot2=_bilinear_interp(pot_field, x1, y1, xmin, ymin, dx, dy, nx, ny)


        if ((pot1 > 1.0 ) and (pot1 > pot2 )):

            tx=neurons_x[i,0, 2]
            neurons_x[i,0, 2]=neurons_x[i,0, 0]
            neurons_x[i,0, 0]=tx
            ty=neurons_y[i,0, 2]
            neurons_y[i,0, 2]=neurons_y[i,0, 0]
            neurons_y[i,0, 0]=ty
            tx=-neurons_velocity[i,0, 2,0]
            neurons_velocity[i,0, 2,0]=-neurons_velocity[i,0, 0,0]
            neurons_velocity[i,0, 0,0]=tx
            ty=-neurons_velocity[i,0, 2,1]
            neurons_velocity[i,0, 2,1]=-neurons_velocity[i,0, 0,1]
            neurons_velocity[i,0, 0,1]=ty
            neurons_velocity[i,0, 1,:]=-neurons_velocity[i,0, 1,:]
            #print(f'inversion {i}')
        



        nk2=2
        nk3=1        ### TO CHECK
        if (nk2 != neurons_nk2[i,ntail-1]):
            #print(f'wrong nk2 {nk2}-> force')
            neurons_nk2[i,ntail-1]=nk2
        if (nk3!= neurons_nk3[i,ntail-1]):
            #print(f'wrong nk3 {nk3}-> force')
            neurons_nk3[i,ntail-1]=nk3




        nmax = tail_nmax[tail_level-1] ### LEVEL
        nptail = nmax
 
        r_idx+=1
        npoints_ini = neurons_npoints_ini[i] # 3 points
        startpoint = neurons_npoints_ini[i]  # 3 is node

        neurons_tail_ntot[i,ntail-1] = npoints_ini
        neurons_tail_node[i,ntail-1] = startpoint  
        if (startpoint !=3):
            print(f' startpoint not 3 !!! node wrong {startpoint}')
        # print(f' mass ini {ntail-1} {npoints_ini}')
        m_ini = neurons_mass_ini[i,ntail-1, npoints_ini-1] # this is the mass of point 3
        m_fin = tail_mass_max[tail_level-1]           # this is the mass on the tail line
        
        

        for j in range(nptail):
            # set up the masses
            index = startpoint+j  # +1  we modify mass of point 3 as well
            # print(f"nptail {i} {nptail}")
            ##neurons_mass[i,ntail-1, index-1] = m_ini + \
            ##    (j+1)*(m_fin-m_ini)/nptail
            ## V42 mases dividred except last
            neurons_mass[i,ntail-1, index-1] = m_fin
        
        j = nptail
        index = startpoint+j 
        neurons_mass[i,ntail-1, index-1] = m_ini  # last point treates separately

        for j in range(nptail):
            # set up positions
            #: in this version it is the soma and the first node that fix the global direction
            npoints = neurons_tail_ntot[i,ntail-1]  # this is the last point
            index = startpoint+j+1
            x1 = neurons_x[i,ntail-1, index-3]
            y1 = neurons_y[i,ntail-1, index-3]
            x2 = neurons_x[i,ntail-1, index-2]
            y2 = neurons_y[i,ntail-1, index-2]
            ddx = x2-x1  #### TO  DEBUG FOR PBC####
            if box_PBC:
                ddx = ddx- xmax * round(ddx / xmax)
            ddy = y2-y1
            norm = math.sqrt(ddx*ddx+ddy*ddy)
            if (norm < 0.01):
                norm = 0.01
            mycos = ddx/norm
            if (mycos < -1.0):
                mycos = -1.0
            if (mycos > 1.0):
                mycos = 1.0
            ang1 = np.arccos(mycos)

            if (ddy < 0):
                ang1 = 2*pi-ang1

                # pick up angle and length
            amplitude = tail_l0[tail_level-1]    * (1+tail_l[tail_level-1]*2*(rng_vals[r_idx]-0.5))
            r_idx+=1
            ang = ang1+2.0*(rng_vals[r_idx]-0.5) *  tail_angle[tail_level-1]*pi/180.0  
            r_idx+=1
            ddx = amplitude*np.cos(ang)
            ddy = amplitude*np.sin(ang)

            x = neurons_x[i,ntail-1, index-2]+ddx
            y = neurons_y[i,ntail-1, index-2]+ddy
            if (box_PBC == False):
                if ((x > xmin) and (x < xmax) and (y > ymin+2*dy) and (y < ymax-2*dy)):
                    # really add point
                    # print(f"index-1 ddx ddy {index-1} {ddx} {ddy}")
                    neurons_x[i,ntail-1, index-1] = x
                    neurons_y[i,ntail-1, index-1] = y
                    neurons_nk2[i,ntail-1] += 1
                    nk2 = neurons_nk2[i,ntail-1]
                    neurons_k2[i,ntail-1, nk2-1] = tail_k2[tail_level-1] 
                    neurons_k2_id[i,ntail-1, nk2-1, 0] = index-1
                    neurons_k2_id[i,ntail-1, nk2-1, 1] = index

                    neurons_length[i,ntail-1, index-2,index-1] = amplitude ### DEBUG tail_l0[tail_level-1] ###FAUX LEVEL
                    neurons_size[i,ntail-1, index -1] = neurons_size[i,ntail-1, npoints_ini-1]
                    neurons_jmp_dyn[i,ntail-1, index - 1] = neurons_jmp_dyn[i,ntail-1, index-2]
                    neurons_jmp_dyn[i,ntail-1, index-2] = 0
                    neurons_jmp_ang[i,ntail-1, index -1] = neurons_jmp_ang[i,ntail-1, index-2]
                    neurons_jmp_ang[i,ntail-1, index-2] = 0
                    neurons_jmp_rate[i,ntail-1, index -1] = neurons_jmp_rate[i,ntail-1, index-2]
                    neurons_jmp_rate[i,ntail-1, index-2] = 0
                    neurons_jmp_amp[i,ntail-1, index -1] = neurons_jmp_amp[i,ntail-1, index-2]
                    neurons_jmp_amp[i,ntail-1, index-2] = 0
                    
                    
                    
                    neurons_nk3[i,ntail-1] += 1
                    nk3 = neurons_nk3[i,ntail-1]
                    neurons_k3_id[i,ntail-1, nk3-1, 0] = index-2
                    neurons_k3_id[i,ntail-1, nk3-1, 1] = index-1
                    neurons_k3_id[i,ntail-1, nk3-1, 2] = index
                    # neurons_velocity[index-1,:]=0.0
                    neurons_cosangle[i,ntail-1, nk3 - 1] = neurons_cosangle[i,ntail-1, nk3-2]
                    neurons_k3[i,ntail-1, nk3-1] = tail_k3[tail_level-1] 


                    neurons_tail_ntot[i,ntail-1] += 1

            else:
                if ((y > ymin+2*dy) and (y < ymax-2*dy)):
                    # really add point
                    # print(f"index-1 ddx ddy {index-1} {ddx} {ddy}")

                    if (x >= xmax):
                        x -= xmax
                    if (x <= xmin):
                        x += xmax

                    neurons_x[i,ntail-1, index-1] = x
                    neurons_y[i,ntail-1, index-1] = y
                    neurons_nk2[i,ntail-1] += 1
                    nk2 = neurons_nk2[i,ntail-1]
                    neurons_k2[i,ntail-1, nk2-1] = tail_k2[tail_level-1]
                    neurons_k2_id[i,ntail-1, nk2-1, 0] = index-1
                    neurons_k2_id[i,ntail-1, nk2-1, 1] = index

                    neurons_length[i,ntail-1, index-2,index-1] = amplitude ###DEBUGtail_l0[tail_level-1]
                    neurons_size[i,ntail-1, index -1] = neurons_size[i,ntail-1, npoints_ini-1]
                    neurons_jmp_dyn[i,ntail-1, index - 1] = neurons_jmp_dyn[i,ntail-1, index-2]
                    neurons_jmp_dyn[i,ntail-1, index-2] = 0
                    neurons_jmp_ang[i,ntail-1, index - 1] = neurons_jmp_ang[i,ntail-1, index-2]
                    neurons_jmp_ang[i,ntail-1, index-2] = 0
                    neurons_jmp_rate[i,ntail-1, index - 1] = neurons_jmp_rate[i,ntail-1, index-2]
                    neurons_jmp_rate[i,ntail-1, index-2] = 0
                    neurons_jmp_amp[i,ntail-1, index - 1] = neurons_jmp_amp[i,ntail-1, index-2]
                    neurons_jmp_amp[i,ntail-1, index-2] = 0
                    neurons_nk3[i,ntail-1] += 1
                    nk3 = neurons_nk3[i,ntail-1]
                    neurons_k3_id[i,ntail-1, nk3-1, 0] = index-2
                    neurons_k3_id[i,ntail-1, nk3-1, 1] = index-1
                    neurons_k3_id[i,ntail-1, nk3-1, 2] = index
                    # neurons_velocity[index-1,:]=0.0
                    neurons_cosangle[i,ntail-1, nk3 - 1] = neurons_cosangle[i,ntail-1, nk3-2]
                    neurons_k3[i,ntail-1, nk3-1] = tail_k3[tail_level-1]

                    neurons_tail_ntot[i,ntail-1] += 1

#here we check that the tail was not created in a high potential area, if so we rotate it
        npoints = neurons_tail_ntot[i,ntail-1]  # this is the last point of tail
        xend = neurons_x[i, ntail-1, npoints-1]
        yend = neurons_y[i, ntail-1, npoints-1]
        xendm1 = neurons_x[i, ntail-1, npoints-2]
        yendm1 = neurons_y[i, ntail-1, npoints-2]
        ixendm1 = int(round(xendm1/dx))
        if (ixendm1 < 0):
            ixendm1 = 0
        if (ixendm1 > nx-1):
            ixendm1 = nx-1
        iyendm1 = int(round(yendm1/dy))
                    
        if (iyendm1 < 0):
            iyendm1 = 0
        if (iyendm1 > ny-1):
            iyendm1 = ny-1


        ixend = int(round(xend/dx))
        if (ixend < 0):
            ixend = 0
        if (ixend > nx-1):
            ixend = nx-1
        iyend = int(round(yend/dy))
        if (iyend < 0):
            iyend = 0
        if (iyend > ny-1):
            iyend = ny-1
        
        x1 = neurons_x[i, ntail-1, 1] # soma
        y1 = neurons_y[i, ntail-1, 1]
            
        ix1 = int(round(x1/dx))
        if (ix1 < 0):
            ix1 = 0
        if (ix1 > nx-1):
            ix1 = nx-1
        iy1 = int(round(y1/dy))
        if (iy1 < 0):
            iy1 = 0
        if (iy1 > ny-1):
            iy1 = ny-1
        pot1=_bilinear_interp(pot_field, xend, yend, xmin, ymin, dx, dy, nx, ny)
        pot2=_bilinear_interp(pot_field, x1, y1, xmin, ymin, dx, dy, nx, ny)
        if ((pot1> 1.0 ) and (pot1>pot2 )):
            #print(f' rotating end of tail {i} {ntail} {npoints} {xend} {yend} {ixend} {iyend} {pot_field[ixend,iyend]} {pot_field[ix1,iy1]}')
            # find direction of the field at the end of tail  neuron rebounds on it
            fx = pot1 - _bilinear_interp(pot_field, xendm1, yend, xmin, ymin, dx, dy, nx, ny) ####WRONG DEBUG
            fy = pot1 - _bilinear_interp(pot_field, xend, yendm1, xmin, ymin, dx, dy, nx, ny) #### should be the gradient

            norm = math.sqrt(fx*fx+fy*fy)
            if (norm > 0.01):
                fx = fx/norm
                fy = fy/norm
                # find angle of the field
                ang_field = np.arctan2(-fy, -fx)
                # find angle of the neuron
                ddx = xend - x1
                if box_PBC: ###DEBUG FOR PBC
                    ddx = ddx- xmax * round(ddx / xmax)
                ddy = yend - y1
                norm = math.sqrt(ddx*ddx+ddy*ddy)
                if (norm > 0.01):
                    ddx = ddx/norm
                    ddy = ddy/norm
                    ang_neuron = np.arctan2(ddy, ddx)
                    # find angle difference and rotate neuron accordingly
                    ang_diff = ang_neuron - ang_field  # this value is between -2pi and 2pi
                    #if above pi change sign and value to have between 0 and pi
                    if (ang_diff > pi):
                        ang_diff = ang_diff - 2*pi
                    if (ang_diff < -pi):
                        ang_diff = ang_diff + 2*pi
                    # if this angle is smaller than PI/2 do nothing, else rebound
                    if (abs(ang_diff) > pi/2.0):
                        sign=ang_diff/abs(ang_diff)
                        ang_diff = -sign*abs(ang_diff-pi/2.0)*2.0  # this is a rebound
                        #ang_diff = -sign*abs(ang_diff-pi/2.0)*1.0  # this is alignment with perpendicular to field

                        cos_diff = np.cos(ang_diff)
                        sin_diff = np.sin(ang_diff)
                        for j in range(neurons_tail_ntot[i,ntail-1]):
                            xj = neurons_x[i,ntail-1,j] - x1
                            if box_PBC:
                                xj = xj- xmax * round(xj / xmax)  ###DEBUG FOR PBC
                            yj = neurons_y[i,ntail-1,j] - y1
                            neurons_x[i,ntail-1,j] = x1 + cos_diff*xj - sin_diff*yj
                            neurons_y[i,ntail-1,j] = y1 + sin_diff*xj + cos_diff*yj
                            # speeds also habe to be rotated
                            vxj = neurons_velocity[i,ntail-1,j,0]
                            vyj = neurons_velocity[i,ntail-1,j,1]
                            neurons_velocity[i,ntail-1,j,0] = cos_diff*vxj - sin_diff*vyj
                            neurons_velocity[i,ntail-1,j,1] = sin_diff*vxj + cos_diff*vyj
        j=npoints-1
        neurons_jmp_dyn[i,ntail-1, j] = neurons_jmp_dyn_ini[i,0, 2]
        neurons_jmp_ang[i,ntail-1, j] = neurons_jmp_ang_ini[i,0, 2]
        neurons_jmp_amp[i,ntail-1, j] = neurons_jmp_amp_ini[i,0, 2]
        neurons_jmp_rate[i,ntail-1, j] = neurons_jmp_rate_ini[i,0, 2]
        neurons_mass[i,ntail-1, j] = neurons_mass_ini[i,0, 2]


    return


def create_branch(i, itail, taillevel, n_levels, n_tails, box_x0, box_xf, box_y0, box_yf,box_nx, box_ny,box_dx, box_dy,box_PBC,neurons_npoints_ini,\
    neurons_x,neurons_y, neurons_tail_ntot,\
    neurons_tail_active, neurons_tail_level, neurons_tail_nchildren, neurons_tail_hasparent,\
    neurons_tail_node, neurons_tail_angle, neurons_bulge_active,\
    tail_delete_rate,neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
    neurons_nk2, neurons_nk3, neurons_velocity, neurons_mass, neurons_size,\
    neurons_k2, neurons_k2_id, neurons_k3, neurons_k3_id, tail_k2, neurons_length,\
    tail_l0, tail_k3, neurons_cosangle, neurons_mass_ini,\
    tail_mass_max, tail_l, tail_angle, tail_angle_start_min, tail_angle_start_max, rng_vals,potvalue,\
    neurons_jmp_dyn_ini, neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini):


    # in this V1 there is a single routine for creating a branch (all plays with levels) on itail
    # when we create a branch, it is attached to a node on a parent.
    # the nr of points of the branch is the same as that of the parent tail
    # the branch is created by rotating points after the node for a given angle
    # rotation is always towards outside of main axis
    # for simplicity here, we will use all same angles, massmax springs etc for all levels (otherwise needs to identify levels and re-distribute coefficients)
    # the number of tails is the number of LOOSE ENDS (not segments).   
    # each time a tail is added a child, its level INCREASES by 1

    # a level 1 has no problem, can create a child and transform into level 2 (the child is level2 as well)
    # a level 2 can have only one child OR one parent
    # a level 3 is created on a level 2

    # a level 4 can have :
    #                     3 children
    #                     or 2 children and 1 parent
    #                     or 1 child    and 1 parent
    #                     or 1 parent

    #identify characteristics
    # itail is the NEW BRANCH and it will have a level taillevel.

    nr_children = neurons_tail_nchildren[i,itail-1]  # this is ther nr of children of the new tail, should be 0 at creation
    if (nr_children !=0):
                #print(f' problem, tail should have no child at creation {nr_children} -> forced 0')
                neurons_tail_nchildren[i,itail-1]=0

    nr_parent=neurons_tail_hasparent[i,itail-1]
    node_on_parent=neurons_tail_node[i,nr_parent-1]


    # double check
    if (taillevel != neurons_tail_level[i, itail-1]):
        print(f' level to create is incorrect {taillevel} {neurons_tail_level[i, itail-1]}')

    is_bulge_active = sum(neurons_bulge_active[i,:])


    can_make_child=True
    if (is_bulge_active >0):
        can_make_child=False

    # simply, if not on last level, one can add a child
    if( taillevel > n_levels):
            can_make_child=False

    if (not can_make_child):
        return


    rng_vals = np.random.rand(100000)
    r_idx=i
    xmin = box_x0
    xmax = box_xf
    ymin = box_y0
    ymax = box_yf
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    

    OK = True
    npoints = neurons_tail_ntot[i,nr_parent-1] # should be the max - that tail should have been extenbded to nmax before branching
    nr_existing_tails=sum(neurons_tail_active[i,:])
    inode_parent = neurons_tail_node[i,nr_parent-1] # this is the node on the parent


    # in this revised version, the whole structure is created with the first tail (to allo)
    first_branch = False

    if (nr_existing_tails == 1):
        # create the whole fractallike struture (nodes and angles wiyth sign)
        first_branch = True


    if (inode_parent == neurons_tail_ntot[i,nr_parent-1]-1 ):
        OK = False
        # we skip if no more point after node to create branch. The parent is already attached to the forelast point.

    '''if (npoints != 7) :
        print(f' problem points i {i} itail {itail} npoints {npoints}')'''

    if (npoints < 3):
        OK = False ## JY HAS TO BE CORRECTED FOR PBC
    if (box_PBC==False):
        if (neurons_x[i,0, 1]/dx < 10):
            OK = False
        if (neurons_x[i,0, 1]/dx > nx-10):
            OK = False
    if (neurons_y[i,0, 1]/dy < 10):
        OK = False
    if (neurons_y[i,0, 1]/dy > ny-10):
        OK = False

    if (OK == True):
        # in this version, no branch on soma, only on nodes, which can be point nr3 at min.
        i_newtail=itail
        level_newtail=taillevel # neurons_tail_level[i,i_newtail-1]  # useless already in taillevel as argument
        



        get_out_of_here=False
        r_idx+=1
        if (first_branch == True):  # fior the first we pick the node. The others are filled automatically
            ##inode = inode_parent + 1 + int(( npoints-inode_parent-1 ) *rng_vals[r_idx])
            inode = inode_parent -1 + int(( npoints-inode_parent+1) *rng_vals[r_idx])  ### check if it works
            r_idx+=1
            if (inode ==0):
                print(f' BUG here inode 0 on first branch {inode}')
        else:
            inode=neurons_tail_node[i,itail-1] # this was fixed at creation of first branch
            if (inode ==0):
                print(f' BUG here inode 0 on not first branch {itail}  {nr_parent} {inode}')
                for t in range(n_tails):
                    print(f' inodes: {t}, {neurons_tail_node[i,t-1]}')



        for j in range(n_tails):
            if (neurons_tail_active[i,j]==1):
                if((inode == neurons_tail_node[i,j]) and( nr_parent == neurons_tail_hasparent[i,j])):  
                    # do not activate !!! this node on parent is already used
                    
                    get_out_of_here=True
                    # except if that parent is tail 1 (can share node)
                    #if (nr_parent == 1) and ( j !=2 ):
                    #    get_out_of_here=False  
                        
        if (inode > neurons_tail_ntot[i,nr_parent-1]-1):
            # do not activate !!!
            get_out_of_here=True

        if (inode == inode_parent):
            # do not activate !!!
            get_out_of_here=True

        if (get_out_of_here == True):
            neurons_tail_active[i,i_newtail-1] = 0
            return

        # activate new tail
        neurons_tail_active[i,i_newtail-1] = 1
        # we here enforc that inode children > inode parent

        #print(' we create a branch on inode {inode}')



        #print(f' start activation on {i} tail {i_newtail}')

        # the parent level is NOT CHANGED but it has one more child
        ####neurons_tail_level[i,i_newtail-1] = level_newtail # removed. Tail levels fixed once and foir all at startup
        neurons_tail_nchildren[i,nr_parent-1] +=1
        level_parent=neurons_tail_level[i,nr_parent-1]
        
        if (first_branch == True):
            neurons_tail_node[i,i_newtail-1] = inode
            if (inode ==0):
                print(f'BUG inode 0 on first branch {inode}')



        startpoint = neurons_tail_node[i,i_newtail-1] #  inode
        # neurons_tail_node[i,i_newtail-1] = inode   nodes had been set up with first branch, so no need to set up again

        # all points before startpoint/inode (included) are simply copied from parent for security
        neurons_nk2[i,i_newtail-1] = 0
        neurons_nk3[i,i_newtail-1] = 0
        neurons_tail_ntot[i,i_newtail-1] = 0
        
        nk2 = 0
        nk3 = 0
        neurons_length[i,i_newtail-1, :, :] = neurons_length[i,nr_parent-1, :, :]
        end_parent=neurons_tail_ntot[i,nr_parent-1]

        for j in range(npoints):    
            # for safety copy everything from parent(full 7 pooints) excdept x y k2 k3 data
            neurons_x[i,i_newtail-1, j] = neurons_x[i,nr_parent-1, j]
            neurons_y[i,i_newtail-1, j] = neurons_y[i,nr_parent-1, j]
            neurons_velocity[i,i_newtail-1, j, 0] = neurons_velocity[i,nr_parent-1, j, 0]
            neurons_velocity[i,i_newtail-1, j, 1] = neurons_velocity[i,nr_parent-1, j, 1]
            neurons_mass[i,i_newtail-1, j] = neurons_mass[i,nr_parent-1, j]
            neurons_size[i,i_newtail-1, j] = neurons_size[i,nr_parent-1, j]
            neurons_jmp_dyn[i,i_newtail-1, j] = neurons_jmp_dyn[i,nr_parent-1, j]
            neurons_jmp_ang[i,i_newtail-1, j] = neurons_jmp_ang[i,nr_parent-1, j]
            neurons_jmp_amp[i,i_newtail-1, j] = neurons_jmp_amp[i,nr_parent-1, j]
            neurons_jmp_rate[i,i_newtail-1, j] = neurons_jmp_rate[i,nr_parent-1, j]
            
            
        #if (i_newtail ==4):
            #print(f'tail 4 : parent {nr_parent}')
        for j in range(startpoint): # all k2/k3 data before startpoint/inode (included) are actually copied
            neurons_tail_ntot[i,i_newtail-1] += 1
            neurons_x[i,i_newtail-1, j] = neurons_x[i,nr_parent-1, j]
            neurons_y[i,i_newtail-1, j] = neurons_y[i,nr_parent-1, j]

            if (j > 0):
                nk2 = nk2+1
                neurons_nk2[i,i_newtail-1]=nk2
                neurons_k2[i,i_newtail-1, nk2-1] = neurons_k2[i,nr_parent-1, nk2-1]
                neurons_k2_id[i,i_newtail-1, nk2-1, :] = neurons_k2_id[i,nr_parent-1, nk2-1, :]

            if (j > 1):
                nk3 += 1
                neurons_nk3[i,i_newtail-1]=nk3
                ###neurons_cosangle[i,i_newtail-1, nk3-1] = neurons_cosangle[i,nr_parent-1, nk3-1]  **** this can be a bug !!!! 
                neurons_cosangle[i,i_newtail-1, nk3-1] = np.cos(tail_angle[level_parent-1]*pi/180.0) #****
                neurons_k3[i,i_newtail-1, nk3-1] = tail_k3[level_parent-1]  #**** neurons_k3[i,nr_parent-1, nk3-1]
                neurons_k3_id[i,i_newtail-1, nk3-1, 0] = j-2 ###neurons_k3_id[i,nr_parent-1, nk3-1, :]  **** this can be a bug !!!!
                neurons_k3_id[i,i_newtail-1, nk3-1, 1] = j-1
                neurons_k3_id[i,i_newtail-1, nk3-1, 2] = j  
        # go on with following points
        # the branch we grw has the same size as its parent
        # up to now the tail length was maximum, hjere we allow shorter
        ##nptail = neurons_tail_ntot[i,nr_parent-1] - startpoint
        nptail = 1+int(rng_vals[r_idx]*(neurons_tail_ntot[i,nr_parent-1] - startpoint)) # this should be in future versions, but needs adapatation  for copying jmp data etc from parent
        
        #only relevant data should be copied from parent


        # whatever the case the mass ini is that of point 3

        for j in range(nptail):

            index = inode+j+1
            iparent = end_parent-nptail+j+1  # this is the corresponding point on parent ***

            neurons_mass[i,i_newtail-1, index-1] = neurons_mass[i,nr_parent-1, iparent-1]
            neurons_size[i,i_newtail-1, index-1] = neurons_size[i,nr_parent-1, iparent-1]
            neurons_jmp_dyn[i,i_newtail-1, index-1] = neurons_jmp_dyn[i,nr_parent-1, iparent-1]
            neurons_jmp_ang[i,i_newtail-1, index-1] = neurons_jmp_ang[i,nr_parent-1, iparent-1]
            neurons_jmp_amp[i,i_newtail-1, index-1] = neurons_jmp_amp[i,nr_parent-1, iparent-1]
            neurons_jmp_rate[i,i_newtail-1, index-1] = neurons_jmp_rate[i,nr_parent-1, iparent-1]




            if (j == 0):
                # the first branch point is attached to inode and makes an angle between min and max with inode+1 on other channel
                # rotate vector by +- this angle TOWARDS THE OUTSIDE OF NEURON
                # if only one tail active, any direction
                # for this we compute the center of gravity of all active tail endpoints
                # be careful in case of PBC, we have to compute the COG with PBC in mind
                         
                x0 = neurons_x[i,nr_parent-1, inode-1]
                x1 = neurons_x[i,nr_parent-1, inode]
                y0 = neurons_y[i,nr_parent-1, inode-1]
                y1 = neurons_y[i,nr_parent-1, inode]  # vector on parent
                dxx = x1-x0
                if box_PBC:
                    dxx = dxx- xmax * round(dxx / xmax) ###DEBUG FOR PBC
                dyy = y1-y0
                norm = math.sqrt(dxx*dxx+dyy*dyy)
                ang1 = np.arctan2(dyy, dxx)

                angmin = tail_angle_start_min[taillevel-1]  #as an argument taillevel is the level of the colling tail (1,2...) since we increase by 1 there is no -1 here
                angmax = tail_angle_start_max[taillevel-1]
                ang_inc = angmin+rng_vals[r_idx]*(angmax-angmin)
                #print(f' ang_inc {ang_inc} {angmin} {angmax}')

                # the angle is OK but the distance can be problematic. Let's use the tail l0 instead               
                pr = rng_vals[r_idx]
                r_idx+=1
    
                if (pr < 0.5):
                    ang_inc = -ang_inc
                if (first_branch == False):  # the angle to use is already in neuros_tail_angle
                    ang_inc = neurons_tail_angle[i,i_newtail-1]

                ang = ang1+ang_inc*pi/180.0
                true_cosangle=np.cos(ang_inc*pi/180.0) 
                

                parent_length=math.sqrt(dxx*dxx+dyy*dyy)
                true_length = (1+tail_l[taillevel-1]*2 * \
                             (rng_vals[r_idx]-0.5))*tail_l0[taillevel-1]

                cosang = np.cos(ang)
                sinang = np.sin(ang)
                dxp = true_length*cosang
                dyp = true_length*sinang


                neurons_x[i,i_newtail-1, index -1] = neurons_x[i,i_newtail-1, index-2]+dxp  #OK
                if box_PBC:
                    if (neurons_x[i,i_newtail-1, index -1]> xmax):
                        neurons_x[i,i_newtail-1, index -1] -= xmax
                    if (neurons_x[i,i_newtail-1, index -1]< xmin):
                        neurons_x[i,i_newtail-1, index -1] += xmax

                neurons_y[i,i_newtail-1, index -1] = neurons_y[i,i_newtail-1, index-2]+dyp
                neurons_velocity[i,i_newtail-1, index -1, 0] = 0.0
                neurons_velocity[i,i_newtail-1, index -1, 1] = 0.0
                
                #  If this is the first branch, then we fix all other angles and nodes here.
                # this first branch is a level 2
                if (nr_existing_tails == 1): # inewtail should be 2 ideally - to check
                    if (i_newtail != 2):
                        print(f' problem with tail numbering {i_newtail} {nr_existing_tails}')
                    ix=i_newtail-1
                    neurons_tail_angle[i,0] = 0.0 ### CHECK
                    neurons_tail_angle[i,ix] = ang_inc
                    if (n_levels >= 3):
                        neurons_tail_angle[i,ix+1] = ang_inc/2.0
                        neurons_tail_angle[i,ix+2] = -ang_inc/2.0

                    if (n_levels >= 4):
                        neurons_tail_angle[i,ix+3] = ang_inc/2.0
                        neurons_tail_angle[i,ix+4] = -ang_inc/2.0
                        neurons_tail_angle[i,ix+5] = -ang_inc/2.0
                        neurons_tail_angle[i,ix+6] = ang_inc/2.0


                    
                    if (n_levels >= 5):
                        neurons_tail_angle[i,ix+7] = ang_inc/2.0
                        neurons_tail_angle[i,ix+8] = -ang_inc/2.0
                        neurons_tail_angle[i,ix+9] = -ang_inc/2.0
                        neurons_tail_angle[i,ix+10] = ang_inc/2.0
                        neurons_tail_angle[i,ix+11] = -ang_inc/2.0
                        neurons_tail_angle[i,ix+12] = ang_inc/2.0
                        neurons_tail_angle[i,ix+13] = ang_inc/2.0
                        neurons_tail_angle[i,ix+14] = -ang_inc/2.0




                    # let's fix the nodes as well
                    # there is a limit depending on np_tail
                    # if np_tail=3 can make level 2 node is inode+2
                    # if np_tail=4 can make level 3 node is inode+3
                    # if np_tail=5 can make level 4 node is inode+4
                    # 
                    #  
                    if (nptail ==1):
                        # can not make tail 2
                        neurons_tail_active[i,i_newtail-1] == 0
                        neurons_tail_nchildren[i,nr_parent-1] -=1  ###TOCHECK JY BUGGGGGG
                        go_back=True
                        

                    # check all other nodes on parent

                    if (nptail >=2):
                        if (n_levels >= 3):

                            neurons_tail_node[i,ix+1] = inode+1
                            neurons_tail_node[i,ix+2] = inode+1
                    if (nptail >=3):
                        if (n_levels >= 3):
                            pr = int(rng_vals[r_idx]*(nptail-2))
                            r_idx+=1
                            neurons_tail_node[i,ix+1] = inode+2+pr ## inode+2
                            pr = int(rng_vals[r_idx]*(nptail-2))
                            r_idx+=1
                            neurons_tail_node[i,ix+2] = inode+1+pr ##  inode+2
                    '''if (nptail >=4):
                        if (n_levels >= 4):
                            neurons_tail_node[i,ix+3] = inode+3
                            neurons_tail_node[i,ix+4] = inode+3
                            neurons_tail_node[i,ix+5] = inode+3
                            neurons_tail_node[i,ix+6] = inode+3
                    if (nptail >=5):
                        if (n_levels >= 5):
                            neurons_tail_node[i,ix+7] = inode+3
                            neurons_tail_node[i,ix+8] = inode+3
                            neurons_tail_node[i,ix+9] = inode+3
                            neurons_tail_node[i,ix+10] = inode+3
                            neurons_tail_node[i,ix+11] = inode+3
                            neurons_tail_node[i,ix+12] = inode+3
                            neurons_tail_node[i,ix+13] = inode+3
                            neurons_tail_node[i,ix+14] = inode+3

                    '''
                
                x = neurons_x[i,i_newtail-1, index-1]
                y = neurons_y[i,i_newtail-1, index-1]

                go_back=False
                if (box_PBC == False):
                    if (x < xmin) or (x > xmax) or (y < ymin+2*dy) or (y > ymax-2*dy):
                        neurons_tail_active[i,i_newtail-1] == 0
                        neurons_tail_nchildren[i,nr_parent-1] -=1  # remove this attempted child
                        print(f' we are outside box {x} {y} {xmin} {xmax} {ymin} {ymax}{dxx} {dyy} {dxp} {dyp} {nr_existing_tails} {i_newtail} {nr_parent} {inode}')
                        print(f' for i {i} inewtail {i_newtail} index {index} level {taillevel} k2 {tail_k2[taillevel-1]} tail_l0 {tail_l0[taillevel-1]} ')
                        go_back=True
                        return
                        
                    if (potvalue[int(x),int(y)] > 0.1):
                        neurons_tail_active[i,i_newtail-1] == 0
                        neurons_tail_nchildren[i,nr_parent-1] -=1  # remove this attempted child
                        #print(f' we are on a potential {x} {y} {potvalue[int(x),int(y)]}')
                        go_back=True
                        return
                else:
                    if (y < ymin+2*dy) or (y > ymax-2*dy):
                        neurons_tail_active[i,i_newtail-1] == 0
                        neurons_tail_nchildren[i,nr_parent-1] -=1  # remove this attempted child
                        go_back=True
                        return
                    # we stop the j loop here
                
                nk2 += 1   
                neurons_nk2[i,i_newtail-1] = nk2            
                neurons_k2_id[i,i_newtail-1, nk2-1, 0] = inode
                neurons_k2_id[i,i_newtail-1, nk2-1, 1] = index
                neurons_k2[i,i_newtail-1, nk2-1] = tail_k2[taillevel-1]  # no need for +1 it is here
                neurons_length[i,i_newtail-1, inode-1,index-1] =true_length # was tail_l0[taillevel-1]  # if we add a child on a level1"
                #if (true_length > 7.0):
                    #print(f' truelengh very large {true_length}')
                #print(f' level and length {taillevel}, {tail_k2[taillevel-1]}, {true_length} {true_cosangle}')    
                ### CORRECTION HERE ONLY ONE K3 TO AVOID ROTATIONS
                # j==0 there are two k3 to fix
                # inode+1  inode index: angle
                
                # inode-1  inode index: cosang inode-2 inode-1 inode
                nk3 += 1
                neurons_nk3[i,i_newtail-1] = nk3
   
                neurons_k3_id[i,i_newtail-1, nk3-1,  0] = 100 +  inode+1  # this one is on PARENT!!!
                neurons_k3_id[i,i_newtail-1, nk3-1,  1] = inode
                neurons_k3_id[i,i_newtail-1, nk3-1,  2] = index
                neurons_cosangle[i,i_newtail-1, nk3-1] = true_cosangle #was coscosang
                neurons_k3[i,i_newtail-1, nk3-1] = tail_k3[taillevel-1] ### HARCODING MORE RIGID JY BUGGGGG XXXXXX
    
                neurons_tail_ntot[i,i_newtail-1] += 1
    
            if (j >= 1):
                x0 = neurons_x[i,i_newtail-1, index-3]
                x1 = neurons_x[i,i_newtail-1, index-2]
                y0 = neurons_y[i,i_newtail-1, index-3]
                y1 = neurons_y[i,i_newtail-1, index-2]
                ddx = x1-x0
                if box_PBC:
                    ddx = ddx- xmax * round(ddx / xmax) ###DEBUG FOR PBC
                ddy = y1-y0
                norm = math.sqrt(ddx*ddx+ddy*ddy)
                if (norm < 0.01):
                    norm = 0.01
                mycos = ddx/norm
                if (mycos < -1.0):
                    mycos = -1.0
                if (mycos > 1.0):
                    mycos = 1.0
                ang1 = np.arccos(mycos)

                if (ddy < 0):
                    ang1 = 2*pi-ang1

                #norm = math.sqrt(ddx*ddx+ddy*ddy)
                #ang1 = np.arctan2(ddy, ddx)

                amplitude = (1+tail_l[taillevel-1]*2 * \
                             (rng_vals[r_idx]-0.5))*tail_l0[taillevel-1]
                r_idx+=1
                ang = ang1
                ang_inc=+2.0*(rng_vals[r_idx]-0.5) *  tail_angle[taillevel-1]*pi/180.0  # pas juste
                ang=ang1+ang_inc
                r_idx+=1

                cosang = np.cos(ang)
                sinang = np.sin(ang)
                ddx = amplitude*cosang
                ddy = amplitude*sinang
    
                neurons_x[i,i_newtail-1, index -1] = neurons_x[i,i_newtail-1, index-2]+ddx
                neurons_y[i,i_newtail-1, index -1] = neurons_y[i,i_newtail-1, index-2]+ddy
                neurons_velocity[i,i_newtail-1, index -1, 0] = 0.0
                neurons_velocity[i,i_newtail-1, index -1, 1] = 0.0
                
                x = neurons_x[i,i_newtail-1, index-1]
                if box_PBC:
                    if (neurons_x[i,i_newtail-1, index -1]> xmax):
                        neurons_x[i,i_newtail-1, index -1] -= xmax
                    if (neurons_x[i,i_newtail-1, index -1]< xmin):
                        neurons_x[i,i_newtail-1, index -1] += xmax

                y = neurons_y[i,i_newtail-1, index-1]
                if ((x < xmin) or (x > xmax) or (y < ymin+2*dy) or (y > ymax-2*dy) or (potvalue[int(x),int(y)] > 1)):

                    neurons_tail_active[i,i_newtail-1] = 0 ### be careful to give back old values ###            
                    neurons_tail_nchildren[i,nr_parent-1] -=1  # remove this attempted child
                    return


                neurons_nk2[i,i_newtail-1] += 1
                nk2 = neurons_nk2[i,i_newtail-1]
                neurons_k2_id[i,i_newtail-1, nk2-1, 0] = index-1
                neurons_k2_id[i,i_newtail-1, nk2-1, 1] = index
                neurons_k2[i,i_newtail-1, nk2-1] = tail_k2[taillevel-1]
                ### JY DEBUG AMPLITUDE !!!
                ##neurons_length[i,ntail-1, index-2,index-1] = neurons_tail_l0[i,ntail-1]
                neurons_length[i,i_newtail-1, index-2,index-1]=amplitude
                
                neurons_size[i,i_newtail-1, index - 1] = neurons_size[i,i_newtail-1, index-2]
                # there are one k3 to fix
                # [index-3, index-2, index-1]
    
                neurons_nk3[i,i_newtail-1] += 1
                nk3 = neurons_nk3[i,i_newtail-1]
                neurons_k3_id[i,i_newtail-1, nk3-1, 0] = index-2
                neurons_k3_id[i,i_newtail-1, nk3-1, 1] = index-1
                neurons_k3_id[i,i_newtail-1, nk3-1, 2] = index
                
                neurons_cosangle[i,i_newtail-1, nk3-1] = np.cos(pi-ang_inc) #np.cos(tail_angle[taillevel-1]*pi/180.0) # to simplify for sped
                neurons_k3[i,i_newtail-1, nk3-1] = tail_k3[taillevel-1]
    
                neurons_tail_ntot[i,i_newtail-1] += 1
            
        if (neurons_tail_active[i,i_newtail-1] == 1):  ##DEBUG ok we created the branch. Set all velocities back to 0
            neurons_velocity[i,:, :, :] = 0.0
            #print(f' we have created branch {i_newtail } on i {i} with level {taillevel}')
        ### juist to be sure reenforce end of tail dyanamics

        j=startpoint+nptail-1
        neurons_jmp_dyn[i,i_newtail-1, j] = neurons_jmp_dyn_ini[i,0, 2]
        neurons_jmp_ang[i,i_newtail-1, j] = neurons_jmp_ang_ini[i,0, 2]
        neurons_jmp_amp[i,i_newtail-1, j] = neurons_jmp_amp_ini[i,0, 2]
        neurons_jmp_rate[i,i_newtail-1, j] = neurons_jmp_rate_ini[i,0, 2]
        neurons_mass[i,i_newtail-1, j] = neurons_mass_ini[i,0, 2]
      

    return xmax


def create_or_erase_tail(n_neurons, n_tails, n_levels, potvalue, tailrate_field, perturbations_field, slipstick_field, accept_field,mcfield_sign, pmax, pmin, box_temperature,box_timestep,\
                         box_x0, box_xf, box_y0, box_yf, box_nx, box_ny, box_dx, box_dy, box_PBC,box_slipstick_pause,neurons_pause,\
                         neurons_x, neurons_y, neurons_npoints_ini,neurons_length_ini,\
                         neurons_tail_active, neurons_tail_ntot,neurons_bulge_active,\
                         neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
                         neurons_jmp_dyn_ini,neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini,\
                         neurons_nk2, neurons_nk3, neurons_velocity,neurons_tail_node, neurons_tail_angle, neurons_mass, neurons_mass_ini, neurons_size,neurons_size_ini,\
                         neurons_k2, neurons_k2_id, neurons_k3, neurons_k3_id, neurons_length,\
                         neurons_cosangle,\
                         tail_nmax, tail_l, tail_l0, tail_k2, tail_k3,\
                         tail_angle_start_min, tail_angle_start_max, tail_angle,\
                         tail_mass_max,tail_create_rate, tail_delete_rate,\
                         tail_create_multfactor, tail_delete_multfactor,\
                         neurons_tail_level, neurons_tail_hasparent, neurons_tail_nchildren, neurons_branches,friction_field,box_slipstick_amplitude,\
                         neurons_gc_split, neurons_L2,neurons_L3,neurons_NK_freq):

    # 4activate or deactivate a tail for all neurons

    xmin = box_x0
    xmax = box_xf
    ymin = box_y0
    ymax = box_yf
    nx = box_nx
    ny = box_ny
    deltax = xmax/nx
    deltay = ymax/ny
    rng=np.random.default_rng()
    rng_vals=rng.random(1000*n_neurons)
    r_idx=1

    for i in range(n_neurons):
        if (neurons_pause[i]>0):
            neurons_pause[i]-=1
        
        if (neurons_pause[i]==0): 
            x = neurons_x[i, 0, 1]
            y = neurons_y[i, 0, 1]
            
            ix = int(round(x/deltax))
            if (ix < 0):
                ix = 0
            if (ix > nx-1):
                ix = nx-1
            iy = int(round(y/deltay))
            if (iy < 0):
                iy = 0
            if (iy > ny-1):
                iy = ny-1

        
            # first : look for ERASE

            # change in this version: any tail can be erased, but then levels have to change
            # carefule if this is the ONLY active tail, one should not erase the first 3 points
            # if this is the only tail, either simple erase or Slip-stick
            # 
            # case 1 : only one tail : possible slipstick
            # 
            nactive=sum(neurons_tail_active[i,:])
            if (nactive == 1):
                #identify this only active tail, it should be a level 1
                for j in range(n_tails): 
                    if (neurons_tail_active[i,j] == 1):
                        itail = j+1
                        ###neurons_tail_level[i,itail-1]=1 # nod needed. Forced cvheck.
                        taillevel=neurons_tail_level[i,itail-1]
                        

                # two possibilities if only one tail : erase tail points and keep 3 or do slipstick
                # only if more than 3 points
                endtailpoint=neurons_tail_ntot[i,itail-1]
                xend = neurons_x[i, itail-1, endtailpoint-1]
                yend = neurons_y[i, itail-1, endtailpoint-1]
            
                ixend = int(round(xend/deltax))
                if (ixend < 0):
                    ixend = 0
                if (ixend > nx-1):
                    ixend = nx-1
                iyend = int(round(yend/deltay))
                if (iyend < 0):
                    iyend = 0
                if (iyend > ny-1):
                    iyend = ny-1

                # a try erase
                p_tot=tail_delete_rate[taillevel-1] * tailrate_field[ixend, iyend] * (1+perturbations_field[ixend,iyend]*tail_delete_multfactor[taillevel-1])
               
                prb_tail = rng_vals[r_idx]
                r_idx += 1

                p_slipstick=slipstick_field[ix,iy]*p_tot
                #print(f'pslip {p_slipstick} p_tot {p_tot} ')
                if ( (prb_tail < p_tot) and (endtailpoint>3) and (sum(neurons_bulge_active[i,:]) == 0)):
                    erase_pts_tail1(i, itail, neurons_npoints_ini,\
                                    neurons_tail_ntot,\
                                    neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
                                    neurons_jmp_dyn_ini,neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini,\
                                    neurons_nk2, neurons_nk3, neurons_mass, neurons_mass_ini, \
                                    neurons_tail_active, neurons_tail_level, neurons_tail_hasparent, neurons_tail_nchildren,\
                                    neurons_size,neurons_size_ini )
                    #print('erase')
                    continue # next value of i (neuron)

                #there are two possibilities : erase or slipstick
                    #proba erase=1-proba_slipstick
                prb_tail = rng_vals[r_idx]
                r_idx += 1

                p_slipstick=slipstick_field[ix,iy]*p_tot
                if ( (prb_tail < p_slipstick) and (endtailpoint>3) and (sum(neurons_bulge_active[i,:]) == 0)):
      
                    slipstick_itail(rng_vals,itail, taillevel,neurons_x,neurons_y, i,  n_neurons,  pmax, pmin, box_temperature, box_timestep, accept_field,mcfield_sign, \
                                    box_x0, box_xf, box_y0, box_yf,box_nx, box_ny,box_PBC,box_slipstick_pause,neurons_pause,neurons_length_ini,\
                                    neurons_bulge_active,\
                                    neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
                                    neurons_mass,\
                                    neurons_jmp_dyn_ini,neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini, neurons_mass_ini,\
                                    neurons_velocity,neurons_nk2, neurons_nk3,\
                                    neurons_tail_ntot,potvalue,friction_field,box_slipstick_amplitude)
                    #print('slip')
                    neurons_NK_freq[i]+=1.0
                        
                    continue # next value of i (neuron)


            else:   # there is more tan one active tail. let's erase one
                    # rates depend on level !
                    # the tail we erase can be of any level but it CAN NOT HAVE A CHILD
                # loop until we find an active tail to erase, then we will check if we erase or not according to the level and the field value at the position of the soma
                havefind=False
                init=int(rng_vals[r_idx]*n_tails)+1
                r_idx +=1

                # We can only erase a last level (except if maxlevel =2)
                #identify max level among active tails
                maxlevel=0
                for j in range(n_tails):
                    if ((neurons_tail_active[i,j]==1) and (neurons_tail_level[i,j] > maxlevel)):
                        maxlevel=neurons_tail_level[i,j]

                if (maxlevel >=3 ):
                    for j in range(n_tails): 
                        tail_to_erase= j + init
                        if (tail_to_erase > n_tails):
                            tail_to_erase = tail_to_erase - n_tails
                        if (tail_to_erase > n_tails):
                            tail_to_erase = tail_to_erase - n_tails
                        nr_children=neurons_tail_nchildren[i, tail_to_erase-1]

                        if ((neurons_tail_level[i, tail_to_erase-1] == maxlevel) and (neurons_tail_active[ i,tail_to_erase-1] == 1) and (nr_children == 0)):
                            havefind=True
                            taillevel=neurons_tail_level[i,tail_to_erase-1]
                            break

                if (maxlevel ==2 ):
                     for j in range(n_tails): 
                        tail_to_erase= j + init
                        if (tail_to_erase > n_tails):
                            tail_to_erase = tail_to_erase - n_tails
                        if (tail_to_erase > n_tails):
                            tail_to_erase = tail_to_erase - n_tails

                        if  (neurons_tail_active[ i,tail_to_erase-1] == 1) :
                            havefind=True
                            taillevel=neurons_tail_level[i,tail_to_erase-1]
                            break
                if (not havefind):
                     print(f'problem with have_find i {i} maxlevel {maxlevel} ')
                      

                if (havefind):
                    endtailpoint=neurons_tail_ntot[i,tail_to_erase-1]
                    xend = neurons_x[i, tail_to_erase-1, endtailpoint-1]
                    yend = neurons_y[i, tail_to_erase-1, endtailpoint-1]
            
                    ixend = int(round(xend/deltax))
                    if (ixend < 0):
                        ixend = 0
                    if (ixend > nx-1):
                        ixend = nx-1
                    iyend = int(round(yend/deltay))
                    if (iyend < 0):
                        iyend = 0
                    if (iyend > ny-1):
                        iyend = ny-1


                    p_tot=tail_delete_rate[taillevel-1] * tailrate_field[ix, iy] * (1+perturbations_field[ixend,iyend]*tail_delete_multfactor[taillevel-1])
                    '''if (perturbations_field[ixend,iyend] > 1.0):
                        p_tot=0.0
                    '''
                    prb_tail = rng_vals[r_idx]
                    r_idx += 1
                    if (prb_tail < p_tot) :
                        #print(f' we erase full tail {tail_to_erase} level {taillevel} on i = {i}')
                        erasefull_tail(i, n_tails, tail_k2, tail_k3, neurons_tail_active, \
                                   tail_to_erase, neurons_tail_nchildren, neurons_tail_hasparent,  neurons_tail_level,neurons_tail_node,\
                                   neurons_x, neurons_y, neurons_nk2, neurons_k2, neurons_k2_id,\
                                   neurons_length, neurons_size, neurons_jmp_dyn, neurons_jmp_ang, neurons_jmp_rate, neurons_jmp_amp,\
                                   neurons_nk3, neurons_k3, neurons_k3_id, neurons_tail_ntot, neurons_velocity,\
                                   neurons_jmp_dyn_ini, neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini,neurons_mass_ini,neurons_mass)
                        continue # next value of i (neuron)

            # second : look for CREATE
            # this is tricky. Find an active tail (not on last level) and add one level
            nactive=sum(neurons_tail_active[i,:])
            
            # If there is only one active tail and only 3 points, we can extend to the max
            if (nactive ==1):
                for j in range(n_tails): 
                    if (neurons_tail_active[i,j] == 1):
                        itail = j+1
                        taillevel=neurons_tail_level[i,itail-1]


                endtailpoint=neurons_tail_ntot[i,itail-1]
               
                xend = neurons_x[i, itail-1, endtailpoint-1]
                yend = neurons_y[i, itail-1, endtailpoint-1]
            
                ixend = int(round(xend/deltax))
                if (ixend < 0):
                    ixend = 0
                if (ixend > nx-1):
                    ixend = nx-1
                iyend = int(round(yend/deltay))
                if (iyend < 0):
                    iyend = 0
                if (iyend > ny-1):
                    iyend = ny-1

                if (taillevel !=1 ):
                    print('we have a problem with taillevel not equal to 1')

                if (endtailpoint == neurons_npoints_ini[i]):
                    can_extend_to_max=True
                else:
                    can_extend_to_max=False

                if (potvalue[ixend,iyend] >  3.0):
                    can_extend_to_max=False
                



                if (can_extend_to_max):
                    r_idx += 1
                    prb_tail = rng_vals[r_idx]
                    p_tot=tail_create_rate[taillevel-1] * tailrate_field[ix, iy] * (1+perturbations_field[ix,iy]*tail_create_multfactor[taillevel-1])
                # tail creation
                # should use k2 k3 length l0 according to LEVEL

                    if  (prb_tail < p_tot):
                        #print(f' we extend a tail on i = {i}')

                        extend_onlytail(i, itail, taillevel, box_x0, box_xf, box_y0, box_yf,box_nx, box_ny,box_dx, box_dy,box_PBC,neurons_npoints_ini,\
                        neurons_x,neurons_y, neurons_tail_active, neurons_tail_ntot,neurons_bulge_active,neurons_tail_hasparent,\
                        neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
                        neurons_nk2, neurons_nk3, neurons_velocity, neurons_tail_node,neurons_mass, neurons_size,\
                        neurons_k2, neurons_k2_id, neurons_k3, neurons_k3_id, tail_k2, neurons_length,\
                        tail_l0, tail_k3, neurons_cosangle, tail_nmax, neurons_mass_ini,\
                        tail_mass_max, tail_l, tail_angle, rng_vals,potvalue, neurons_jmp_dyn_ini, neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini)


                        # Once extended to the maw we ca create branches # but not on this turn !!!
                        continue # next value of i (neuron)

            
            # For creating a branch we first need full extension of the mother branch
            for j in range(n_tails): 
                    if (neurons_tail_hasparent[i,j] == 0):
                        itail = j+1
                        endtailpoint=neurons_tail_ntot[i,itail-1]

            if (endtailpoint != neurons_npoints_ini[i]+tail_nmax[0]): #level1 tail + initial nr points
                continue # no full extension, no branching -> next neuron
                


            # tricky here: we have to find the level of the tail and check if we can create a branch according to the level (if we are already at max level, no branch)
            # pick a tail among the active ones
            # can create a level 3 only if level2 full etc...

            #  check level max and nr of tails at this level max. if not full, create one more at this level, if full, create one at the next level.
              # We can only erase a last level (except if maxlevel =2)
                #identify max level among active tails

            can_create_branch=False          
            n_activetails=sum(neurons_tail_active[i,:])

            if (n_activetails < n_tails):  #there is a possibility 

                maxlevel=0
                for j in range(n_tails):
                    if ((neurons_tail_active[i,j]==1) and (neurons_tail_level[i,j] > maxlevel)):
                        maxlevel=neurons_tail_level[i,j]

                nr_tails_at_maxlevel=0
                for j in range(n_tails):
                    if ((neurons_tail_active[i,j]==1) and (neurons_tail_level[i,j] == maxlevel)):
                         nr_tails_at_maxlevel+=1

                level_is_full=False
                if (maxlevel == 1):
                    level_is_full=True

                if (maxlevel > 2):
                    if (nr_tails_at_maxlevel == 2**(maxlevel-2) ):
                        level_is_full=True
                else:
                    if ((maxlevel == 2) and (nr_tails_at_maxlevel == 1 )):
                        level_is_full=True

                if (level_is_full):
                    taillevel=maxlevel+1
                    if (taillevel>n_levels):
                        print('problem with taillevel bigger than n_levels')
                else:
                    taillevel=maxlevel

                # now find an inactive tail with this level and an active parent
                havefind=False
                
                init=int(rng_vals[r_idx]*n_tails)+1
                r_idx+=1

                for j in range(n_tails): 
                    tail_to_branch = init + j
                    if (tail_to_branch > n_tails):
                        tail_to_branch = tail_to_branch - n_tails

                    j_parent=neurons_tail_hasparent[i,tail_to_branch-1]
                    
                    if ((neurons_tail_active[i,tail_to_branch-1] == 0) and (neurons_tail_active[i,j_parent-1] == 1) and (taillevel==neurons_tail_level[i,tail_to_branch-1])) :
                        itail = tail_to_branch  # this is indeed an inactive branch with an active parent. Use it.
                        havefind=True
                        parent_tail=j_parent
                        can_create_branch=True
                        break

                if (not havefind):
                     print(f'problem with have_find i {i} unactivated at level {taillevel} nr_active at maxlevel {maxlevel} is {nr_tails_at_maxlevel} ')

          

                if (can_create_branch):

                    endtailpoint=neurons_tail_ntot[i,parent_tail-1]
               
                    xend = neurons_x[i, parent_tail-1, endtailpoint-1]
                    yend = neurons_y[i, parent_tail-1, endtailpoint-1]
            
                    ixend = int(round(xend/deltax))
                    if (ixend < 0):
                        ixend = 0
                    if (ixend > nx-1):
                        ixend = nx-1
                    iyend = int(round(yend/deltay))
                    if (iyend < 0):
                        iyend = 0
                    if (iyend > ny-1):
                        iyend = ny-1
               
                    r_idx += 1
                    prb_tail = rng_vals[r_idx]
                    if (potvalue[ixend,iyend] > 1.0): ##high pot do not create more branches
                        p_tot=0.0
                    else:
                        p_tot=tail_create_rate[taillevel-1] * tailrate_field[ix, iy] * (1+perturbations_field[ixend,iyend]*tail_create_multfactor[taillevel-1])
                    
                    ##p_tot=tail_create_rate[taillevel-1] * tailrate_field[ix, iy] * (1+perturbations_field[ixend,iyend]*tail_create_multfactor[taillevel-1])

                
                    if (prb_tail < p_tot):
                        #print(f' we create a branch on i = {i} n points = {neurons_tail_ntot[i,tail_to_branch-1]} level = {taillevel}')
                    
                    # call branch creation this branch will be itail, with level taillevel
                        create_branch(i, itail, taillevel, n_levels, n_tails, box_x0, box_xf, box_y0, box_yf,box_nx, box_ny,box_dx, box_dy,box_PBC,neurons_npoints_ini,\
                            neurons_x,neurons_y,neurons_tail_ntot, \
                            neurons_tail_active, neurons_tail_level, neurons_tail_nchildren, neurons_tail_hasparent,\
                            neurons_tail_node, neurons_tail_angle, neurons_bulge_active,\
                            tail_delete_rate,neurons_jmp_dyn,neurons_jmp_rate, neurons_jmp_amp, neurons_jmp_ang,\
                            neurons_nk2, neurons_nk3, neurons_velocity, neurons_mass, neurons_size,\
                            neurons_k2, neurons_k2_id, neurons_k3, neurons_k3_id, tail_k2, neurons_length,\
                            tail_l0, tail_k3, neurons_cosangle, neurons_mass_ini,\
                            tail_mass_max, tail_l, tail_angle, tail_angle_start_min, tail_angle_start_max, rng_vals,potvalue,\
                            neurons_jmp_dyn_ini, neurons_jmp_rate_ini, neurons_jmp_amp_ini, neurons_jmp_ang_ini)
                        if (taillevel==2):
                            neurons_gc_split[i] += 1.0
                            neurons_L2[i]+=1
                            
                        if (taillevel==3):
                            neurons_L3[i]+=1    

                        continue # next value of i (neuron)
                  
        valbranches=sum(neurons_tail_active[i,:])
        if (valbranches==0):
                        valbranches = 1
        neurons_branches[i] = neurons_branches[i] + valbranches
        
    return
###################################################################################
