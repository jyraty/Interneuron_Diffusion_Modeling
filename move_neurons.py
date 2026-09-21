# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:45:42 2025

@author: jyrat
"""
from numba import njit, prange


@njit(parallel=True)
##@njit
def move_neurons(neurons_pause, n_neurons, n_tails, friction_field, box_x0, box_xf, box_y0, box_yf, box_dx, box_dy, box_nx, box_ny, box_PBC,
                 box_timestep, neurons_npoints_tot, neurons_mass, neurons_bulge_active, neurons_bulge_mass,
                 neurons_velocity, neurons_x, neurons_y, neurons_force, neurons_tail_node, neurons_tail_hasparent,neurons_tail_ntot, potfield, potmax):
    # Iupdate positions velocities for all points of all neurons

    dt = box_timestep

    xmin = box_x0
    xmax = box_xf
    ymin = box_y0
    ymax = box_yf
    deltax = box_dx
    deltay = box_dy
    nx = box_nx
    ny = box_ny

    f_force = float
    a_force = float

    for i in prange(n_neurons):
        if (neurons_pause[i]==0):
            
            #print(f' check5 {i} { neurons_x[i,0,2]} { neurons_x[i,0,1]} { neurons_x[i,0,0]}')
            #print(f' check7 vx {i} { neurons_velocity[i,0,2,0]} { neurons_velocity[i,0,1,0]} { neurons_velocity[i,0,0,0]}')
            ###################################################
            ###itail = 0  # this is the tail, always there, move all points
            #### NO HAS TO FIND THE TAIL WHICH HAS NO PARENT
            for k in range(n_tails):
                if (neurons_tail_hasparent[i, k] == 0):
                    itail = k
                    main_branch=k
                    break

            ###################################################
            for j in range(neurons_npoints_tot[i, itail]):
                mass = neurons_mass[i, itail, j]
    
                if (neurons_bulge_active[i, itail] == 1):
                    if (j < 3):
                        mass = neurons_bulge_mass[i, j]
    
                vx_prev = neurons_velocity[i, itail, j, 0]
                vy_prev = neurons_velocity[i, itail, j, 1]
                
    
                # friction depends on (old) position
                ix = int(neurons_x[i, itail, j]/deltax)
                if (ix < 0):
                    ix = 0
                if (ix > nx-1):
                    ix = nx-1
                iy = int(neurons_y[i, itail, j]/deltay)
                if (iy < 0):
                    iy = 0
                if (iy > ny-1):
                    iy = ny-1
                nu=float 
                nu = 1.0*friction_field[ix, iy]  ## JY DEBUG_CHECK
                # print(f' {ix} {iy} {nu}')
                a_force = neurons_force[i, itail, j, 0]/neurons_mass[i, itail, j]
                f_force = -1.0 * neurons_velocity[i, itail, j, 0] * nu / neurons_mass[i, itail, j]
                neurons_velocity[i, itail, j, 0] += (a_force+f_force)*dt
                dx = vx_prev*dt+(a_force+f_force)*dt*dt/2.0
                mymass=neurons_mass[i, itail, j]
                if (abs(dx) > 0.1):
                    ###print(f' too large dx for i itail' ,i, itail, dx, vx_prev, a_force, f_force, nu,mymass)
                    dx=0.0 ##0.5*dx/abs(dx)
                    neurons_velocity[i, itail, j, 0] = vx_prev
                
                a_force = neurons_force[i, itail, j, 1]/neurons_mass[i, itail, j]
                f_force = -1.0 * neurons_velocity[i, itail, j, 1] * nu / neurons_mass[i, itail, j]
                neurons_velocity[i, itail, j, 1] += (a_force+f_force)*dt
                dy = vy_prev*dt+(a_force+f_force)*dt*dt/2.0
                if (abs(dy) > 0.1):
                    ###print(f' too large dy for i itail' ,i, itail, dy, vy_prev, a_force, f_force, nu,mymass)
                    dy=0.0 ## 0.5*dy/abs(dy)
                    neurons_velocity[i, itail, j, 1]=vy_prev 
    
    
                x = neurons_x[i, itail, j] + dx
                y = neurons_y[i, itail, j] + dy
    
                if (box_PBC == False):
                    if ((x > xmin+deltax) and (x < xmax-deltax)):
                        neurons_x[i, itail, j] = x
                    else:
                        neurons_x[i, itail, j] = x - 2*dx  # rebond en x 
                        neurons_velocity[i, itail, j, 0] = - neurons_velocity[i, itail, j, 0]
                    # refuse if climbing on high potential
                    ''' if(potfield[ix, iy]>potmax/8):
                        neurons_x[i, itail, j] = x - 2*dx
                        neurons_velocity[i, itail, j, 0] = - neurons_velocity[i, itail, j, 0]
                        neurons_y[i, itail, j] = y - 2*dy
                        neurons_velocity[i, itail, j, 1] = - neurons_velocity[i, itail, j, 1]                   
                    '''


                else:  # PBC
                    if (x >= xmax):
                        x -= xmax
                    if (x <= xmin):
                        x += xmax
                    neurons_x[i, itail, j] = x
    
                if ((y > ymin+0*deltay) and (y < ymax-0*deltay)):
                    neurons_y[i, itail, j] = y
                else:
                    neurons_y[i, itail, j] = y - 2*dy  # rebond en y
                    neurons_velocity[i, itail, j, 1] = - neurons_velocity[i, itail, j, 1]
    
            ###################################################
            #itail = 1  # this is the 'branch' copy velocities and positions to node, then move othjer points
            ###################################################
            for itail in range(n_tails):
                id_parent=neurons_tail_hasparent[i, itail]

                if (id_parent != 0):  # this is not the main branch, move all points after node
                    
                    for j in range(neurons_tail_node[i, itail]):
                        neurons_x[i, itail, j] = neurons_x[i, id_parent-1, j] ##### use id_parent, not 0

                        neurons_y[i, itail, j] = neurons_y[i, id_parent-1, j]
                        neurons_velocity[i, itail, j, 0] = neurons_velocity[i, id_parent-1, j, 0]
                        neurons_velocity[i, itail, j, 1] = neurons_velocity[i, id_parent-1, j, 1]
    
                    # move next points
                    for j in range(neurons_tail_node[i, itail], neurons_tail_ntot[i, itail]):
                        mass = neurons_mass[i, itail, j]
                        if (neurons_bulge_active[i, itail] == 1):
                            if (j < 3):
                                mass = neurons_bulge_mass[i, j]
    
                        vx_prev = neurons_velocity[i, itail, j, 0]
                        vy_prev = neurons_velocity[i, itail, j, 1]
                        # vx_prev=0.0
                        # vy_prev=0.0
    
                        ix = int(neurons_x[i, itail, j]/deltax)
                        if (ix < 0):
                            ix = 0
                        if (ix > nx-1):
                            ix = nx-1
                        iy = int(neurons_y[i, itail, j]/deltay)
                        if (iy < 0):
                            iy = 0
                        if (iy > ny-1):
                            iy = ny-1
                        nu=float 
                        nu = 1.0*friction_field[ix, iy] 
                        # JY CHECK !neurons_mass[i,itail,j]
                        a_force = neurons_force[i, itail, j, 0]/mass
                        # JY CHECK  neurons_mass[i,itail,j]
                        f_force = -1.0*neurons_velocity[i, itail, j, 0]*nu / mass
                        # f_force=0
                        neurons_velocity[i, itail, j, 0] += (a_force+f_force)*dt
                        dx = vx_prev*dt+(a_force+f_force)*dt*dt/2.0
                        if (abs(dx) > 1.0):
                            print(f' too large fx for i itail' ,i, itail, dx, vx_prev, a_force, f_force, nu,mymass)
                            dx=1.0*dx/abs(dx)
                            neurons_velocity[i, itail, j, 0] = vx_prev
 
                        a_force = neurons_force[i, itail, j, 1]/mass
                        # JY CHECK  neurons_mass[i,itail,j]
                        f_force = -1.0*neurons_velocity[i, itail, j, 1]*nu / mass
                        # f_force=0
                        neurons_velocity[i, itail, j, 1] += (a_force+f_force)*dt
                        dy = vy_prev*dt+(a_force+f_force)*dt*dt/2.0
                        if (abs(dy) > 1.0):
                            print(f' too large fx for i itail' ,i, itail, dy, vy_prev, a_force, f_force, nu,mymass)
                            dy=1.0*dy/abs(dy)
                            neurons_velocity[i, itail, j, 1] = vy_prev
       
              
    
                        x = neurons_x[i, itail, j] + dx
                        y = neurons_y[i, itail, j] + dy
    
                        if (box_PBC == False):
                            if ((x > xmin+0*deltax) and (x < xmax-0*deltax)):
                                neurons_x[i, itail, j] = x
                            else:
                                neurons_x[i, itail, j] = x - 2*dx
                                neurons_velocity[i, itail, j, 0] = - \
                                    neurons_velocity[i, itail, j, 0]
                            ''' (potfield[ix, iy]>potmax/8):
                                neurons_x[i, itail, j] = x - 2*dx
                                neurons_velocity[i, itail, j, 0] = - neurons_velocity[i, itail, j, 0]
                                neurons_y[i, itail, j] = y - 2*dy
                                neurons_velocity[i, itail, j, 1] = - neurons_velocity[i, itail, j, 1]
                            '''
                        else:
                            if (x <= xmin):
                                x += xmax
                            if (x >= xmax):
                                x -= xmax
                            neurons_x[i, itail, j] = x
    
                        if ((y > ymin+0*deltay) and (y < ymax-0*deltay)):
                            neurons_y[i, itail, j] = y
                        else:
                            neurons_y[i, itail, j] = y - 2*dy
                            neurons_velocity[i, itail, j, 1] = - neurons_velocity[i, itail, j, 1]
                #print(f' check6 {i} { neurons_x[i,0,2]} { neurons_x[i,0,1]} { neurons_x[i,0,0]}')
    return
