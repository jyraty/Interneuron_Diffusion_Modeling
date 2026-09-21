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

from asyncio import run_coroutine_threadsafe
import numpy as np
import time
import secrets
import classes

from numba import njit, prange
from PIL import Image
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
import cv2
import math


global pi
pi = np.float64(np.pi)
global pi2
pi2=pi/2.0
global twopi
twopi = np.float64(2.0*np.pi)
@njit(parallel=True, fastmath=True)

def loop_poins(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau):
    for ipoint in prange(n_points):
        print(f' ipoint {ipoint}')
        xi=edges_list[ipoint][1]
        yi=edges_list[ipoint][0]  
        
        for i in range(int(xi-lim),int(xi+lim)):
            if ((i >= 0) and (i<= box_nx-1)):
                
                for j in range(int(yi-lim), int(yi+lim)):
                    if ((j >= 0) and (j <= box_ny-1)):
                        
                        A = (i-xi)
                        B = (j-yi)
                        d=A*A+B*B
                        val=height/(1.0+math.exp( (math.sqrt(float(d))-width) /decay))
                        if (val  > tableau[i,j]) : 
                            tableau[i,j] = val

@njit(parallel=True, fastmath=True)                           
def loop_neg_poins(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau):
    for ipoint in prange(n_points):
        print(f' ipoint {ipoint}')
        xi=edges_list[ipoint][1]
        yi=edges_list[ipoint][0]  
        
        for i in range(int(xi-lim),int(xi+lim)):
            if ((i >= 0) and (i<= box_nx-1)):
                
                for j in range(int(yi-lim), int(yi+lim)):
                    if ((j >= 0) and (j <= box_ny-1)):
                        
                        A = (i-xi)
                        B = (j-yi)
                        d=A*A+B*B
                      
                        val=height/(1.0+math.exp( (math.sqrt(float(d))-width) /width)**decay)

                        if (val  < tableau[i,j]) :
                            tableau[i,j] = val
                        
    
@njit(parallel=True, fastmath=True)
def loop_pot_poins(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau):
    for ipoint in prange(n_points):
        print(f' ipoint {ipoint}')
        xi=edges_list[ipoint][1]
        yi=edges_list[ipoint][0]
        for i in range(int(xi-lim),int(xi+lim)):
            if ((i >= 0) and (i<= box_nx-1)):
                
                for j in range(int(yi-lim), int(yi+lim)):
                    if ((j >= 0) and (j <= box_ny-1)):  
                        A = float(i-xi)
                        B = float(j-yi)
                        d=A*A+B*B
                        if (d > 0): 
                            val=height/( ((math.sqrt(d)+width)/width)**decay)
                        else:
                            val=height
                        if (val  > tableau[i,j]) : 
                            ##print(f' {j} {k} {val} {tableau[j,k]}')
                            tableau[i,j] = val
@njit(parallel=True, fastmath=True)
def loop_pot_poins_power(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau):
    for ipoint in prange(n_points):
        print(f' ipoint {ipoint}')
        xi=edges_list[ipoint][1]
        yi=edges_list[ipoint][0]
        for i in range(int(xi-lim),int(xi+lim)):
            if ((i >= 0) and (i<= box_nx-1)):

                for j in range(int(yi-lim), int(yi+lim)):
                    if ((j >= 0) and (j <= box_ny-1)):
                        A = float(i-xi)
                        B = float(j-yi)
                        d=A*A+B*B
                        if (math.sqrt(d)<width)and (d >0):
                            val=1.0*height*abs((math.sqrt(d)-1.0*width)/(1*width))**(1.0*decay)
                        else:
                            val=0.0
                        if (val  > tableau[i,j]) :
                            tableau[i,j] = val
                            
@njit(parallel=True, fastmath=True)
def loop_pot_poins_fermi(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau):
    for ipoint in prange(n_points):
        print(f' ipoint {ipoint}')
        xi=edges_list[ipoint][1]
        yi=edges_list[ipoint][0]
        for i in range(int(xi-lim),int(xi+lim)):
            if ((i >= 0) and (i<= box_nx-1)):

                for j in range(int(yi-lim), int(yi+lim)):
                    if ((j >= 0) and (j <= box_ny-1)):
                        A = float(i-xi)
                        B = float(j-yi)
                        d=A*A+B*B
                        if (d >0):
                            val=height/(1.0+math.exp( (math.sqrt(float(d))-width) /width)**decay)
                        else:
                            val=height
                        if (val  > tableau[i,j]) :
                            tableau[i,j] = val


@njit(parallel=True, fastmath=True)
def zero_affinty(affinity_field, tableau, box_nx, box_ny):
    for i in prange(box_nx):
        for j in range(box_ny):
            if (tableau[i,j] > 0.999):
                affinity_field[i,j]=0.0
    return
                    
def add_mask(field, mask_type, mask_file, mask_shape, mask_params, box_nx, box_ny):


    path_image=mask_file
    print(f' {path_image}')
    image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de lire l'image.")

    # --- 2. Dimensions d'origine ---
    hauteur, largeur = image.shape[:2]
    print(f"Dimensions d'origine : {largeur}x{hauteur}")
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    hauteur, largeur = rotated.shape[:2]
    print(f"apres rotation : {largeur}x{hauteur}")   

    cible_w, cible_h = box_nx, box_ny  
    tableau = np.zeros((cible_w, cible_h), dtype=float)
 
    contours=(255.0-rotated)/255.0

        


    # Redimensionner les contours pour tenir dans la zone cible
    contours_redim = cv2.resize(contours, (cible_h, cible_w), interpolation=cv2.INTER_NEAREST)
    tableau[:, :] = contours_redim
    
    if (mask_type=='Potential') :
        if (mask_shape=='Power'):
            height=mask_params[0]
            tableau[:, :] = contours_redim*height ## first everything rescaled 0-height
            amin = tableau.min()
            amax = tableau.max()
            width=mask_params[1]
            decay=mask_params[2]
            valmin=np.min(contours_redim)
            valmax=np.max(contours_redim)
            threshold=0.999
            sobel_x = cv2.Sobel(contours_redim, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(contours_redim, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = (magnitude > threshold).astype(np.uint8)
            ys, xs = np.where(edges == 1)
        
        # Convertit en liste de tuples (x, y)
            edges_list = list(zip(xs, ys))
            n_points=len(edges_list)

            ##print(f'n_points shape valmax valmin {n_points} {mask_shape} {valmax} {valmin}')

            arr_scaled=np.zeros((box_ny,box_nx), dtype=float )
            lim=2*width
            

            loop_pot_poins_power(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau)
                    

            amin = tableau.min()
            amax = tableau.max()
            ##print(f'amin amax {amin} {amax}')
            
            for j in range(box_nx):
                for k in range(box_ny):
                    arr_scaled[box_ny-k-1,j] = (tableau[j,k] - amin) / (amax - amin) * 255.0

            arr_uint8 = arr_scaled.astype(np.uint8)


    # Création de l'image PIL en mode "L" (grayscale)
            img = Image.fromarray(arr_uint8, mode="L")
           # Sauvegarde en PNG
            filename=str(f'{path_image}mask_{box_nx}_{box_ny}_{height}_{width}_{decay}.png')
            img.save(filename, format="PNG")
        if (mask_shape=='Fermi'):
            height=mask_params[0]
            tableau[:, :] = contours_redim*height ## first everything rescaled 0-height
            amin = tableau.min()
            amax = tableau.max()
            width=mask_params[1]
            decay=mask_params[2]
            valmin=np.min(contours_redim)
            valmax=np.max(contours_redim)
            threshold=0.999
            sobel_x = cv2.Sobel(contours_redim, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(contours_redim, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = (magnitude > threshold).astype(np.uint8)
            ys, xs = np.where(edges == 1)
        
        # Convertit en liste de tuples (x, y)
            edges_list = list(zip(xs, ys))
            n_points=len(edges_list)

            ##print(f'n_points shape valmax valmin {n_points} {mask_shape} {valmax} {valmin}')

            arr_scaled=np.zeros((box_ny,box_nx), dtype=float )
            lim=4*width
            

            loop_pot_poins_fermi(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau)
                    

            amin = tableau.min()
            amax = tableau.max()
            ##print(f'amin amax {amin} {amax}')
            
            for j in range(box_nx):
                for k in range(box_ny):
                    arr_scaled[box_ny-k-1,j] = (tableau[j,k] - amin) / (amax - amin) * 255.0

            arr_uint8 = arr_scaled.astype(np.uint8)


    # Création de l'image PIL en mode "L" (grayscale)
            img = Image.fromarray(arr_uint8, mode="L")
           # Sauvegarde en PNG
            filename=str(f'{path_image}mask_{box_nx}_{box_ny}_{height}_{width}_{decay}.png')
            img.save(filename, format="PNG")

        if (mask_shape=='Walls'):
            height=mask_params[0]
            tableau[:, :] = contours_redim*height ## first everything rescaled 0-height
            amin = tableau.min()
            amax = tableau.max()
            width=mask_params[1]
            decay=mask_params[2]
            valmin=np.min(contours_redim)
            valmax=np.max(contours_redim)
            threshold=0.999
            sobel_x = cv2.Sobel(contours_redim, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(contours_redim, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = (magnitude > threshold).astype(np.uint8)
            ys, xs = np.where(edges == 1)
    
        # Convertit en liste de tuples (x, y)
            edges_list = list(zip(xs, ys))
            n_points=len(edges_list)
            
    
            arr_scaled=np.zeros((box_ny,box_nx), dtype=float ) 
            lim=20*width

            
            loop_pot_poins(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau)
                
                                                     
            amin = tableau.min()
            amax = tableau.max()
            ##print(f'amin amax {amin} {amax}')

            for j in range(box_nx):
                for k in range(box_ny):
                    arr_scaled[box_ny-k-1,j] = (tableau[j,k] - amin) / (amax - amin) * 255.0
    
            arr_uint8 = arr_scaled.astype(np.uint8)
    
    
    # Création de l'image PIL en mode "L" (grayscale)
            img = Image.fromarray(arr_uint8, mode="L")

    
    
           # Sauvegarde en PNG
            filename=str(f'{path_image}mask_{box_nx}_{box_ny}_{height}_{width}_{decay}.png')
            img.save(filename, format="PNG")
 
        if (mask_shape=='Rescale'):
            ### simply add the rescaled mask
            height=mask_params[0]
            valmin=np.min(contours_redim)
            valmax=np.max(contours_redim)
            
            print(f'Potential valmax valmin {valmax} {valmin}')
            for j in range(box_nx):
                for k in range(box_ny):                            
                    tableau[j,k] = height*(valmax-contours_redim[j,k]) 
                    
            
    #######################################################
    if (mask_type=='Friction') :
        fmin =mask_params[0]
        fmax =mask_params[1]
        valmin=np.min(contours_redim)
        valmax=np.max(contours_redim) 
        tableau=fmin+ (contours_redim-valmin)/(valmax-valmin)*(fmax-fmin)
        
    if (mask_type=='Affinity') :

        if (mask_shape=='Rescale'):
            height=mask_params[0]
            valmin=np.min(contours_redim)
            valmax=np.max(contours_redim)

            
            print(f'Affinity valmax valmin {valmax} {valmin}')
            #for j in range(box_nx):
                #for k in range(box_ny):                            
                    #tableau[j,k] = height*(valmax-contours_redim[j,k])/valmax  ## 0-1
            
        if (mask_shape=='Fermi'):
            height=mask_params[0]
            width=mask_params[1]
            decay=mask_params[2]
            valmin=np.min(contours_redim)
            valmax=np.max(contours_redim)
            threshold=0.9
            sobel_x = cv2.Sobel(contours_redim, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(contours_redim, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = (magnitude > threshold).astype(np.uint8)
            ys, xs = np.where(edges == 1)

        # Convertit en liste de tuples (x, y)
            edges_list = list(zip(xs, ys))
            n_points=len(edges_list)
            tableau[:,:]=0.0 
            print(f'n_points shape valmax valmin {n_points} {mask_shape} {valmax} {valmin}')
   
            arr_scaled=np.zeros((box_ny,box_nx), dtype=float )

            lim=8*width
            if (height < 0):
                loop_neg_poins(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau)
            else:
                loop_poins(n_points, edges_list,lim, box_nx, box_ny, height,width,decay,tableau)
            

            
            amin = tableau.min()
            amax = tableau.max()
            print(f'amin amax {amin} {amax} image saved 0-255')
            for j in range(box_nx):
                for k in range(box_ny):
                    arr_scaled[box_ny-k-1,j] = (tableau[j,k] - amin) / (amax - amin) * 255.0

            arr_uint8 = arr_scaled.astype(np.uint8)
    

    # Création de l'image PIL en mode "L" (grayscale)
            img = Image.fromarray(arr_uint8, mode="L")
            # save figure in case


           # Sauvegarde en PNG
            filename=str(f'affinitymask_{box_nx}_{box_ny}_{height}_{width}_{decay}.png')
            img.save(filename, format="PNG")
            
        if (mask_shape=='Gradient'):
            val_y0=mask_params[0]
            val_ny=mask_params[1]
            for j in range(box_nx):
                for k in range(box_ny):                            
                    val=val_y0+k*(val_ny-val_y0)/box_ny
                    tableau[j,k]+= val
                    
        if (mask_shape=='Multiline_Cortex') or (mask_shape=='Multilineprod') or \
             (mask_shape=='Multiline_Brain') or  (mask_shape=='Multiline_Slice') :
            x = np.linspace(0, box_nx, box_nx)
            y = np.linspace(0, box_ny, box_ny)
            X, Y = np.meshgrid(x, y)
            if (mask_shape=='Multiline_Cortex'):
                filename="multiline_affinity_59pts_cortex_v9-800x700.txt"
            if (mask_shape=='Multiline_Brain'):
                filename="multiline_affinity_139pts_full_v12-800x1200.txt"
            if (mask_shape=='Multiline_Slice'):
                filename="multiline_affinity4points.txt"
            with open(filename, "r", encoding="utf-8") as f:
                lines=f.readlines()
                print(" reading multiline "+str(len(lines)))
                il = 0
                while il < len(lines):    
                    parts = lines[il].split()
                    if "#" in lines[il]:
                        il+=1
                        continue
        
                    #print(" il "+str(il)+" "+str(lines[il]))
                    parts = lines[il].split()
                    #the first line has the number of points
                    if (il == 0 ):
                        npoints=int(parts[0])
                        print(f' nr of points in multiline {npoints}')
                        ptx = np.zeros(npoints, dtype=int)
                        pty = np.zeros(npoints, dtype=int)
                        ptz = np.zeros(npoints, dtype=float)
                        points=np.zeros((npoints,2),dtype=float)
                        il+=1
                    else:
                        i = il-1
                        ptx[i]=int(parts[0])
                        pty[i]=int(parts[1])
                        ptz[i]=float(parts[2])
                        il+=1


            points[:,0]=ptx[:]
            points[:,1]=pty[:]
            tabswap=np.zeros((box_ny,box_nx),dtype=float)

            tabswap+=griddata(points,ptz, (X,Y), method='cubic')
            tableau=np.swapaxes(tabswap, 0, 1)  #*10


        
    
    for i in range(box_nx):
        for j in range(box_ny):
            if (mask_shape=='Multilineprod'):
                field[i,j]*=tableau[i,j]
            else:
                field[i,j]+=tableau[i,j]
    return


def mask_potentialOnAffinity(affinity_field, mask_file,box_nx, box_ny):
    path_image=mask_file
    print(f' {path_image}')
    image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de lire l'image.")

    # --- 2. Dimensions d'origine ---
    hauteur, largeur = image.shape[:2]
    print(f"Dimensions d'origine : {largeur}x{hauteur}")
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    hauteur, largeur = rotated.shape[:2]
    print(f"apres rotation : {largeur}x{hauteur}")   

    cible_w, cible_h = box_nx, box_ny  
    tableau = np.zeros((cible_w, cible_h), dtype=float)
    contours=(255.0-rotated)/255.0
    # Redimensionner les contours pour tenir dans la zone cible
    contours_redim = cv2.resize(contours, (cible_h, cible_w), interpolation=cv2.INTER_NEAREST)
    tableau[:, :] = contours_redim
    zero_affinty(affinity_field, tableau,box_nx, box_ny)

   
    return


def update_trajectories(n_neurons: int, tx, ty, itime, timestep, neurons_x, neurons_y,neurons_xprev, neurons_yprev,
                        neurons_velocity,box_xf, box_yf, neurons_tail_active, neurons_nbranches, neurons_ncontacts,gc_split, L2,L3,NK_freq):
    index = tx.index
    vecx = np.zeros(n_neurons, dtype=float)
    vecy = np.zeros(n_neurons, dtype=float)
    vx = np.zeros(n_neurons, dtype=float)
    vy = np.zeros(n_neurons, dtype=float)
    

    if (index >= tx.nlines):
        tx.add_time(1000)
        ty.add_time(1000)
        tx.nlines += 1000
        ty.nlines += 1000

    for i in range(n_neurons):
        vecx[i] = neurons_x[i, 0, 1]
        vecy[i] = neurons_y[i, 0, 1]
        

        tx.traj[index, 0] = itime
        tx.traj[index, 1] = itime*timestep
        tx.traj[index, 2+i] = neurons_x[i, 0, 1]  
        ty.traj[index, 0] = itime
        ty.traj[index, 1] = itime*timestep
        ty.traj[index, 2+i] = neurons_y[i, 0, 1]
        if (index == 0):
            vx[i] = 0.0 
            vy[i] = 0.0 
        else:
            vx[i] = (neurons_x[i, 0, 1]-neurons_xprev[i, 0, 1])/( tx.traj[index, 1]- tx.traj[index-1, 1])
            vy[i] = (neurons_y[i, 0, 1]-neurons_yprev[i, 0, 1])/( ty.traj[index, 1]- ty.traj[index-1, 1])

    if (index == 0):

        with open('traj_file_x.txt', 'w') as trajx_file:
            
            ligne = f"{itime:12d} {itime*timestep:3f}\t" + \
                "\t".join(f"{val:.3f}" for val in vecx) + "\n"
            trajx_file.write(ligne)
        with open('traj_file_y.txt', 'w') as trajy_file:
            ligne = f"{itime:12d} {itime*timestep:3f}\t" + \
                "\t".join(f"{val:.3f}" for val in vecy) + "\n"
            trajy_file.write(ligne)
        with open('traj_t_x_y_vx_vy.txt', 'w') as traj_file:
            for i in range(n_neurons):
                nb = neurons_nbranches[i]
                nc = neurons_ncontacts[i]
                ligne = f"{itime*timestep:3f} \t {vecx[i]:9.3f} \t { vecy[i]:9.3f}  \t {vx[i]:9.5f} \t {vy[i]:9.5f} \t {nb:9.5f} \t  {nc:9.5f}\t {NK_freq:9.5f}\t {gc_split:9.5f}\t {L2:9.5f}\t {L3:9.5f}\n   "
                traj_file.write(ligne)
    else:
       
        
        with open('traj_file_x.txt', 'a') as trajx_file:
            ligne = f"{itime:12d} {itime*timestep:3f}\t" + \
                "\t".join(f"{val:.3f}" for val in vecx) + "\n"
            trajx_file.write(ligne)
        with open('traj_file_y.txt', 'a') as trajy_file:
            ligne = f"{itime:12d} {itime*timestep:3f}\t" + \
                "\t".join(f"{val:.3f}" for val in vecy) + "\n"
            trajy_file.write(ligne)
        with open('traj_t_x_y_vx_vy.txt', 'a') as traj_file:
            for i in range(n_neurons):
                nb = neurons_nbranches[i]/( tx.traj[index, 0]- tx.traj[index-1, 0])
                nc = neurons_ncontacts[i] 

                ligne = f"{itime*timestep:3f} \t {vecx[i]:9.3f} \t { vecy[i]:9.3f}  \t {vx[i]:9.5f} \t {vy[i]:9.5f} \t {nb:9.5f} \t {nc:9.5f}\t {NK_freq:9.5f}\t {gc_split:9.5f}\t {L2:9.5f}\t {L3:9.5f}\n "
                traj_file.write(ligne)

    tx.index += 1
    ty.index += 1
    neurons_xprev[:,:,:]=neurons_x[:,:,:]
    neurons_yprev[:,:,:]=neurons_y[:,:,:]


def update_diffusion(n_neurons: int, diffx, diffy, difftot, itime, timestep,
                     neurons_x, neurons_y, neurons_x_ini, neurons_y_ini,box_xf, box_yf,\
                    neurons_times,ymin_time,ymax_time,neurons_tail_active, neurons_tail_ntot, neurons_tail_node,nmaxtails):
    index = diffx.index
    dx = np.zeros(n_neurons, dtype=float)
    dy = np.zeros(n_neurons, dtype=float)
    dtot = np.zeros(n_neurons, dtype=float)
    dtot = np.zeros(n_neurons, dtype=float)
    if (index >= diffx.nlines):
        diffx.add_time(1000)
        diffy.add_time(1000)
        diffx.nlines += 1000
        diffy.nlines += 1000
        difftot.add_time(1000)
        difftot.nlines += 1000
    for i in range(n_neurons):
        diffx.traj[index, 0] = itime
        diffx.traj[index, 1] = itime*timestep
        diffx.traj[index, 2+i] = (neurons_x[i, 0, 1] -
                                  neurons_x_ini[i, 0, 1])**2  
        
        if (index != 0):
            diffy.traj[index, 0]  += np.abs(neurons_y[i, 0, 1]-neurons_y_ini[i, 0, 1])/ diffx.traj[index, 1]
        else:
            diffy.traj[index, 0] = 0.0
        difftot.traj[index,0] += (neurons_y[i, 0, 1] -
        neurons_y_ini[i, 0, 1])**2/(1.0*n_neurons)
        difftot.traj[index,0] += (neurons_x[i, 0, 1]-neurons_x_ini[i, 0, 1])**2/(1.0*n_neurons)
        
        dx[i] = diffx.traj[index, 2+i]
        dy[i] = diffy.traj[index, 2+i]
        dtot[i] +=(neurons_y[i, 0, 1]-neurons_y_ini[i, 0, 1])**2+(neurons_x[i, 0, 1]-neurons_x_ini[i, 0, 1])**2
    diffy.traj[index, 0]= diffy.traj[index, 0]/(1.0*n_neurons)
     
    if (index == 0):
        with open('speed.txt', 'w') as speed_file:
            ligne = "neuron        tmin        tmax       speed \n"
            speed_file.write(ligne)

        with open('branches.txt', 'w') as branches_file:
            ligne = "index  neuron   ntails   itail   ltail \n"
            branches_file.write(ligne)
            
            
        for i in range(n_neurons):
          
            dx[i] = neurons_x[i, 0, 1]
            dy[i] = neurons_y[i, 0, 1]
            dtot[i] = neurons_x[i, 0, 1]
           
           

            ligne =  f"{itime:12d} {itime*timestep:3f} {diffy.traj[index,0]:11.5f} \n"
        with open('diffusion_x.txt', 'w') as diffx_file:
            ligne = f"{itime:12d} {itime*timestep:3f}\t" + \
                "\t".join(f"{val:09.3f}" for val in dx) + "\n"
            diffx_file.write(ligne)
        with open('diffusion_y.txt', 'w') as diffy_file:
            ligne = f"{itime:12d} {itime*timestep:3f} {diffy.traj[index,0]:11.5f} \n"
            diffy_file.write(ligne)
        with open('diffusion_tot.txt', 'w') as difftot_file:
            ligne = f"{itime:12d} {itime*timestep:3f} {difftot.traj[index,0]:11.5f} \n"
            difftot_file.write(ligne)
    else:
        
        for i in range(n_neurons):
            with open('branches.txt', 'a') as branches_file:
               ntails=sum(neurons_tail_active[i, :])
               for j in range(nmaxtails):
                   if (neurons_tail_active[i, j] == 1):
                        itail = j
                        ltail = neurons_tail_ntot[i, j]-neurons_tail_node[i, j]
                        if (j==0):
                            ltail=neurons_tail_ntot[i, j]
                        ligne = f"{itime:12d} {i:6d} {ntails:4d}  {itail:4d} {ltail:4d} \n"
                        branches_file.write(ligne)

            if ((neurons_y[i, 0, 1] >= ymin_time  ) and (neurons_times[i,0] <= 0.000001)):
                neurons_times[i,0] = itime*timestep
                
            if ((neurons_y[i, 0, 1] >= ymax_time  ) and (neurons_times[i,1] <= 0.000001)):
                neurons_times[i,1] = itime*timestep 
                avgspeed=(ymax_time-ymin_time)/(neurons_times[i,1]-neurons_times[i,0]) 
                neurons_times[i,2] = avgspeed
                
                with open('speed.txt', 'a') as speed_file:                  
                    ligne = f"{i:6d} {neurons_times[i,0]:11.3f} {neurons_times[i,1]:11.3f} {avgspeed:11.3f} \n"
                    speed_file.write(ligne)
                with open('speed_hist.txt', 'w') as speedhist_file:
                    data = neurons_times[:, 2]
    
                    # Définition des bornes
                    xmin = data.min()
                    xmax = data.max()
                    
                    # Calcul de l'histogramme
                    if (xmin > 0) and (xmax < 10000):
                        hist, bin_edges = np.histogram(data, bins=10, range=(xmin, xmax))
                    
                        for j in range(10):
                            ligne= f"{bin_edges[j]:9.3f} {hist[j]:9.3f} \n"
                            speedhist_file.write(ligne)


        
        with open('diffusion_x.txt', 'a') as diffx_file:
            ligne = f"{itime:12d} {itime*timestep:3f}\t" + \
                "\t".join(f"{val:09.3f}" for val in dx) + "\n"
            diffx_file.write(ligne)
        with open('diffusion_y.txt', 'a') as diffy_file:
            ligne = f"{itime:12d} {itime*timestep:3f} {diffy.traj[index,0]:11.5f} \n"
            diffy_file.write(ligne)
        with open('diffusion_tot.txt', 'a') as difftot_file:
            ligne = f"{itime:12d} {itime*timestep:3f} {difftot.traj[index,0]:11.3f} \n"
            difftot_file.write(ligne)
    diffx.index += 1
    diffy.index += 1
    difftot.index += 1
    return


def create_wall_field(wheight: float, wsigma: float, walls, box_dx, box_dy, box_nx, box_ny, box_PBC):
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    # print(" dx "+str(dx))
    nj = int(20*wsigma/dx)
    for j in range(nj):
        walls[:, j] += wheight/((j+1)*dx/wsigma)**4       
        walls[:, ny-j-1] += wheight/((j+1)*dx/wsigma)**4
    nj = int(20*wsigma/dy)
    if (box_PBC == False):
        for j in range(nj):
            walls[j, :] += wheight/((j+1)*dx/wsigma)**4      
            walls[nx-j-1,:] += wheight/((j+1)*dx/wsigma)**4
    return

def create_affinity_field(affinity, box_dx, box_dy, box_nx, box_ny, box_PBC):
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    affinity[:,:]=0.0
    return


@njit(parallel=True, fastmath=True)
def create_l_perturbation(n_line_pert: int, l_anisotropy: float,
                          l_length: float, l_height: float, l_width: float,
                          l_slength: float, l_sheight: float, l_swidth: float,
                          l_biased: bool,    nx: int, ny: int, pert_field, box_dx, box_dy,
                          box_xf, box_yf,box_PBC,line_pert_profile,line_pert_params, rng_vals):
    xmax=box_xf
    ymax=box_yf
    prob_field= np.zeros((nx, ny), dtype=float)
    prob_field[:,:]=1.0    
    if  (line_pert_profile == 'half'):
        vleft=line_pert_params[0]
        print(f' Setting perturbation field type : {line_pert_profile}')
        print(f' Parameters : {vleft} {line_pert_params[1]}')
        for i in range(0, nx):
            if (i < nx/2):
                prob_field[i,:]=line_pert_params[0]
            else:
                prob_field[i,:]=line_pert_params[1]

    if  (line_pert_profile == 'halftop'):
        vleft=line_pert_params[0]
        print(f' Setting perturbation field type : {line_pert_profile}')
        print(f' Parameters : {vleft} {line_pert_params[1]}')
        for i in range(0, ny):
            if (i < ny/2):
                prob_field[:,i]=line_pert_params[0]
            else:
                prob_field[:,i]=line_pert_params[1]
                
    if  (line_pert_profile == 'gradient_y'):
        v0=line_pert_params[0]
        vny=line_pert_params[1]
        print(f' Setting perturbation field type : {line_pert_profile}')
        print(f' Parameters : {vleft} {line_pert_params[1]}')
        for i in range(ny):
                prob_field[:,i]=v0+i*(vny-v0)/(ny-1)
  
    
    maxangle_rad = l_anisotropy/180.0*pi
    dx = box_dx
    dy = box_dy
    if l_biased:
        root = int(math.sqrt(n_line_pert))
        n_line_pert = root * root

    for i in range(n_line_pert):
        r_idx = i * 10  # consume random numbers deterministically
        angle = (rng_vals[r_idx] - 0.5) * 2 * maxangle_rad
        r_idx += 1
        if not l_biased:
            ACCEPT=False
            while not ACCEPT :
                 
                 x0 = rng_vals[r_idx] * nx * dx
                 r_idx += 1
                 y0 = rng_vals[r_idx] * ny * dy
                 r_idx += 1
                 ix=int(x0/dx)
                 iy=int(y0/dy)
                 prob_accept=prob_field[ix,iy]
                 
                 if (rng_vals[r_idx] < prob_accept):

                    ACCEPT=True           
                    l0 = l_length * (1 + (rng_vals[r_idx]-0.5) * 2 * l_slength)
                    r_idx += 1
                    h0 = l_height * (1 + (rng_vals[r_idx]-0.5) * 2 * l_sheight)
                    r_idx += 1
                    w0 = l_width * (1 + (rng_vals[r_idx]-0.5) * 2 * l_swidth)
                    xi= x0-l0/2.0*math.cos(angle)
                    yi= y0-l0/2.0*math.sin(angle)
                    xf = x0+l0/2.0*math.cos(angle)
                    yf = y0+l0/2.0*math.sin(angle)
                     
                    if (xi <=0 ):
                        xi = 0
                    if (xi > xmax ):
                        xi =xmax   
                    if (xf <=0 ):
                        xf = 0
                    if (xf > xmax ):
                        xf =xmax                          
                    if (yi <=0 ):
                        yi=0.0
                    if (yi > ymax ):
                        yi=ymax
                    if (yi <=0 ):
                        yi=0.0
                    if (yi > box_yf ):
                        yi=ymax
                         
                 r_idx+=1
        
        else:
       

            iline = math.sqrt(n_line_pert)
            ix0 = box_xf/iline*np.mod(i, iline)
            ix0 = box_xf / iline * (i % iline)  
            iy0 = box_yf / iline * (i // iline)  
            ixf = ix0 + box_xf / iline
            iyf = iy0 + box_yf / iline

            xi = ix0 + rng_vals[r_idx] * (ixf - ix0)
            r_idx += 1
            yi = iy0 + rng_vals[r_idx] * (iyf - iy0)
            r_idx += 1

            l0 = l_length * (1 + (rng_vals[r_idx]-0.5) * 2 * l_slength)
            r_idx += 1
            h0 = l_height * (1 + (rng_vals[r_idx]-0.5) * 2 * l_sheight)
            r_idx += 1
            w0 = l_width * (1 + (rng_vals[r_idx]-0.5) * 2 * l_swidth)
            xf = x0+l0*math.cos(angle)
            yf = y0+l0*math.sin(angle)

        # choose (x0,y0) as center
        for j in prange(nx):
            for k in range(ny):
                A = j*dx-xi
                B = k*dy-yi
                if box_PBC:
                    dx2 = A- xmax * round(A / xmax)
                    A=dx2
                NA = math.sqrt(A*A+B*B)
                X = xf-xi
                Y = yf-yi
                if box_PBC:
                    dx2 = X- xmax * round(X / xmax)
                    X=dx2
                NX = math.sqrt(X*X+Y*Y)
                if NA > 0 and NX > 0:
                    cos1 = (A * X + B * Y) / (NA * NX)
                else:
                    cos1 = -1

                if (cos1 < 0):
                    # use sphere
                    t = h0*np.exp(-(A*A+B*B)/(2*w0*w0))
                    pert_field[j, k] += t
                else:
                    A = j*dx-xf
                    B = k*dy-yf
                    if box_PBC:
                        dx2 = A- xmax * round(A / xmax)
                        A=dx2
                    NA = math.sqrt(A*A+B*B)
                    X = xi-xf
                    Y = yi-yf
                    if box_PBC:
                        dx2 = X- xmax * round(X / xmax)
                        X=dx2
                    NX = math.sqrt(X*X+Y*Y)
                    if NA > 0 and NX > 0:
                        cos1 = (A * X + B * Y) / (NA * NX)
                    else:
                        cos1 = -1
                    if (cos1 < 0):
                        # use sphere
                        t = h0*math.exp(-(A*A+B*B)/(2*w0**2))
                        pert_field[j, k] += t
                    else:
                        A = j*dx-xi
                        B = k*dy-yi
                        if box_PBC:
                            dx2 = A- xmax * round(A / xmax)
                            A=dx2
                        NA = math.sqrt(A*A+B*B)
                        X = yi-yf
                        Y = xf-xi
                        if box_PBC:
                            dx2 = X- xmax * round(X / xmax)
                            X=dx2
                        NX = math.sqrt(X*X+Y*Y)
                        if NX > 0:
                            X /= NX
                            Y /= NX
                        l = A*X+B*Y
                        t = h0*math.exp(-(l*l)/(2*w0**2))
                        pert_field[j, k] += t

    return


def glials_field(n_glials: int, gl_length: float, gl_height: float, gl_width: float, glials, box_dx, box_dy, box_nx, box_ny):

    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny

    glials[:, :] = 0.0
    for i in range(n_glials):

        pos = nx*(i+1)/(n_glials+1)
        for j in range(nx):
            for k in range(int(gl_length)):
                iy = ny-k
                xx = abs(j-pos)
                t = gl_height*np.exp(-(xx*xx)/(2*gl_width**2))
                glials[j, iy-1] += (t*(gl_length-k))/(1.0*gl_length)

    return


def create_friction_field(fr_type, fr_params, friction, box_dx, box_dy, box_nx, box_ny, box_friction_profile,
                          box_friction_min, box_friction_max):
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny

            
    if (fr_type == 'linear'):
        box_friction_profile = 'linear'
        fmax = fr_params[0]
        box_friction_max = fmax
        fmin = fr_params[1]
        box_friction_min = fmin
        for j in range(0, ny):
            friction[:, j] = fmax+j*(fmin-fmax)/(1.*(ny-1))

                    

    if (fr_type == 'channel_grad'):
        box_friction_profile = 'channel_grad'
        c_number = fr_params[0]
        c_height = fr_params[1]
        v0 = fr_params[2]
        vny = fr_params[3]
        friction[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                friction[j, k] = c_height * \
                    np.cos((1.0*j)/(1.0*nx)*c_number*pi)**2.0
                friction[j, k] += v0+k*(vny-v0)/(1.*(ny-1))
                
                
    if (fr_type == 'trenches_grad'):
        box_friction_profile = 'trenches_grad'
        c_number = int(fr_params[0])
        c_width = float(fr_params[1])
        c_sigma = float(fr_params[2])
        h_0 = fr_params[3]
        h_ny = fr_params[4]
        l_0 = fr_params[5]
        l_ny = fr_params[6]
        friction[:, :] = 0.0
        for i in range(c_number):
            pos = (2*i+1)*c_width            
            left = pos-c_width/2.0
            right = pos+c_width/2.0
            for j in range(0, nx):
                for k in range(0, ny):
                    l=l_0+k*(l_ny-l_0)/(1.*(ny-1))
                    h=h_0+k*(h_ny-h_0)/(1.*(ny-1))
                    if (j*dx <= pos):
                        fermi=1.0/(1.0+np.exp( -(j*dx - left)/c_sigma ))                        
                    else:
                        fermi=1.0/(1.0+np.exp( (j*dx - right)/c_sigma )) 
                        
                    friction[j, k] = friction[j,k]+ (h-l)*fermi+l
    return

def create_slipstick_field(slipstick_type, slipstick_params, slipstick, box_dx, box_dy, box_nx, box_ny, box_slipstick_profile):
                          
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny

            
    if (slipstick_type == 'linear'):
        box_slipstick_profile = 'linear'
        fmax = slipstick_params[0]
        box_slipstick_max = fmax
        fmin = slipstick_params[1]
        box_slipstick_min = fmin

        for j in range(0, ny):
            slipstick[:, j] = fmax+j*(fmin-fmax)/(1.*(ny-1))

                    

    if (slipstick_type == 'channel_grad'):
        box_slipstick_profile = 'channel_grad'
        c_number = slipstick_params[0]
        c_height = slipstick_params[1]
        v0 = slipstick_params[2]
        vny = slipstick_params[3]

        slipstick[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                slipstick[j, k] = c_height * \
                    np.cos((1.0*j)/(1.0*nx)*c_number*pi)**2.0
                slipstick[j, k] += v0+k*(vny-v0)/(1.*(ny-1))
                
                
    if (slipstick_type == 'trenches_grad'):
        box_slipstick_profile = 'trenches_grad'
        c_number = int(slipstick_params[0])
        c_width = float(slipstick_params[1])
        c_sigma = float(slipstick_params[2])
        h_0 = slipstick_params[3]
        h_ny = slipstick_params[4]
        l_0 = slipstick_params[5]
        l_ny = slipstick_params[6]

        slipstick[:, :] = 0.0
        for i in range(c_number):
            pos = (2*i+1)*c_width/2.0           
            left = pos-c_width/2.0
            right = pos+c_width/2.0
            for j in range(0, nx):
                for k in range(0, ny):
                    l=l_0+k*(l_ny-l_0)/(1.*(ny-1))
                    h=h_0+k*(h_ny-h_0)/(1.*(ny-1))
                    if (j*dx <= pos):
                        fermi=1.0/(1.0+np.exp( -(j*dx - left)/c_sigma ))                        
                    else:
                        fermi=1.0/(1.0+np.exp( (j*dx - right)/c_sigma ))                
                    slipstick[j, k] = slipstick[j,k]+ (h-l)*fermi+l
    return

def create_tailrate_field(tailrate_type, tailrate_params, tailrate, box_dx, box_dy, box_nx, box_ny, box_tailrate_profile,
                          box_tailrate_min, box_tailrate_max):
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    if (tailrate_type == 'linear'):
        box_tailrate_profile = 'linear'
        fmax = tailrate_params[0]
        box_tailrate_max = fmax
        fmin = tailrate_params[1]
        box_tailrate_min = fmin
        for j in range(0, ny):
            tailrate[:, j] = fmax+j*(fmin-fmax)/(1.*(ny-1))
    
    if (tailrate_type == 'channel_grad'):
        box_tailrate_profile = 'channel_grad'
        c_number = tailrate_params[0]
        c_height = tailrate_params[1]
        v0 = tailrate_params[2]
        vny = tailrate_params[3]
        tailrate[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                tailrate[j, k] = c_height * \
                    np.cos((1.0*j)/(1.0*nx)*c_number*pi)**2.0
                tailrate[j, k] += v0+k*(vny-v0)/(1.*(ny-1))
    if (tailrate_type == 'half_vertical'):
        box_tailrate_profile = 'half_vertical'
        yleft = tailrate_params[0]
        yright = tailrate_params[1]
        sigma = tailrate_params[2]

        print(f' we are here {box_tailrate_profile} {tailrate_params[0]} {tailrate_params[1]} { tailrate_params[2]} ')
        tailrate[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                fermi=yleft/(1.0+np.exp( -(j - nx/2))/sigma )+yright/(1.0+np.exp( -(nx/2-j))/sigma ) 
                tailrate[j, k] = fermi
                
                
    if (tailrate_type == 'trenches_grad'):
        box_tailrate_profile = 'trenches_grad'
        c_number = int(tailrate_params[0])
        c_width = float(tailrate_params[1])
        c_sigma = float(tailrate_params[2])
        h_0 = tailrate_params[3]
        h_ny = tailrate_params[4]
        l_0 = tailrate_params[5]
        l_ny = tailrate_params[6]
        tailrate[:, :] = 0.0
        for i in range(c_number):
            pos = (2*i+1)*c_width            
            left = pos-c_width/2.0
            right = pos+c_width/2.0
            for j in range(0, nx):
                for k in range(0, ny):
                    l=l_0+k*(l_ny-l_0)/(1.*(ny-1))
                    h=h_0+k*(h_ny-h_0)/(1.*(ny-1))
                    if (j*dx <= pos):
                        fermi=1.0/(1.0+np.exp( -(j*dx - left))/c_sigma )                        
                    else:
                        fermi=1.0/(1.0+np.exp( (j*dx - right))/c_sigma )                
                    tailrate[j, k] = tailrate[j,k]+ (h-l)*fermi+l
    return


def create_branchrate_field(branchrate_type, branchrate_params, branchrate, box_dx, box_dy, box_nx, box_ny, box_branchrate_profile,
                            box_branchrate_min, box_branchrate_max):
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    if (branchrate_type == 'linear'):
        box_branchrate_profile = 'linear'
        fmax = branchrate_params[0]
        box_branchrate_max = fmax
        fmin = branchrate_params[1]
        box_branchrate_min = fmin
        for j in range(0, ny):
            branchrate[:, j] = fmax+j*(fmin-fmax)/(1.*(ny-1))
    if (branchrate_type == 'channel_grad'):
        box_branchrate_profile = 'channel_grad'
        c_number = branchrate_params[0]
        c_height = branchrate_params[1]
        v0 = branchrate_params[2]
        vny = branchrate_params[3]
        branchrate[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                branchrate[j, k] = c_height * \
                    np.cos((1.0*j)/(1.0*nx)*c_number*pi)**2.0
                branchrate[j, k] += v0+k*(vny-v0)/(1.*(ny-1))
    if (branchrate_type == 'half_vertical'):
        box_branchrate_profile = 'half_vertical'
        yleft = branchrate_params[0]
        yright = branchrate_params[1]
        sigma = branchrate_params[2]

        branchrate[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                fermi=yleft/(1.0+np.exp( -(j - nx/2))/sigma )+yright/(1.0+np.exp( -(nx/2-j))/sigma ) 
                branchrate[j, k] = fermi

    if (branchrate_type == 'trenches_grad'):
        box_branchrate_profile = 'trenches_grad'
        c_number = int(branchrate_params[0])
        c_width = float(branchrate_params[1])
        c_sigma = float(branchrate_params[2])
        h_0 = branchrate_params[3]
        h_ny = branchrate_params[4]
        l_0 = branchrate_params[5]
        l_ny = branchrate_params[6]
        branchrate[:, :] = 0.0
        for i in range(c_number):
            pos = (2*i+1)*c_width/2.0
            left = pos-c_width/2.0
            right = pos+c_width/2.0
            for j in range(0, nx):
                for k in range(0, ny):
                    l=l_0+k*(l_ny-l_0)/(1.*(ny-1))
                    h=h_0+k*(h_ny-h_0)/(1.*(ny-1))
                    if (j*dx <= pos):
                        fermi=1.0/(1.0+np.exp( -(j*dx - left))/c_sigma )                        
                    else:
                        fermi=1.0/(1.0+np.exp( (j*dx - right))/c_sigma )                
                    branchrate[j, k] = branchrate[j,k]+ (h-l)*fermi+l
    return


def create_jumpamp_field(jumpamp_type, jumpamp_params, jumpamp, box_dx, box_dy, box_nx, box_ny, box_jumpamp_profile,
                         box_jumpamp_min, box_jumpamp_max):
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    if (jumpamp_type == 'linear'):
        box_jumpamp_profile = 'linear'
        fmax = jumpamp_params[0]
        box_jumpamp_max = fmax
        fmin = jumpamp_params[1]
        box_jumpamp_min = fmin
        for j in range(0, ny):
            jumpamp[:, j] = fmax+j*(fmin-fmax)/(1.*(ny-1))
    if (jumpamp_type == 'channel_grad'):
        box_jumpamp_profile = 'channel_grad'
        c_number = jumpamp_params[0]
        c_height = jumpamp_params[1]
        v0 = jumpamp_params[2]
        vny = jumpamp_params[3]
        jumpamp[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                jumpamp[j, k] = c_height * \
                    np.cos((1.0*j)/(1.0*nx)*c_number*pi)**2.0
                jumpamp[j, k] += v0+k*(vny-v0)/(1.*(ny-1))                                
    if (jumpamp_type == 'trenches_grad'):
        box_jumpamp_profile = 'trenches_grad'
        c_number = int(jumpamp_params[0])
        c_width = float(jumpamp_params[1])
        c_sigma = float(jumpamp_params[2])
        h_0 = jumpamp_params[3]
        h_ny = jumpamp_params[4]
        l_0 = jumpamp_params[5]
        l_ny = jumpamp_params[6]
        jumpamp[:, :] = 0.0
        for i in range(c_number):
            pos = (2*i+1)*c_width            
            left = pos-c_width/2.0
            right = pos+c_width/2.0
            for j in range(0, nx):
                for k in range(0, ny):
                    l=l_0+k*(l_ny-l_0)/(1.*(ny-1))
                    h=h_0+k*(h_ny-h_0)/(1.*(ny-1))
                    if (j*dx <= pos):
                        fermi=1.0/(1.0+np.exp( -(j*dx - left))/c_sigma )                        
                    else:
                        fermi=1.0/(1.0+np.exp( (j*dx - right))/c_sigma )                
                    jumpamp[j, k] = jumpamp[j,k]+ (h-l)*fermi+l
    return


def create_jumprate_field(jumprate_type, jumprate_params, jumprate, box_dx, box_dy, box_nx, box_ny, box_jumprate_profile):
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    if (jumprate_type == 'linear'):
        box_jumprate_profile = 'linear'
        fmax = jumprate_params[0]
        fmin = jumprate_params[1]
        for j in range(0, ny):
            jumprate[:, j] = fmax+j*(fmin-fmax)/(1.*(ny-1))
    if (jumprate_type == 'channel_grad'):
        box_jumprate_profile = 'channel_grad'
        c_number = jumprate_params[0]
        c_height = jumprate_params[1]
        v0 = jumprate_params[2]
        vny = jumprate_params[3]
        jumprate[:, :] = 0.0
        for j in range(0, nx):
            for k in range(0, ny):
                jumprate[j, k] = c_height * \
                    np.cos((1.0*j)/(1.0*nx)*c_number*pi)**2.0
                jumprate[j, k] += v0+k*(vny-v0)/(1.*(ny-1))                                
    if (jumprate_type == 'trenches_grad'):
        box_jumprate_profile = 'trenches_grad'
        c_number = int(jumprate_params[0])
        c_width = float(jumprate_params[1])
        c_sigma = float(jumprate_params[2])
        h_0 = jumprate_params[3]
        h_ny = jumprate_params[4]
        l_0 = jumprate_params[5]
        l_ny = jumprate_params[6]
        jumprate[:, :] = 0.0
        for i in range(c_number):
            pos = (2*i+1)*c_width            
            left = pos-c_width/2.0
            right = pos+c_width/2.0
            for j in range(0, nx):
                for k in range(0, ny):
                    l=l_0+k*(l_ny-l_0)/(1.*(ny-1))
                    h=h_0+k*(h_ny-h_0)/(1.*(ny-1))
                    if (j*dx <= pos):
                        fermi=1.0/(1.0+np.exp( -(j*dx - left))/c_sigma )                        
                    else:
                        fermi=1.0/(1.0+np.exp( (j*dx - right))/c_sigma )                
                    jumprate[j, k] = jumprate[j,k]+ (h-l)*fermi+l
    return

#
###############################################################################
# create all neuron data is set from point 1 to npoints_ini
###############################################################################
#
#
@njit(parallel=True, fastmath=True)
def compute_neighbours(n_neurons, neurons_x, neurons_y, box_xf, box_yf, box_PBC, neighbours_list_rcut, neurons_nneighbours, neurons_neighbour_id):

     for i in prange(n_neurons):
        nneighbours=0
        for j in range(n_neurons):
            if (i!=j):

                k=1
                l=1
                dx = neurons_x[i, 0, k]-neurons_x[j, 0, l]
                dy = neurons_y[i, 0, k]-neurons_y[j, 0, l]
                if box_PBC:
                    dx -= box_xf * round(dx / box_xf)
                    dy -= box_yf * round(dy / box_yf)
                if ((dx*dx+dy*dy) < neighbours_list_rcut*neighbours_list_rcut):
                    neurons_neighbour_id[i, nneighbours] = j
                    nneighbours+=1
                  

        neurons_nneighbours[i]=nneighbours
     return



def create_neurons(n_neurons: int,nshape:str, xmin: float, xmax: float, ymin: float, ymax: float, n_levels:int, n_tails:int,  npoints: int, anisotropy: float,
                   nk2: int, nk3: int,
                   interaction_att, fraction_att, r_width_att, r_cut_att,\
                   interaction_rep, fraction_rep, r_width_rep, r_cut_rep, box_yf:float, \
                   length, angle, mass, size,  jmp_dyn, jmp_rate, jmp_amp, jmp_ang, k2, k2_id, k3, k3_id,
                   neurons_maxpoints, neurons_x, neurons_y, neurons_x_ini, neurons_y_ini, neurons_npoints, neurons_npoints_ini,
                   neurons_tail_active, neurons_tail_ntot, neurons_tail_level,  neurons_tail_number, neurons_tail_haschild, neurons_tail_hasparent,\
                   neurons_repulsion, neurons_r_width, neurons_r_cut,\
                   neurons_nk2, neurons_nk2_ini, neurons_nk3, neurons_nk3_ini, neurons_mass, neurons_mass_ini, neurons_size, neurons_size_ini,
                   neurons_jmp_dyn, neurons_jmp_dyn_ini, neurons_jmp_rate, neurons_jmp_rate_ini, neurons_jmp_amp, neurons_jmp_amp_ini,
                   neurons_jmp_ang, neurons_jmp_ang_ini, neurons_k2_id, neurons_k2_id_ini, neurons_k2, neurons_k2_ini,
                   neurons_length, neurons_length_ini, neurons_k3_id, neurons_k3_id_ini, neurons_k3, neurons_k3_ini,
                   neurons_cosangle, neurons_cosangle_ini, neurons_velocity,neurons_is_repulsive):


    
    
    rng1 = np.random.default_rng()
    rng_vals=rng1.random(5000000)
    r_idx=1
    maxangle_rad = anisotropy/180.0*pi
    available = np.zeros((npoints, npoints), dtype=int)
    available[:, :] = 1

    if (nshape=='circle'):
        xc=xmin
        yc=xmax
        radius=ymin
        
    for i in range(n_neurons):
  
        maxpoints = neurons_maxpoints[i]


        neurons_is_repulsive[i]=0
        
        proba=rng_vals[r_idx]
        r_idx+=1
        if (proba< fraction_rep):
            neurons_is_repulsive[i]=1
            neurons_repulsion[i] = interaction_rep
            neurons_r_width[i] = r_width_rep
            neurons_r_cut[i] = r_cut_rep
        else:
            if (proba< fraction_rep+fraction_att):
                neurons_is_repulsive[i]=-1
                neurons_repulsion[i] = interaction_att
                neurons_r_width[i] = r_width_att
                neurons_r_cut[i] = r_cut_att
            else:
                neurons_is_repulsive[i]=0
                neurons_repulsion[i] = 0.0
                neurons_r_width[i] = 1.0
                neurons_r_cut[i] = 1.0


        is_OK = False
        while (not is_OK):
            is_OK = True
            theta = 2*(rng_vals[r_idx]-0.5)*maxangle_rad
            r_idx+=1
            costheta = np.cos(theta+pi2)
            sintheta = np.sin(theta+pi2)
            
            x = np.zeros(maxpoints, dtype=float)
            y = np.zeros(maxpoints, dtype=float)
            if (nshape=='rectangle'):
                x[0] = xmin+rng_vals[r_idx]*(xmax-xmin)
                r_idx+=1
                y[0] = ymin+rng_vals[r_idx]*(ymax-ymin)
                r_idx+=1
            else:
                if (nshape =='circle'):
                    theta=rng_vals[r_idx]*2*pi
                    r_idx+=1
                    x[0] = xc+(rng_vals[r_idx]*radius)*np.cos(theta)
                    y[0] = yc+(rng_vals[r_idx]*radius)*np.sin(theta)
                    r_idx+=1
            
            
            
            for k in range(1, npoints, 1):
                x[k] = x[k-1]+length[k-1, k]*costheta
                y[k] = y[k-1]+length[k-1, k]*sintheta

                for j in range(i):
                    for l in range(npoints):
                        dx = x[k]-neurons_x[j, 0, l]
                        dy = y[k]-neurons_y[j, 0, l]
                        if ((dx*dx+dy*dy) < 0.1):
                            is_OK = False

            if (is_OK): 
                for channel in range(n_tails):
                    neurons_x[i, channel, :] = x[:]
                    neurons_y[i, channel, :] = y[:]
                    neurons_x_ini[i, channel, :] = x[:]
                    neurons_y_ini[i, channel, :] = y[:]

        neurons_npoints[i] = npoints
        neurons_npoints_ini[i] = npoints
        neurons_tail_active[i, :] = 0
        neurons_tail_active[i, 0] = 1
        neurons_tail_ntot[i, :] = npoints
        neurons_tail_level[i, 0] = 1  

        for i_level in range(n_levels):
            ix=0
            if (i_level ==0):
                neurons_tail_level[i,ix] = 1
                neurons_tail_hasparent[i,ix] = 0
            if (i_level==1):
                neurons_tail_level[i,ix+1] = 2
                neurons_tail_hasparent[i,ix+1] = 1

            if (i_level == 2):                      
                neurons_tail_level[i,ix+2] = 3
                neurons_tail_level[i,ix+3] = 3
                neurons_tail_hasparent[i,ix+2] = 1
                neurons_tail_hasparent[i,ix+3] = 2

            if (i_level == 3):
                neurons_tail_level[i,ix+4] = 4
                neurons_tail_level[i,ix+5] = 4
                neurons_tail_level[i,ix+6] = 4
                neurons_tail_level[i,ix+7] = 4
                neurons_tail_hasparent[i,ix+4] = 1
                neurons_tail_hasparent[i,ix+5] = 2
                neurons_tail_hasparent[i,ix+6] = 3
                neurons_tail_hasparent[i,ix+7] = 4

            if (i_level == 4):
                neurons_tail_level[i,ix+8] = 5
                neurons_tail_level[i,ix+9] = 5
                neurons_tail_level[i,ix+10] = 5
                neurons_tail_level[i,ix+11] = 5
                neurons_tail_level[i,ix+12] = 5
                neurons_tail_level[i,ix+13] = 5
                neurons_tail_level[i,ix+14] = 5
                neurons_tail_level[i,ix+15] = 5
                neurons_tail_hasparent[i,ix+8] = 1
                neurons_tail_hasparent[i,ix+9] = 2
                neurons_tail_hasparent[i,ix+10] = 3
                neurons_tail_hasparent[i,ix+11] = 4
                neurons_tail_hasparent[i,ix+12] = 5
                neurons_tail_hasparent[i,ix+13] = 6
                neurons_tail_hasparent[i,ix+14] = 7
                neurons_tail_hasparent[i,ix+15] = 8



        for itail in range(n_tails-1):
            neurons_tail_haschild[i,itail]=0

            neurons_nk2[i, itail] = nk2
            neurons_nk2_ini[i, itail] = nk2
            neurons_nk3[i, itail] = nk3
            neurons_nk3_ini[i, itail] = nk3
            neurons_mass[i, itail, :npoints] = mass[:npoints]
            neurons_mass_ini[i, itail, :npoints] = mass[:npoints]
            neurons_size[i, itail, :npoints] = size[:npoints]
            neurons_size_ini[i, itail, :npoints] = size[:npoints]
            neurons_jmp_dyn[i, itail, :npoints] = jmp_dyn[:npoints]
            neurons_jmp_dyn_ini[i, itail, :npoints] = jmp_dyn[:npoints]
            neurons_jmp_rate[i, itail, :npoints] = jmp_rate[:npoints]
            neurons_jmp_rate_ini[i, itail, :npoints] = jmp_rate[:npoints]
            neurons_jmp_amp[i, itail, :npoints] = jmp_amp[:npoints]
            neurons_jmp_amp_ini[i, itail, :npoints] = jmp_amp[:npoints]
            neurons_jmp_ang[i, itail, :npoints] = jmp_ang[:npoints]
            neurons_jmp_ang_ini[i, itail, :npoints] = jmp_ang[:npoints]
            neurons_k2_id[i, itail, :nk2, :] = k2_id[:nk2, :]
            neurons_k2_id_ini[i, itail, :nk2, :] = k2_id[:nk2, :]
            neurons_k2[i, itail, :nk2] = k2[:nk2]
            neurons_k2_ini[i, itail, :nk2] = k2[:nk2]
            neurons_length[i, itail, :npoints,
                           :npoints] = length[:npoints, :npoints]
            neurons_length_ini[i, itail, :npoints,
                               :npoints] = length[:npoints, :npoints]
            neurons_k3_id[i, itail, :nk3, :] = k3_id[:nk3, :]
            neurons_k3_id_ini[i, itail, :nk3, :] = k3_id[:nk3, :]
            neurons_k3[i, itail, :nk3] = k3[:nk3]
            neurons_k3_ini[i, itail, :nk3] = k3[:nk3]
            neurons_cosangle[i, itail, :nk3] = np.cos(angle[:nk3]/180.0*pi)
            neurons_cosangle_ini[i, itail, :nk3] = np.cos(angle[:nk3]/180.0*pi)
            neurons_velocity[i, itail, :, :] = 0.0

    return
#
#
###############################################################################
#  create tail branch and bulge to all neurons
###############################################################################
#
#


def add_tail_data(n_neurons, n_tails, neurons_tail_active,neurons_tail_ntot,neurons_npoints_ini,                 
                        neurons_tail_hasparent, neurons_tail_nchildren, neurons_tail_level):

    for i in prange(n_neurons):

        for itail in range (n_tails):

            neurons_tail_active[i, itail] = 0
       

            if (itail ==0):
                 neurons_tail_active[i, itail] = 1  # this is 'tail1'

            neurons_tail_ntot[i, itail] = neurons_npoints_ini[i]
            neurons_tail_nchildren[i,itail]=0
            

    return

def add_bulge_data(n_neurons: int, bulge_pc: float, bulge_minpoints: int, bulge_create_rate: float, bulge_delete_rate: float,
                   neurons_bulge_pc, neurons_bulge_minpoints, neurons_bulge_create_rate, neurons_bulge_delete_rate, neurons_bulge_mass,
                   neurons_mass_ini, neurons_bulge_size, neurons_size_ini):
    for i in prange(n_neurons):
        neurons_bulge_pc[i] = bulge_pc
        neurons_bulge_minpoints[i] = bulge_minpoints
        neurons_bulge_create_rate[i] = bulge_create_rate
        neurons_bulge_delete_rate[i] = bulge_delete_rate
        neurons_bulge_mass[i, 0] = neurons_mass_ini[i, 0, 0]
        neurons_bulge_mass[i, 1] = (
            1.0-neurons_bulge_pc[i]) * neurons_mass_ini[i, 0, 1]
        neurons_bulge_mass[i, 2] = neurons_bulge_pc[i] * \
            neurons_mass_ini[i, 0, 1]
        neurons_bulge_size[i, 0] = neurons_size_ini[i, 0, 0]
        neurons_bulge_size[i, 1] = (
            1.0-neurons_bulge_pc[i])*neurons_size_ini[i, 0, 1]
        neurons_bulge_size[i, 2] = neurons_bulge_pc[i] * \
            neurons_size_ini[i, 0, 1]
    return

#
#



###############################################################################
# create or remove a bulge if only one tail
###############################################################################
#


def create_or_remove_bulge(n_neurons: int, n_tails, perturbations_field, box_dx, box_dy, box_nx, box_ny,
                           neurons_bulge_active, neurons_bulge_minpoints, neurons_x, neurons_y,
                           neurons_bulge_create_rate, neurons_tail_active, neurons_npoints_tot,
                           neurons_tail_node, neurons_bulge_delete_rate, bulge_mult_factor):



    seed = secrets.randbits(128)
    rng1 = np.random.default_rng(seed)
    dx = box_dx
    dy = box_dy
    nx = box_nx
    ny = box_ny
    mult_factor=bulge_mult_factor
    
    for i in range(n_neurons):
        # there can be only one bulge
        nbulge = sum(neurons_bulge_active[i, :])
        minpoints = neurons_bulge_minpoints[i]
        create_or_erase = rng1.random()
        my_x = neurons_x[i, 0, 1]
        my_y = neurons_y[i, 0, 1]
        ix = int(my_x/dx)
        if (ix < 0):
            ix = 0
        if (ix > nx-1):
            ix = nx-1
        iy = int(my_y/dy)
        if (iy < 0):
            iy = 0
        if (iy > ny-1):
            iy = ny-1
        if (create_or_erase < 0.5):
            prob_add = rng1.random()
            
            p_tot = neurons_bulge_create_rate[i] * (1+perturbations_field[ix, iy]*mult_factor)

            if (prob_add < p_tot):

                OK = True
                nr_activetails=sum(neurons_tail_active[i, :])


                if ((nr_activetails == 1) and (nbulge == 0)):

                    for j in range(n_tails):
                        if (neurons_tail_active[i, j] == 1):
                            id_tail = j+1

                    npoints_total = neurons_npoints_tot[i, id_tail - 1]
                    if (npoints_total < minpoints):
                        OK = False
               

                    if (OK == True):
                        neurons_bulge_active[i, id_tail-1] = 1
            else:
                p_tot= neurons_bulge_delete_rate[i] * (1+mult_factor*perturbations_field[ix, iy])
                prob_remove = rng1.random()
                if (prob_remove < p_tot):
                    neurons_bulge_active[i, :] = 0
    return
