import numpy as np
from scipy.optimize import minimize_scalar
from numpy.polynomial import Polynomial



nom_fichier = "traj_t_x_y_vx_vy.txt"
nom_out='traj_t_x_y_vx_vy_65pcmin.txt'
nr_neurons=400
nr_neurons_min=nr_neurons*0.65
y_min=745      #this is the basis of the cortical plate. Horizontal line in 800x1200 brain slice model. 



with open(nom_fichier, "r", encoding="utf-8") as f_read:
    with open(nom_out, "w", encoding="utf-8") as f_avg:    
        lines=f_read.readlines()
        #print(" nlines "+str(len(lines)))
        
        maxpoint=int(len(lines))
        trigger=0
        il=0
        
        while il < maxpoint-nr_neurons-1: 
            n_count=0
            for ineuron in range (0, nr_neurons):               
                parts = lines[il].split()
                il+=1
             

                x=float(parts[1])
                y=float(parts[2])
                
                #print(" il ",il,x,y)
                if (y >= y_min):
                    n_count+=1
                if(trigger==1):
                    ligne = f"{x:9.5f} \t{y:9.5f} \n"
                    f_avg.write(ligne)
            #after this count if nr_min are above limit, print the reduced trajectory file
            if (n_count >= nr_neurons_min):
                print('Threshold reached, printing configuration')
                trigger=1 
            if (trigger==1):
            	print(f'cfg {il/nr_neurons} Fraction of neurons above threshold {n_count/nr_neurons}')







