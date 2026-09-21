import numpy as np

# Nom du fichier texte
nom_fichier = "traj_t_x_y_vx_vy.txt"
nom_out='traj_t_x_y_vx_vy_AVG50'
nneurons=40
navg=50
neurons_nbranches=np.zeros((nneurons), dtype=float)
neurons_contacts=np.zeros((nneurons), dtype=float)
nb_avg=0.0
i_avg=0
c_avg=0.0
avg_speed=0.0
avg_vy=0.0
NK=0.0
L2=0.0
L3=0.0
maxtime_speed=100000
mintime_speed=50000 
# Ouverture et lecture du fichier
with open(nom_fichier, "r", encoding="utf-8") as f_read:
    with open(nom_out, "w", encoding="utf-8") as f_avg:    
        lines=f_read.readlines()
        #print(" nlines "+str(len(lines)))
        il = 0
        maxsteps=int(len(lines)/nneurons/navg)*nneurons*navg
        while il < maxsteps-1:    
            parts = lines[il].split()
            i_avg+=1 
            #print(" il "+str(il)+" "+str(lines[il]))
            neurons_nbranches[:] =0.0
            neurons_contacts[:] = 0.0
            for iavg in range(navg):

                for ineuron in range(nneurons):
                    parts = lines[il].split()
                    il+=1
                    vx=float(parts[3])
                    vy=float(parts[4])
                    if (il < maxtime_speed)and (il > mintime_speed):
                        avg_speed+=np.sqrt(0.00000001+vx*vx+vy*vy) 
                        avg_vy+=vy

                    neurons_nbranches[ineuron] += float(parts[5])
                    neurons_contacts[ineuron] += float(parts[6])
                    NK += float(parts[7])
                    L2 += float(parts[8])
                    L3 += float(parts[10])

                    if ( float(parts[5])<-0.0001):
                        neurons_nbranches[ineuron]+=1
                
                
            for ineuron in range(nneurons):
                neurons_nbranches[ineuron]/=navg
                neurons_contacts[ineuron]/=navg
                nb = neurons_nbranches[ineuron]
                nc =  neurons_contacts[ineuron]
                ligne = f"{nb:9.5f} \t  {nc:9.5f}\n"
                f_avg.write(ligne)
                print(f"{ineuron} { neurons_nbranches[ineuron]} {nc}")
                nb_avg+=nb
                c_avg+=nc
        nb_avg/=(nneurons*i_avg)
        c_avg/=(nneurons*i_avg)
        NK/=(nneurons*navg*i_avg)
        L2/=(nneurons*navg*i_avg)
        L3/=(nneurons*navg*i_avg)
        avg_speed/=(maxtime_speed-mintime_speed)
        avg_vy/=(maxtime_speed-mintime_speed)
        print(f" average branching = {nb_avg}  contacts {c_avg} <v> {avg_speed} <vy> {avg_vy} NK {NK} L2 {L2} L3 {L3}")

                
