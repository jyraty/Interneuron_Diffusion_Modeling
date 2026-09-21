import numpy as np

# Nom du fichier texte
nom_fichier = "branches.txt"
nom_out='branches_AVG50'
nneurons=120
navg=50
neurons_nbranches=np.zeros((nneurons), dtype=float)
neurons_lengths=np.zeros((nneurons), dtype=float)
nb_avg=0.0
i_avg=0
c_avg=0.0
avg_speed=0.0
avg_vy=0.0
maxtime_speed=4000
mintime_speed=2000 
# Ouverture et lecture du fichier
with open(nom_fichier, "r", encoding="utf-8") as f_read:
    with open(nom_out, "w", encoding="utf-8") as f_avg:    
        lines=f_read.readlines()
        #print(" nlines "+str(len(lines)))
        il = 1
        maxsteps=int(len(lines))
        parts = lines[il].split()
        i_avg+=1
        while il < maxsteps-navg*nneurons:    
            parts = lines[il].split()
            i_avg+=1 
            #print(" il "+str(il)+" "+str(lines[il]))
            neurons_nbranches[:] =0.0
            neurons_lengths[:] = 0.0
            for iavg in range(navg):

                    parts = lines[il].split()
                    il+=1
                    index=int(parts[0])
                    newint=index
                    print(f" index {index} {parts[0]}")
                    while (newint==index):
                        ineuron=int(parts[1])
                        neurons_nbranches[ineuron]+=1
                        neurons_lengths[ineuron]+=int(parts[4])/int(parts[2])
                        parts = lines[il].split()
                        il+=1
                        newint=int(parts[0])
                        print(f"{ineuron} {neurons_nbranches[ineuron]} {neurons_lengths[ineuron]}")


            for ineuron in range(nneurons):
                neurons_nbranches[ineuron]/=(1.*navg)
                neurons_lengths[ineuron]/=(1.*navg)
                nb = neurons_nbranches[ineuron]
                nc =  neurons_lengths[ineuron]
                ligne = f"{nb:9.5f} \t  {nc:9.5f}\n"
                f_avg.write(ligne)
                print(f"{ineuron} { neurons_nbranches[ineuron]} {nc}")
                nb_avg+=nb
                c_avg+=nc
        nb_avg/=(nneurons*i_avg)
        c_avg/=(nneurons*i_avg)
        print(f" average branching = {nb_avg}  lengths {c_avg} ")

                
