import numpy as np

# Nom du fichier texte
nom_fichier = "speed.txt"
nom_out='speed_diff_8pc.txt'
nneurons=120
neurons_speed=np.zeros((nneurons), dtype=float)
nb_avg=0.0
avg_speed=0.0
avg_vy=0.0
# Ouverture et lecture du fichier
with open(nom_fichier, "r", encoding="utf-8") as f_read:
    with open(nom_out, "w", encoding="utf-8") as f_avg:    
        lines=f_read.readlines()
        #print(" nlines "+str(len(lines)))
        il = 1
        maxsteps=int(len(lines))
        for i in range(maxsteps-1):
            parts = lines[il].split()
            il+=1
            neurons_speed[i]=float(parts[3])

        #sort
        for i in range(maxsteps-2):
            minim=neurons_speed[i]
            for j in range(i+1,maxsteps-1):
                if neurons_speed[j]< minim:
                    t=neurons_speed[i] 
                    neurons_speed[i]=neurons_speed[j]
                    neurons_speed[j]=t
                    minim= neurons_speed[i]
            print(f" {i} {t}")

        for i in range(int(0.1*maxsteps), int(maxsteps-0.1*maxsteps)):
            nb_avg+=1
            avg_speed+=neurons_speed[i]
            print(f"{i} {neurons_speed[i]}")
        avg_speed/=(1.*nb_avg)
        print(f" avg over {nb_avg} = {avg_speed}")

