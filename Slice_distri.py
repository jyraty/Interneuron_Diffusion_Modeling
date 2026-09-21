import numpy as np






nom_fichier = "traj_t_x_y_vx_vy.txt"
nom_out='traj_t_x_y_vx_vy_slice.txt'
nr_to_avg=8001
nr_neurons=180
x=0.0
y=0.0
y1=63.0+14.0
y2=145.0-14.0



n_channel=0.0
n_center=0.0
avg_channel=0.0
avg_center=0.0
sig_channel=0.0
sig_center=0.0




with open(nom_fichier, "r", encoding="utf-8") as f_read:
    with open(nom_out, "w", encoding="utf-8") as f_avg:    
        lines=f_read.readlines()
        #print(" nlines "+str(len(lines)))
        
        maxpoint=int(len(lines))
        minpoint=maxpoint-nr_to_avg

        if(minpoint < 0):
            minpoint=0
        il=minpoint
        n_cfg=int((maxpoint-1-minpoint)/nr_neurons)
        
        while il < int((maxpoint-1)/nr_neurons)*nr_neurons:    
            parts = lines[il].split()
            il+=1
             

            x=float(parts[1])
            y=float(parts[2])

            if (y >=y1) and (y<=y2):
               n_center+=1.0
            else:
               n_channel+=1.0
        avg_channel=n_channel/n_cfg
        avg_center=n_center/n_cfg
        print(f'averages channel / center {avg_channel} / {avg_center}')

        il=minpoint
        i_neuron=0
        n_center=0.0
        n_channel=0.0

        while il < int((maxpoint-1)/nr_neurons)*nr_neurons:    
            parts = lines[il].split()
            il+=1
            i_neuron+=1

            x=float(parts[1])
            y=float(parts[2])

            if (y >=y1) and (y<=y2):
               n_center+=1.0
            else:
               n_channel+=1.0

            if(i_neuron==nr_neurons):
                sig_channel+=(n_channel-avg_channel)**2.0
                sig_center+=(n_center-avg_center)**2.0
                n_channel=0.0
                n_center=0.0
                i_neuron=0
                #print(f'cfg {il/nr_neurons} sigma channel / center {sig_channel} / {sig_center}')
        
        sig_channel=np.sqrt(sig_channel/n_cfg)
        sig_center=np.sqrt(sig_center/n_cfg)
        print(f'n_cfg {n_cfg} sigma channel / center {sig_channel} / {sig_center}')




