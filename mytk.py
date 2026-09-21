import numpy as np
import matplotlib.pyplot as plt

global fig
global canvas
                
def updatecanvas(fig, ax,canvas,plotvalue, box_x0, box_xf,box_nx,box_y0,box_yf,box_ny,box_plotfield, win_visible ):
    print(f'in update visible {win_visible}')
    xmin=box_x0
    xmax=box_xf
    nx=box_nx
    ymin=box_y0
    ymax=box_yf
    ny=box_ny
    xpoints = np.linspace(xmin, xmax, nx)
    ypoints = np.linspace(ymin, ymax, ny)
    swap_pot=np.swapaxes(plotvalue,0,1)
    #c = ax.pcolormesh(xpoints, ypoints, potvalue, shading='auto', cmap='pink')
     
    if (box_plotfield =='potential'):
        c = ax.pcolormesh(xpoints, ypoints, swap_pot, vmin=-1,vmax=40.0,shading='auto', cmap='pink'                     )
        ax.set_title("Potential")
        #ax.colorbar(c)
    if (box_plotfield =='friction'):
        c = ax.pcolormesh(xpoints, ypoints, swap_pot, shading='auto', cmap='copper'                     )
        ax.set_title("Friction")
        #ax.colorbar(c)
    if (box_plotfield =='jumpamp'):
        c = ax.pcolormesh(xpoints, ypoints, swap_pot, shading='auto', cmap='copper'                     )
        ax.set_title("Jump Amplitude")
        #ax.colorbar(c)
    if (box_plotfield =='jumprate'):
        c = ax.pcolormesh(xpoints, ypoints, swap_pot, shading='auto', cmap='copper'                     )
        ax.set_title("Jump Rate")
        #ax.colorbar(c)
    if (box_plotfield =='tailrate'):
        c = ax.pcolormesh(xpoints, ypoints, swap_pot, shading='auto', cmap='copper'                     )
        ax.set_title("Tail Rate")
    if (box_plotfield =='branchrate'):
        c = ax.pcolormesh(xpoints, ypoints, swap_pot, shading='auto', cmap='copper'                     )
        ax.set_title("Branch Rate")
    if (box_plotfield =='slipstick'):
        c = ax.pcolormesh(xpoints, ypoints, swap_pot, shading='auto', cmap='copper'                     )
        ax.set_title("Slipstick probability")
    if (box_plotfield =='affinity'):
            c = ax.pcolormesh(xpoints, ypoints, swap_pot, shading='auto', cmap='copper'                     )
            ax.set_title("Affinity field")
    fig.canvas.draw()
    if (win_visible):
        canvas.draw()   
    
    
def fig_to_rgb(fig):
    fig.canvas.draw()
    #img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = np.asarray(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    return img

              
def erase_and_replot_neuron(n_neurons:int, n_tails, n_levels, fig, ax,canvas, neurons,box_nx,box_ny,box_xf, shapes,\
                            neurons_tail_ntot,neurons_x,neurons_y, neurons_size,neurons_bulge_active,\
                            neurons_bulge_size, neurons_tail_active,neurons_tail_node, neurons_tail_level, win_visible):
    nx=box_nx
    ny=box_ny
    mid=box_xf/2.0
    #print(f'in replot visible {win_visible}')
# Remove old shapes
    for shape in shapes:
        
        shape.remove()
    
    shapes.clear()
    fig.canvas.draw()
    '''if (win_visible):
        canvas.draw()
    '''
#add neurons

    for i in range(n_neurons):
        #print(f"{n_neurons}  ") #+str(neurons_npoints))
        #depends if tail (0 or 1) is present
        #depends if bulge is present
        # if no tail : easy - no bulge
                  
        itail=0 # plot main branch
        colorlevel=neurons_tail_level[i,itail] 
        
        if (colorlevel == 1):
            myc='green'
        if (colorlevel == 2):
            myc='red'
        if (colorlevel == 3):
            myc='cyan'


        for j in range(neurons_tail_ntot[i,itail]):
            ix=neurons_x[i,0,j] #/box.dx
            iy=neurons_y[i,0,j] # /box.dy
            size=neurons_size[i,0,j]
            if (ix < size):
                ix=size
            if (ix > nx-size):
                ix=nx-size
            if (iy < size):
                 iy=size
            if (iy > ny-size):
                 iy=ny-size
            if (j < 3):
                if (neurons_bulge_active[i,itail] == 1 ):
                    size=neurons_bulge_size[i,j]                       
            circle = plt.Circle((ix, iy), size, color='yellow', alpha=0.5)
            ax.add_patch(circle)
            shapes.append(circle)

            if (j> 0):
                        
                if (colorlevel == 1):
                    myc='green'
                if (colorlevel == 2):
                    myc='red'
                if (colorlevel == 3):
                    myc='cyan'
                ixm1=neurons_x[i,0,j-1]#/box.dx
                iym1=neurons_y[i,0,j-1]#/box.dy
                linew=1
                
                if (j == 2):
                    if (neurons_bulge_active[i,itail] ==1 ):
                        linew=2
                        myc='yellow'
                if ( abs(ixm1-ix) < nx/2.0 )  :
                    line = plt.Line2D([ix, ixm1], [iy, iym1], color=myc, linewidth=linew)
                    ax.add_line(line)
                    shapes.append(line)

        for i_level in range(1,n_levels+1):            
            ilevel=n_levels-i_level+1


            for itail in range(1,n_tails): #+1):
                if (neurons_tail_active[i,itail] == 1) and (neurons_tail_level[i,itail]==ilevel ):
                    startpoint=neurons_tail_node[i,itail]
                    npoints_tot=neurons_tail_ntot[i,itail]
                    colorlevel=neurons_tail_level[i,itail] 
                
                    if (colorlevel == 1):
                        myc='green'
                    if (colorlevel == 2):
                        myc='red'
                    if (colorlevel == 3):
                        myc='cyan'

                    for j in range(startpoint,npoints_tot):
                        ix=neurons_x[i,itail,j]#/box.dx
                        iy=neurons_y[i,itail,j]#/box.dy
                        size=neurons_size[i,itail,j ]
                        if (ix < size):
                            ix=size
                        if (ix > nx-size):
                            ix=nx-size
                        if (iy < size):
                             iy=size
                        if (iy > ny-size):
                             iy=ny-size                       
                        if (j < 3):
                            if (neurons_bulge_active [i,itail] == 1 ):
                                size=neurons_bulge_size[i,j]
                        circle = plt.Circle((ix, iy), size, color='yellow', alpha=0.5)
                        ax.add_patch(circle)
                        shapes.append(circle)
                        if (j> 0):
                            ixm1=neurons_x[i,itail,j-1]#/box.dx
                            iym1=neurons_y[i,itail, j-1]#/box.dy
                            linew=1
                                
                            if (colorlevel == 1):
                                myc='green'
                            if (colorlevel == 2):
                                myc='red'
                            if (colorlevel == 3):
                                myc='cyan'
                            if (j == 2):
                                if (neurons_bulge_active[i,itail] ==1 ):
                                    linew=2
                                    myc='yellow'
                            if (abs(ixm1-ix)< nx/2.0)  :                                         
                                line = plt.Line2D([ix, ixm1], [iy, iym1], color=myc, linewidth=linew)
                                ax.add_line(line)
                                shapes.append(line)

    fig.canvas.draw()
    '''if (win_visible):
        canvas.draw()  '''          
    
    



