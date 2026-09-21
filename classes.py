
import numpy as np

global maxpoints
maxpoints=40
class Neuron:
    def __init__(self, neuron_id):
        self.maxpoints=maxpoints
        self.neuron_id = neuron_id
        self.activation = 0.0  # default activation level
        self.x=np.zeros((2,maxpoints), dtype=float)
        self.y=np.zeros((2,maxpoints), dtype=float)
        self.x_ini=np.zeros((2,maxpoints), dtype=float)
        self.y_ini=np.zeros((2,maxpoints), dtype=float)
        self.npoints=0
        self.npoints_ini=0
        self.npoints_tot=np.zeros((2), dtype=int)
        
        self.size           = np.zeros((2,maxpoints), dtype=float)
        self.size[:]        = 1.0
        self.size_ini       = np.zeros((2,maxpoints), dtype=float)
        self.size_ini[:]    = 1.0
        self.nk2            = np.zeros((2), dtype=int)
        self.nk2[:]         = 0
        self.nk2_ini        = np.zeros((2), dtype=int)
        self.nk2_ini[:]     = 0.0
        self.nk3            = np.zeros((2), dtype=int)
        self.nk3[:]         = 0.0
        self.nk3_ini        = np.zeros((2), dtype=int)
        self.nk3_ini[:]     = 0.0
        self.k2             = np.zeros((2,maxpoints), dtype=float)
        self.k2_ini         = np.zeros((2,maxpoints), dtype=float)
        self.k3             = np.zeros((2,maxpoints), dtype=float)
        self.k3_ini         = np.zeros((2,maxpoints), dtype=float)
        self.k2_id          = np.zeros((2,maxpoints, 2), dtype=int)
        self.k2_id_ini      = np.zeros((2,maxpoints, 2), dtype=int)
        self.k3_id          = np.zeros((2,maxpoints, 3), dtype=int)
        self.k3_id_ini      = np.zeros((2,maxpoints, 3), dtype=int)
        self.length         = np.zeros((2,maxpoints, maxpoints), dtype=float)
        self.length_ini     = np.zeros((2,maxpoints, maxpoints), dtype=float)
        self.cosangle       = np.zeros((2,maxpoints), dtype=float)
        self.cosangle_ini   = np.zeros((2,maxpoints), dtype=float)
        self.mass           = np.zeros((2,maxpoints), dtype=float)
        self.mass_ini       = np.zeros((2,maxpoints), dtype=float)
        self.force          = np.zeros((2,maxpoints,2), dtype=float)
        self.velocity       = np.zeros((2,maxpoints,2), dtype=float)
        self.repulsion      = 10
        self.r_width        = 1.0
        self.r_cut          = 4.0
        
        
        
        
        self.jmp_dyn         = np.zeros((2,maxpoints), dtype=int)
        self.jmp_dyn_ini     = np.zeros((2,maxpoints), dtype=int)
        self.jmp_rate        = np.zeros((2,maxpoints), dtype=float)
        self.jmp_rate_ini    = np.zeros((2,maxpoints), dtype=float)
        self.jmp_amp         = np.zeros((2,maxpoints), dtype=float)
        self.jmp_amp_ini     = np.zeros((2,maxpoints), dtype=float)
        self.jmp_ang         = np.zeros((2,maxpoints), dtype=float) 
        self.jmp_ang_ini     = np.zeros((2,maxpoints), dtype=float) 
        self.tail_angle_start_min =0.0
        self.tail_angle_start_max =0.0
        self.tail_x          = np.zeros((maxpoints), dtype=float)
        self.tail_y          = np.zeros((maxpoints), dtype=float)
        
        self.tail_nmax       = np.zeros((2), dtype=int)
        self.tail_nmax[:]    = maxpoints/2
        self.tail_l          = np.zeros((2), dtype=float)
        self.tail_l[:]       = 0.0   
        self.tail_l0         = np.zeros((2), dtype=float)
        self.tail_l0[:]      = 0.0
        self.tail_k2         = np.zeros((2), dtype=float)
        self.tail_k2[:]      = 0.0
        self.tail_nk2         = np.zeros((2), dtype=int)
        self.tail_nk2[:]      = 0.0
        self.tail_k3         = np.zeros((2), dtype=float)
        self.tail_k3[:]      = 0.0
        self.tail_nk3         = np.zeros((2), dtype=int)
        self.tail_nk3[:]      = 0.0
        
        self.tail_angle      = np.zeros((2), dtype=float)
        self.tail_angle[:]   = 0.0
        self.tail_mass_max   = np.zeros((2), dtype=float)
        self.tail_mass_max[:]=0.0
        self.tail_active     = np.zeros((2), dtype=int)
        self.tail_active[:]  = 0
        self.tail_create_rate       = np.zeros((2), dtype=float)
        self.tail_create_rate[:]    = 0.0
        self.tail_delete_rate       = np.zeros((2), dtype=float)
        self.tail_delete_rate[:]    = 0.0
        self.tail_create_rate_ini       = np.zeros((2), dtype=float)
        self.tail_create_rate_ini[:]    = 0.0
        self.tail_delete_rate_ini       = np.zeros((2), dtype=float)
        self.tail_delete_rate_ini[:]    = 0.0
        self.tail_node       = np.zeros((2), dtype=int)
        self.tail_node[:]    =0.0


        
        self.bulge_active = np.zeros((2), dtype=int)
        self.bulge_pc=0.4
        self.bulge_minpoints=5
        self.bulge_create_rate=0.0
        self.bulge_delete_rate=0.0
        self.bulge_mass=np.zeros((3), dtype=float)
        self.bulge_size=np.zeros((3), dtype=float)
        self.old_mass = np.zeros((3), dtype=float)
        self.old_size = np.zeros((3), dtype=float)


    def set_npoints(self,npoints):
        self.npoints=npoints
    def get_npoints(self):
        npoints=self.npoints
        return npoints
    
    def __repr__(self):
        return f"Neuron(id={self.neuron_id}, activation={self.activation:.3f})"

class Trajectory:
    def __init__(self, maxtime, maxneurons):
        self.nlines=maxtime
        self.ncolumns = maxneurons+2
        self.traj=np.zeros((self.nlines,self.ncolumns), dtype=float)
        self.index=int
        self.plot1=int
        self.plot2=int
        
    def add_time(self, ntime):
        B=np.zeros((ntime, self.ncolumns))
        C= np.vstack((self.traj,B))
        self.traj=C
    def add_neuron(self, nneurons):
        B=np.zeros((self.nlines, nneurons))
        C= np.vstack((self.traj,B))
        self.traj=C   
        

class Universe:
    def __init__(self, xf, yf, nx:int, ny:int):
        self.x0 = 0.0
        self.y0 = 0.0
        self.xf = xf
        self.yf = yf
        self.nx = nx
        self.ny = ny
        self.dx=self.xf/self.nx
        self.dy=self.yf/self.ny
        self.n_neurons=0
        self.friction=0
        self.friction_min=0.0
        self.friction_max=0.0
        self.friction_profile=""
        self.jumprate=1.0
        self.jumprate_min=0.0
        self.jumprate_max=0.0
        self.jumprate_profile=""
        self.jumpamp=1.0
        self.jumpamp_min=0.0
        self.jumpamp_max=0.0
        self.jumpamp_profile=""
        self.tailrate=1.0
        self.tailrate_min=0.0
        self.tailrate_max=0.0
        self.tailrate_profile=""
        self.branchrate=1.0
        self.branchrate_min=0.0
        self.branchrate_max=0.0
        self.branchrate_profile=""
        self.MCFIELD=""
        self.plotfield=""
        
        
        
        
        self.const_force=0.0
        self.reset_vel=0.0
        self.timestep=0
        self.plotfield="pot"
        self.temperature=1.0
        self.accept=1.0
        self.reject=1.0
        self.PBC=False
        
    def set_universe(self, xf, yf, nx, ny):
        self.xf = xf
        self.yf = yf
        self.nx = nx
        self.ny = ny
        self.dx=self.xf/self.nx
        self.dy=self.xy/self.ny

    def resize(self, dx, dy):
        self.xf = dx
        self.yf = dy
    def set_n_neurons (self, n_neurons):
        self.n_neurons=n_neurons
    def get_n_neurons (self, n_neurons):
        return self.n_neurons    
    def get_dx(self):
        return self.dx
    def get_dy(self):
        return self.dy
    def get_nx(self):
        return self.nx
    def get_ny(self):
        return self.ny
    
class Field :
    def __init__(self, nx:int, ny:int, val) :
        self.nx = nx
        self.ny = ny
        self.val = np.array(val)
        # Validate shape
        if self.val.shape != (nx, ny):
            raise ValueError(
                f"Data shape {self.data.shape} does not match ({nx}, {ny})"
            ) 
    def set_field(self, nx:int, ny:int, val) :
        self.nx = nx
        self.ny = ny
        self.val = np.array(val)
        # Validate shape
        if self.val.shape != (nx, ny):
            raise ValueError(
                f"Data shape {self.data.shape} does not match ({nx}, {ny})"
            )            
    def __repr__(self):
        return f"Grid({self.nx}, {self.ny}, data=\n{self.data})"
    def get_value(self, i, j):
        """Return the value at position (i, j)."""
        return self.val[i, j]
    
    def get_val(self):
        return self.val
    
    def set_value(self, i, j, value):
        """Set the value at position (i, j)."""
        self.val[i, j] = value
    def get_nx(self):
        return self.nx
    def get_ny(self):
        return self.ny
# examplegrid = Grid(2, 3, [[1, 2, 3], [4, 5, 6]])
# print(grid)
#print("Value at (0, 2):", grid.get_value(0, 2))  # 3
#grid.set_value(1, 1, 99)
#print(grid)

