# plot_field.py

import numpy as np
from skimage import measure
import matplotlib as mpl
import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# Helper functions

# Compute the eigenvalues and eigenvectors of a symmetric matrix
# and return the eigenvector for the larger eigenvalue
def get_eigvec(smat):
    eigval, eigvec = np.linalg.eigh(smat)
    # Want the angle/phase of the minor axis
    axis = 0 if eigval[0] < eigval[1] else 1
    return eigvec[:,axis]

# List of parameters
class PlotCellFieldParams:
    def __init__(self):
        self.npoints = 0
        self.lx = 0
        self.ly = 0
        self.clx = 0
        self.cly = 0
        self.tstart = 0
        self.tend = 0
        self.tinc = 0
        self.field_dir = None
        self.field_name = None
        self.pos_file = None

class PlotCellField:
    def __init__(self, params):
        self.npoints = params.npoints
        self.lx = params.lx
        self.ly = params.ly
        self.clx = params.clx
        self.cly = params.cly
        self.tstart = params.tstart
        self.tend = params.tend
        self.tinc = params.tinc
        self.field_dir = params.field_dir
        self.field_name = params.field_name
        self.pos_file = params.pos_file
        self.nframes = (self.tend-self.tstart)//self.tinc+1
        self.xbuff = self.lx*0.2
        self.ybuff = self.ly*0.2

        # Define periodic locations
        self.periodic_loc = [(self.lx,-self.ly),(self.lx,0),(self.lx,self.ly),
                             (0,-self.ly),(0,0),(0,self.ly),
                             (-self.lx,-self.ly),(-self.lx,0),
                             (-self.lx,self.ly)]
        
        # Data arrays
        self.polygons = [[] for i in range(self.nframes)]
        self.points = [[] for i in range(self.nframes)]
        self.pos = np.zeros((self.nframes, self.npoints, 2), dtype=float)
        self.index_map = [[] for i in range(self.nframes)]
        self.time_map = [i*self.tinc+self.tstart for i in range(self.nframes)]

        # Extra data
        self.data_val = None
        
        # Extra components to plot
        self.comps = []

        # Focussing on a specific cell
        self.focus_index = None
        
        # Plot settings
        self.fontsize = 20
        self.cell_alpha = 1.0 # Transparency of cell interior
        self.edge_alpha = 1.0 # Transparency of cell edges

        # Latex settings
        rc={"font.size" : self.fontsize,
            "text.usetex" : True,
            "text.latex.preamble" :
            r"\usepackage{amsmath}" +
            r"\usepackage{amssymb}" +
            r"\usepackage{bm}" +
            r"\renewcommand\vec[1]{\bm{#1}}" +
            r"\renewcommand\familydefault{\sfdefault}" +
            r"\usepackage[scaled=1]{helvet}" +
            r"\usepackage[helvet]{sfmath}" +
            r"\setlength{\thinmuskip}{0mu}" +
            r"\setlength{\medmuskip}{0mu}" +
            r"\setlength{\thickmuskip}{0mu}"}
        
        for k in rc:
            mpl.rcParams[k] = rc[k]
        
        #plt.rcParams["font.family"] = "sans-serif"
        #plt.rcParams["font.sans-serif"] = "FreeSans" # A font close to Helvet.

        # Get axes and figures
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal") # For maintaining aspect ratio
        self.ax.set_xlim([0,self.lx])
        self.ax.set_ylim([0,self.ly])
        self.ax.tick_params(axis="both") #, labelsize=self.fontsize)

        # Colour bar
        self.cbar = None # Can be set using set_cbar()
        self.colourmap = mplcm.get_cmap("RdYlBu_r")

        # Default cbar tick params
        self.plot_cell_index = False
        self.tic_min = 0
        self.tic_max = self.npoints
        self.tic_inc = self.npoints/2

        # For mapping data values to colours
        self.mapper = None
        self._set_colour_mapper(0, self.npoints)
        
        # Draw borders but no axes and ticks
        plt.tick_params(axis="both", which="both", bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

        # Set plot margins
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                            wspace=0, hspace=0)        
        
        # Set plotting artists
        self.plt_artists = []
        self.plt_pts = None # For plotting cell cm - set using add_cell_cm()
        self.plt_time_txt = None # For displaying time text at the bottom
        self.time_txt_x = 0.5
        self.time_txt_y = -0.01        
        self.Dr = None
        self.dt = None
        
        # Set the artist for plotting the polygons
        # For drawing the interior of the cell
        self.patches_int = PatchCollection([], linewidth=0)
        self.plt_polygons_int = self.ax.add_collection(self.patches_int)

        # For drawing the exterior of the cell
        self.patches_out = PatchCollection([], linewidth=1)
        self.plt_polygons_out = self.ax.add_collection(self.patches_out)
        self.plt_polygons_out.set_facecolor((0.0,0.0,0.0,0.0))
        self.plt_polygons_out.set_edgecolor((0.0,0.0,0.0,self.edge_alpha))
        
        self.plt_artists.append(self.plt_polygons_int)
        self.plt_artists.append(self.plt_polygons_out)
        
        # Load the data
        self._load_data()
        
    def set_cbar(self, label=None, ticks=None):
        self.cbar = plt.colorbar(self.mapper, fraction=0.15, pad=0.02)
        if (ticks is None):
            self.cbar.set_ticks(np.arange(
                self.tic_min, self.tic_max+self.tic_inc*0.5, self.tic_inc))
        else:
            self.cbar.set_ticks(ticks[0])
            self.cbar.set_ticklabels(ticks[1])
        #self.cbar.ax.tick_params(labelsize=self.fontsize, direction="in",
        #                         size=5)
        self.cbar.ax.tick_params(direction="in", size=5)        
        self.cbar.ax.yaxis.set_ticks_position("both")
        if (label != None):
            self.cbar.ax.get_yaxis().labelpad = 25
            #self.cbar.ax.set_ylabel(r"{:s}".format(label),
            #                        fontsize=self.fontsize, rotation=270)
            self.cbar.ax.set_ylabel(r"{:s}".format(label), rotation=270)

    def set_cell_alpha(self, alpha):
        self.cell_alpha = alpha

    def set_edge_alpha(self, alpha):
        self.edge_alpha = alpha
        
    def colour_by_index(self):
        if (self.data_val is None):
            self.plot_cell_index = True

    def add_time_label(self, Dr, dt):
        self.Dr = Dr
        self.dt = dt
        #self.plt_time_txt = self.ax.text(0.5,0.01, "", fontsize=self.fontsize,
        #                                 horizontalalignment="center",
        #                                 transform=plt.gcf().transFigure)
        #self.plt_time_txt.set_fontsize(self.fontsize)
        self.plt_time_txt = self.ax.text(self.time_txt_x, self.time_txt_y,
                                         "", horizontalalignment="center",
                                         transform=plt.gcf().transFigure)
        self.plt_artists.append(self.plt_time_txt)
        
    def add_cell_cm(self):
        # Empty data
        self.plt_pts, = self.ax.plot([],[], '.', markersize=5, color="black")
        self.plt_artists.append(self.plt_pts)

    def add_data(self, pdata, cmap=None, use_cbar=True, cbar_discrete=False,
                 cbar_label=None, cbar_ticks=None, cmap_min=0.0, cmap_max=1.0):
        self.plot_cell_index = False
        # Reset the colour mapper
        mpl_cmap = mplcm.get_cmap(cmap) if cmap != None else self.colourmap
        self.colourmap = colors.LinearSegmentedColormap.from_list(
            "trunc_{:s}".format(mpl_cmap.name),
            mpl_cmap(np.linspace(cmap_min, cmap_max, 256)))
        self.tic_min = pdata.tic_min if pdata.tic_min != None else pdata.vmin
        self.tic_max = pdata.tic_max if pdata.tic_max != None else pdata.vmax
        self.tic_inc = pdata.tic_inc if pdata.tic_inc != None else \
            (self.tic_max-self.tic_min)/2.0
        if (cbar_discrete):
            bounds = np.arange(pdata.tic_min-self.tic_inc*0.5,
                               pdata.tic_max+self.tic_inc*1.5, self.tic_inc)
            self._set_colour_mapper(pdata.vmin, pdata.vmax, bounds)
        else:
            self._set_colour_mapper(pdata.vmin, pdata.vmax)
        if (use_cbar): self.set_cbar(cbar_label, cbar_ticks)
                            
    def add_plot_comp(self, comp):
        self.comps.append(comp)
        for a in comp.get_artists():
            self.plt_artists.append(a)

    def set_bound(self, xmin, xmax, ymin, ymax):
        self.ax.set_xlim([xmin, xmax])
        self.ax.set_ylim([ymin, ymax])

    def focus(self, index, width=None, time=None):
        if (width == None): width = 20
        if (time == None): time = self.tstart
        focus_width = width
        frame = (time-self.tstart)//self.tinc
        xcm = self.pos[frame,index,0]
        ycm = self.pos[frame,index,1]
        xmin = xcm-focus_width*0.5
        ymin = ycm-focus_width*0.5
        xmax = xcm+focus_width*0.5
        ymax = ycm+focus_width*0.5
        self.set_bound(xmin, xmax, ymin, ymax)
        self.focus_index = index
        
    def _plot(self, frame):
        print("Rendering frame {:d} out of {:d}".format(frame+1, self.nframes))
        # Plot cell polygons
        colours = []
        for pt in range(len(self.polygons[frame])):
            if (self.data_val is not None):
                colours.append(self.mapper.to_rgba(
                    self.data_val[frame,self.index_map[frame][pt]],
                    alpha=self.cell_alpha))
            elif (self.plot_cell_index):
                colours.append(self.mapper.to_rgba(
                    self.index_map[frame][pt], alpha=self.cell_alpha))
            else:
                # Plot cells in white (transparent)
                colours.append((0,0,0,0))
        self.plt_polygons_int.set_paths(self.polygons[frame])
        self.plt_polygons_int.set_facecolor(colours)
        
        self.plt_polygons_out.set_paths(self.polygons[frame])
        colour = self.plt_polygons_out.get_edgecolor()[0]
        colour[3] = self.edge_alpha
        self.plt_polygons_out.set_edgecolor(colour)
        
        # Plot centres of cells
        if (self.plt_pts != None):
            self.plt_pts.set_xdata(self.points[frame][:,0])
            self.plt_pts.set_ydata(self.points[frame][:,1])
            self.plt_pts.set_alpha(self.edge_alpha)
        
        # Plot time label
        if (self.plt_time_txt != None):
            # Adjust label location if the colorbar is on
            self.time_txt_x = 0.5 if self.cbar == None else 0.48
            self.plt_time_txt.set_position((self.time_txt_x,self.time_txt_y))
            self.plt_time_txt.set_text(r"$D_rt = {:g}$".format(
                self.time_map[frame]*self.Dr*self.dt))

        # Call other artists
        for comp in self.comps:
            comp.update(frame, self)
        
        return self.plt_artists
        
    def plot(self, out_file, make_movie=False, print_to_screen=False):
        if (make_movie):
            if (print_to_screen):
                ani = FuncAnimation(self.fig, self._plot,
                                    np.arange(self.nframes), 
                                    fargs=[], interval=1)
                plt.show()
            else:
                ani = FuncAnimation(self.fig, self._plot,
                                    np.arange(self.nframes), 
                                    fargs=[], interval=1)
                Writer = animation.writers["ffmpeg"]
                writer = Writer(fps=10, bitrate=1000)
                ani.save(out_file, writer=writer)
        else:
            self._plot(0) # Plot the start frame (the only frame)
            if (print_to_screen):
                plt.show()
            else:
                plt.savefig(out_file, bbox_inches="tight", transparent=True)

    def _set_colour_mapper(self, vmin, vmax, bounds=None):
        if (bounds is None):
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = mpl.colors.BoundaryNorm(bounds, self.colourmap.N)
        self.mapper = mplcm.ScalarMappable(norm=norm, cmap=self.colourmap)
        self.mapper.set_array([])
                
    def _load_data(self):
        # Read position and cell field data
        nlines = self.npoints+2
        reader = open(self.pos_file,'r')
        while True:
            # Read header section (including time info)
            for i in range(2):
                line = reader.readline()            
            if (not line): break # Reached end of file
            data = line.split()
            time = int(data[1])
            if (time > self.tend):
                break
            elif (time < self.tstart or (time-self.tstart)%self.tinc != 0):
                for i in range(self.npoints):
                    line = reader.readline()
            else:
                print("Reading positions at timestep = {:d}".format(time))
                frame = (time-self.tstart)//self.tinc
                for n in range(self.npoints):
                    line = reader.readline()
                    data = line.split()
                    
                    # Read wrapped position data
                    self.pos[frame,n,0] = float(data[0])
                    self.pos[frame,n,1] = float(data[1])
                    
                    # Read cell field data
                    self._read_field(n, frame, time)
        reader.close()
        # Convert pos data to numpy arrays
        for i in range(self.nframes):
            self.points[i] = np.array(self.points[i])
    
    def _add_polygon(self, index, cxcm, cycm, xcm, ycm, dx, dy, contour,
                     frame):
        xcm += dx
        ycm += dy
        if (xcm < self.lx+self.xbuff and xcm > -self.xbuff and
            ycm < self.ly+self.ybuff and ycm > -self.ybuff):
            poly = np.copy(contour)
            poly[:,0] += (xcm-cxcm)+0.5
            poly[:,1] += (ycm-cycm)+0.5
            self.points[frame].append((xcm,ycm))
            self.polygons[frame].append(Polygon(poly))
            self.index_map[frame].append(index)    
        
    def _read_field(self, index, frame, time):
        filename = "{:s}/cell_{:d}/{:s}_cell_{:d}.dat.{:d}".format(
            self.field_dir, index, self.field_name, index, time)
        with open(filename,'r') as reader:
            # For computing the local field centre of mass
            xavg = 0.0
            yavg = 0.0
            mass = 0.0
            local_field = np.zeros((self.clx,self.cly))
            for l, line in enumerate(reader):
                data = line.split()
                if (len(data) != 3): continue
                i = int(data[0])
                j = int(data[1])
                phi = float(data[2])
                local_field[i,j] = phi
                xavg += (phi*(i+0.5)) # Use the centre of a lattice element
                yavg += (phi*(j+0.5))
                mass += phi
            if (mass > 0.0):
                xavg /= mass
                yavg /= mass
            else:
                xavg = 0.0
                yavg = 0.0
            contours = measure.find_contours(local_field, 0.5)
            contour = contours[0]
            for pt in self.periodic_loc:
                self._add_polygon(
                    index, xavg, yavg, self.pos[frame,index,0], 
                    self.pos[frame,index,1], pt[0], pt[1], contour, frame)


class PlotForce:
    def __init__(self, pfield, force_file, scale=600.0):
        self.pfield = pfield
        npoints = pfield.npoints
        nframes = pfield.nframes
        tstart = pfield.tstart
        tend = pfield.tend
        tinc = pfield.tinc
        self.ncomps = 5 # Number of force components
        zorder = 5
        self.plot_order = np.array([4,0,3,1,2])+zorder

        # Default plot settings
        self.scale = scale
        self.width = 0.25

        # Set up colours for plotting different force components
        self.colourmap = mplcm.get_cmap("tab10")
        self.colours = [None for i in range(self.ncomps)]
        self.colours[0] = self.colourmap(3) # Capillary - red
        self.colours[1] = self.colourmap(0) # Polar - blue
        self.colours[2] = (0,0,0,0) # Active shear - gray (not shown)
        self.colours[3] = self.colourmap(2) # Viscous - green
        self.colours[4] = self.colourmap(1) # Damping - orange
            
        if (pfield.focus_index == None):
            self.forces = np.zeros((nframes, npoints, self.ncomps,2),
                                   dtype=float)
        else:
            self.forces = np.zeros((nframes, self.ncomps, 2), dtype=float)
        
        # Read the data
        reader = open(force_file,'r')
        while True:
            # Read header section (including time info)
            for i in range(2):
                line = reader.readline()
            if (not line): break # Reached end of file
            data = line.split()
            time = int(data[1])
            if (time > tend):
                break
            elif (time < tstart or (time-tstart)%tinc != 0):
                for i in range(npoints):
                    reader.readline()
            else:
                print("Reading forces at timestep {:d}".format(time))
                frame = (time-tstart)//tinc
                for i in range(npoints):
                    line = reader.readline()
                    data = line.split()
                    if (pfield.focus_index == None):
                        for n in range(self.ncomps):
                            self.forces[frame,i,n,0] = float(data[n*2])
                            self.forces[frame,i,n,1] = float(data[n*2+1])
                    elif (pfield.focus_index == i):
                        for n in range(self.ncomps):
                            self.forces[frame,n,0] = float(data[n*2])
                            self.forces[frame,n,1] = float(data[n*2+1])
        reader.close()

        # Set up the artist
        ax = pfield.ax
        self.plt_forces = []
        for n in range(self.ncomps):
            if (pfield.focus_index == None):
                self.plt_forces.append(
                    ax.quiver(pfield.pos[0,:,0], pfield.pos[0,:,1],
                              self.forces[0,:,n,0], self.forces[0,:,n,1],
                              units="xy", scale=1.0/self.scale,
                              color=self.colours[n], width=self.width,
                              zorder=self.plot_order[n]))
            else:
                self.plt_forces.append(
                    ax.quiver(pfield.pos[0,:,0], pfield.pos[0,:,1],
                              self.forces[0,n,0], self.forces[0,n,1],
                              units="xy", scale=1.0/self.scale,
                              color=self.colours[n], width=self.width,
                              zorder=self.plot_order[n]))
                
    def update(self, frame, pfield):
        if (pfield.focus_index == None):
            for n in range(self.ncomps):
                self.plt_forces[n].set_offsets(pfield.pos[frame,:,0:2])
                self.plt_forces[n].set_UVC(self.forces[frame,:,n,0],
                                           self.forces[frame,:,n,1])
        else:
            for n in range(self.ncomps):
                self.plt_forces[n].set_offsets(
                    pfield.pos[frame,pfield.focus_index,0:2])
                self.plt_forces[n].set_UVC(self.forces[frame,n,0],
                                           self.forces[frame,n,1])
        
    def get_artists(self):
        return self.plt_forces,

    
class PlotDeformAngle:
    def __init__(self, pfield, deform_file):
        pi = np.pi
        self.vmin = 0.0
        self.vmax = pi
        self.tic_min = self.vmin
        self.tic_max = self.vmax
        self.tic_inc = pi/2.0
        pfield.data_val = np.zeros((pfield.nframes,pfield.npoints))
        smat = np.zeros((2,2))
        reader = open(deform_file,'r')
        while True:
            # Read header section (including time info)
            for i in range(2):
                line = reader.readline()
            if (not line): break # Reached end of file
            data = line.split()
            time = int(data[1])
            if (time > pfield.tend):
                break
            elif (time < pfield.tstart or
                  (time-pfield.tstart)%pfield.tinc != 0):
                for i in range(pfield.npoints):
                    reader.readline()
            else:
                frame = (time-pfield.tstart)//pfield.tinc
                for n in range(pfield.npoints):
                    line = reader.readline()
                    data = line.split()
                    smat[0,0] = float(data[0])
                    smat[1,1] = float(data[1])
                    smat[0,1] = float(data[2])
                    smat[1,0] = smat[0,1]
                    eigvec = get_eigvec(smat)
                    angle = np.arctan(eigvec[1]/eigvec[0])
                    if (angle < 0.0): angle += pi
                    pfield.data_val[frame,n] = angle                        
        reader.close()

    def update(self, frame, pfield):
        pass # Nothing to update
    
    def get_artists(self):
        pass # No new artist generated

class PlotSimpleData:
    def __init__(self, pfield, data_col, vmin, vmax, data_file,
                 tic_min=None, tic_max=None, tic_inc=None):
        self.vmin = vmin
        self.vmax = vmax
        self.tic_min = tic_min
        self.tic_max = tic_max
        self.tic_inc = tic_inc
        pfield.data_val = np.zeros((pfield.nframes,pfield.npoints))
        reader = open(data_file,'r')
        while True:
            # Read header section (including time info)
            for i in range(2):
                line = reader.readline()
            if (not line): break # Reached end of file
            data = line.split()
            time = int(data[1])
            if (time > pfield.tend):
                break
            elif (time < pfield.tstart or
                  (time-pfield.tstart)%pfield.tinc != 0):
                for i in range(pfield.npoints):
                    reader.readline()
            else:
                frame = (time-pfield.tstart)//pfield.tinc
                for n in range(pfield.npoints):
                    line = reader.readline()
                    data = line.split()
                    pfield.data_val[frame][n] = float(data[data_col])
        reader.close()
    
    def update(self, frame, pfield):
        pass # Nothing to update
    
    def get_artists(self):
        pass # No new artist generated


class PlotDirector:
    def __init__(self, pfield, director_file, alpha=1.0):
        self.director_file = director_file
        self.lx = pfield.lx
        self.ly = pfield.ly
        ax = pfield.ax
        self.ndir = np.zeros((self.lx, self.ly, 2))        
        YY, XX = np.meshgrid(np.arange(0,self.lx,2), np.arange(0,self.ly,2))
        self.plt_ndir = ax.quiver(XX, YY, self.ndir[::2,::2,0],
                                  self.ndir[::2,::2,1], angles="xy",
                                  units="xy", scale_units="xy", pivot="mid",
                                  scale=0.5, width=0.3, headlength=0,
                                  headaxislength=0, color="black", alpha=alpha)
        # self.plt_def_vec = ax.quiver(self.vec[0,:,0], self.vec[0,:,1],
        #                              self.vec[0,:,2], self.vec[0,:,3],
        #                              angles="xy", units="xy", scale_units="xy",
        #                             pivot="mid", color="black", headlength=0,
        #                            headaxislength=0, alpha=alpha)

        
    def update(self, frame, pfield):
        # Read the director field data
        time = pfield.time_map[frame]
        dir_file = "{:s}.{:d}".format(self.director_file,time)
        with open(dir_file, "r") as reader:
            for line in reader:
                data = line.split()
                if (len(data) == 0): continue
                x = int(data[0])
                y = int(data[1])
                self.ndir[x,y,0] = float(data[2])
                self.ndir[x,y,1] = float(data[3])
        self.plt_ndir.set_UVC(self.ndir[::2,::2,0], self.ndir[::2,::2,1])
        
    def get_artists(self):
        return self.plt_ndir,

class PlotDefect:
    def __init__(self, pfield, defect_file):
        self.defect_file = defect_file
        ax = pfield.ax

        # Read defects
        self.defects = [[] for n in range(pfield.nframes)]
        with open(self.defect_file, "r") as reader:
            while True:
                # Read header
                data = reader.readline().split()
                if (len(data) == 0): break
                ndefects = int(data[1])
                data = reader.readline().split()
                time = int(data[1])
                if (time > pfield.tend):
                    break
                elif (time < pfield.tstart or
                      (time-pfield.tstart) % pfield.tinc != 0):
                    for n in range(ndefects):
                        reader.readline()
                    continue
                frame = int((time-pfield.tstart) // pfield.tinc)
                for n in range(ndefects):
                    data = np.asarray(reader.readline().split()).astype(float)
                    self.defects[frame].append(data)
                self.defects[frame] = np.asarray(self.defects[frame])
        
        self.plt_pve_defects, = ax.plot([],[],'o',markersize=8,color="red")
        self.plt_nve_defects, = ax.plot([],[],'s',markersize=8,color="blue")

    def update(self, frame, pfield):
        dft = self.defects[frame]
        if (len(dft) > 0):
            pve_defects = dft[dft[:,2] > 0.0]
            nve_defects = dft[dft[:,2] < 0.0]
            self.plt_pve_defects.set_xdata(pve_defects[:,0])
            self.plt_pve_defects.set_ydata(pve_defects[:,1])
            self.plt_nve_defects.set_xdata(nve_defects[:,0])
            self.plt_nve_defects.set_ydata(nve_defects[:,1])
        else:
            self.plt_pve_defects.set_xdata([])
            self.plt_pve_defects.set_ydata([])
            self.plt_nve_defects.set_xdata([])
            self.plt_nve_defects.set_ydata([])
            
    def get_artists(self):
        return self.plt_pve_defects, self.plt_nve_defects,

class PlotDeformAxis:
    def __init__(self, pfield, deform_file, alpha=1.0):
        self.vec = np.zeros((pfield.nframes,pfield.npoints,4))
        vec_len = 0.5*np.sqrt(pfield.lx*pfield.ly/pfield.npoints)
        smat = np.zeros((2,2))
        reader = open(deform_file,'r')
        while True:
            # Read header section (including time info)
            for i in range(2):
                line = reader.readline()
            if (not line): break # Reached end of file
            data = line.split()
            time = int(data[1])
            if (time > pfield.tend):
                break
            elif (time < pfield.tstart or
                  (time-pfield.tstart)%pfield.tinc != 0):
                for i in range(pfield.npoints):
                    reader.readline()
            else:
                frame = (time-pfield.tstart)//pfield.tinc
                for n in range(pfield.npoints):
                    line = reader.readline()
                    data = line.split()
                    smat[0,0] = float(data[0])
                    smat[1,1] = float(data[1])
                    smat[0,1] = float(data[2])
                    smat[1,0] = smat[0,1]
                    eigvec = get_eigvec(smat)
                    eigvec *= vec_len
                    self.vec[frame,n,0] = pfield.pos[frame,n,0]#-eigvec[0]*0.5
                    self.vec[frame,n,1] = pfield.pos[frame,n,1]#-eigvec[1]*0.5
                    self.vec[frame,n,2] = eigvec[0]
                    self.vec[frame,n,3] = eigvec[1]                        
        reader.close()
        
        # Set up the artist
        ax = pfield.ax
        self.plt_def_vec = ax.quiver(self.vec[0,:,0], self.vec[0,:,1],
                                     self.vec[0,:,2], self.vec[0,:,3],
                                     angles="xy", units="xy", scale_units="xy",
                                     pivot="mid", color="black", headlength=0,
                                     headaxislength=0, alpha=alpha)

    def update(self, frame, pfield):
        self.plt_def_vec.set_offsets(self.vec[frame,:,0:2])
        self.plt_def_vec.set_UVC(self.vec[frame,:,2], self.vec[frame,:,3])
    
    def get_artists(self):
        return self.plt_def_vec,

    
"""
class PlotDefectTrajectory:
    def __init_(self, pfield, traj_file):
        self.traj_file = traj_file
        self.lx = pfield.lx
        self.ly = pfield.ly

        # Read defect trajectories
        self.defect_traj_charge = []
        defect_traj_pos = []
        defect_traj_pos_pbc = []
        self.defect_traj_start = []
        self.defect_traj_end = []
        self.defect_traj_time = []        
        with open(traj_file, "r") as reader:
            for line in reader:
                data = np.asarray(line.split()).astype(float)
                self.defect_traj_charge.append(data[0])
                self.defect_traj_start.append(data[1])
                self.defect_traj_end.append(data[2])
                defect_traj_pos.append(np.reshape(data[4:],(-1,2)))
        self.defect_traj_charge = np.asarray(self.defect_traj_charge)
        self.defect_traj_start = np.asarray(self.defect_traj_start, dtype=int)
        self.defect_traj_end = np.asarray(self.defect_traj_end, dtype=int)

        # Check periodic boundaries for trajectories
        for i,traj in enumerate(defect_traj_pos):
            defect_traj_pos_pbc.append([[traj[0,0],traj[0,1]],])
            defect_traj_time.append([defect_traj_start[i]])
            npts = traj.shape[0]
            for j in range(1,npts):
                time = defect_traj_start[i]+j*pfield.tinc
                ix = int(np.round((traj[j,0]-traj[j-1,0])/self.lx))
                ix = int(np.round((traj[j,1]-traj[j-1,1])/self.ly))
                if (ix != 0 or iy != 0):
                    if (ix > 0):
                        x0,x1 = 0,lx
                    elif (ix < 0):
                        x0,x1 = lx,0
                    else:
                        x0 = x1 = (traj[j,0]+traj[j-1,0])*0.5
                    if (iy > 0):
                        y0,y1 = 0,ly
                    elif (iy < 0):
                        y0,y1 = ly,0
                    else:
                        y0 = y1 = (traj[j,1]+traj[j-1,1])*0.5
                    defect_traj_pos_pbc[i].append([x0,y0])
                    defect_traj_pos_pbc[i].append([np.nan,np.nan])
                    defect_traj_pos_pbc[i].append([x1,y1])
                    defect_traj_time[i] += [time]*3
                defect_traj_pos_pbc[i].append(traj[j,0],traj[j,1])
                defect_traj_time[i] += [time]
            defect_traj_pos_pbc[i] = np.asarray(defect_traj_pos_pbc[i])
            defect_traj_time[i] = np.asarray(defect_traj_time[i])
        del defect_traj_pos
        defect_traj_pos = defect_traj_pos_pbc
        del defect_traj_pos_pbc
                        
        ax = pfield.ax
        self.ndir = np.zeros((self.lx, self.ly, 2))        
        YY, XX = np.meshgrid(X,Y)
        self.plt_ndir = ax.quiver(XX, YY, self.ndir[::2,::2,0],
                                  self.ndir[::2,::2,1], angles="xy",
                                  units="xy", scale_units="xy", pivot="mid",
                                  scale=0.5, width=0.1, headlength=0,
                                  headaxislength=0)

    def update(self, frame, pfield):
        # Read the director field data
        time = pfield.time_map[frame]
        dir_file = "{:s}.{:d}".format(self.director_file,time)
        with open(dir_file, "r") as reader:
            data = line.split()
            if (len(data) == 0): continue
            x = int(data[0])
            y = int(data[1])
            self.ndir[x,y,0] = float(data[2])
            self.ndir[x,y,1] = float(data[3])
        self.plt_ndir.set_UVC(self.ndir[::2,::2,0], self.ndir[::2,::2,1])
        
    def get_artists(self):
        return self.plt_ndir
    
"""
