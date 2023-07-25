#!/bin/env python3
# plot_cell_field.py

import sys
import numpy as np
from plot_field import *

args = sys.argv
if (len(args) < 15):
    print("Usage: plot_cell_field.py npoints lx ly clx cly tstart tend tinc make_movie print_to_screen field_file_dir field_file_name pos_file out_file [--options ...]")
    sys.exit(1)

params = PlotCellFieldParams()

params.npoints = int(args.pop(1))
params.lx = int(args.pop(1))
params.ly = int(args.pop(1))
params.clx = int(args.pop(1))
params.cly = int(args.pop(1))
params.tstart = int(args.pop(1))
params.tend = int(args.pop(1))
params.tinc = int(args.pop(1))
make_movie = bool(int(args.pop(1)))
print_to_screen = bool(int(args.pop(1)))
params.field_dir = args.pop(1)
params.field_name = args.pop(1)
params.pos_file = args.pop(1)
out_file = args.pop(1)

pfield = PlotCellField(params)

# Check if other extra options are added
found_new_options = False
options = ["--force", "--data", "--colour-by-index", "--cm", "--director",
           "--focus", "--time-label", "--deform-angle", "--defect", "--cbar",
           "--deform-axis", "--edge-alpha", "--cell-alpha"]
show_cbar = False


# Tokenize by --
opts = []
nopts = -1
if (not args[1].startswith("--")):
    print("Error: optional arguments must begin with --")
    sys.exit(1)
for i in range(1,len(args)):
    if (args[i].startswith("--")):
        opts.append([])
        nopts += 1
    if (nopts >= 0):
        opts[nopts].append(args[i])

for opt in opts:
    option = opt[0]
    if (not option in options):
        print("Error: option {:s} does not exist".format(option))
        sys.exit(1)
    if (option == "--force"):
        if (len(opt) != 2):
            print("Usage: --force force_file")
            sys.exit(1)
        force_file = opt.pop(1)
        pforce = PlotForce(pfield, force_file)
        pfield.add_plot_comp(pforce)
    elif (option == "--data"):
        if (len(opt) < 10):
            print("Usage: --data data_col vmin vmax tic_min tic_max",
                  "tic_inc discrete label data_file [cmap_min cmap_max]")
            sys.exit(1)
        data_col = int(opt.pop(1))
        vmin = float(opt.pop(1))
        vmax = float(opt.pop(1))
        tic_min = float(opt.pop(1))
        tic_max = float(opt.pop(1))
        tic_inc = float(opt.pop(1))
        discrete = bool(opt.pop(1))
        label = opt.pop(1)
        data_file = opt.pop(1)
        cmap_min = 0.0
        cmap_max = 1.0
        if (len(opt) > 1): cmap_min = float(opt.pop(1))
        if (len(opt) > 1): cmap_max = float(opt.pop(1))
        pdata = PlotSimpleData(pfield, data_col, vmin, vmax, data_file,
                               tic_min, tic_max, tic_inc)
        pfield.add_data(pdata, None, show_cbar, discrete, label, None,
                        cmap_min, cmap_max)
    elif (option == "--colour-by-index"):
        pfield.colour_by_index()
    elif (option == "--cm"):
        pfield.add_cell_cm()
    elif (option == "--cbar"):
        show_cbar = True
    elif (option == "--cell-alpha"):
        if (len(opt) != 2):
            print("Usage: --cell-alpha alpha")
            sys.exit(1)
        alpha = float(opt.pop(1))
        pfield.set_cell_alpha(alpha)        
    elif (option == "--edge-alpha"):
        if (len(opt) != 2):
            print("Usage: --edge-alpha alpha")
            sys.exit(1)
        alpha = float(opt.pop(1))
        pfield.set_edge_alpha(alpha)
    elif (option == "--focus"):
        if (len(opt) < 2):
            print("Usage: --focus cell_index [width] [time]")
            sys.exit(1)
        cell_index = int(opt.pop(1))
        width = int(opt.pop(1)) if (len(opt) > 1) else None
        time = int(opt.pop(1)) if (len(opt) > 1) else None
        pfield.focus(cell_index, width, time)
    elif (option == "--time-label"):
        if (len(opt) != 3):
            print("Usage: --time-label Dr dt")
            sys.exit(1)
        Dr = float(opt.pop(1))
        dt = float(opt.pop(1))
        pfield.add_time_label(Dr,dt)
    elif (option == "--deform-angle"):
        if (len(opt) != 2):
            print("Usage: --deform-angle deform_file")
            sys.exit(1)
        deform_file = opt.pop(1)
        pdata = PlotDeformAngle(pfield, deform_file)
        pi = np.pi
        pfield.add_data(pdata, "twilight_shifted", show_cbar, False,
                        r"$\theta$",
                        [[0.0,pi*0.5,pi],["0",r"$\pi/2$",r"$\pi$"]])
    elif (option == "--director"):
        if (len(opt) < 2):
            print("Usage: --director director_file [alpha]")
            sys.exit(1)
        director_file = opt.pop(1)
        alpha = opt.pop(1) if (len(opt) > 1) else 1.0        
        pdirect = PlotDirector(pfield, director_file, alpha)
        pfield.add_plot_comp(pdirect)
    elif (option == "--deform-axis"):
        if (len(opt) < 2):
            print("Usage: --deform-axis deform_file [alpha]")
            sys.exit(1)
        deform_file = opt.pop(1)
        alpha = opt.pop(1) if (len(opt) > 1) else 1.0
        pdeform = PlotDeformAxis(pfield, deform_file, alpha)
        pfield.add_plot_comp(pdeform) 
    elif (option == "--defect"):
        if (len(opt) != 2):
            print("Usage: --defect defect_file")
            sys.exit(1)
        defect_file = opt.pop(1)
        pdefect = PlotDefect(pfield, defect_file)
        pfield.add_plot_comp(pdefect)


pfield.plot(out_file, make_movie, print_to_screen)
