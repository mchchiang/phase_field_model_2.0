# deff_transition.py
# A program to map a transition line in a phase diagram based on simple
# linear interpolation

import sys
import numpy as np
import pandas as pd

args = sys.argv

if (len(args) != 8):
    print("transition.py xcol ycol vcol vstart vthres data_file out_file")
    sys.exit(1)

xcol = int(args.pop(1))
ycol = int(args.pop(1))
vcol = int(args.pop(1))
vstart = float(args.pop(1))
vthres = float(args.pop(1))
data_file = args.pop(1)
out_file = args.pop(1)

# Read the data matrix
data = pd.read_csv(data_file, header=None, delim_whitespace=True)

# Find unique x values
xs = data[xcol].unique()

from_below = vstart < vthres
vprev = None
yprev = None
found = False
ycrit = []

def interpolate(yprev,y,vprev,v,vthres):
    m = (v-vprev)/(y-yprev)
    return y-(v-vthres)/m

for x in xs:
    mat = data[data[xcol] == x].iloc[:,[ycol,vcol]]
    n = mat.shape[0]
    found = False
    if (from_below):
        for i in range(n):
            y = mat.iloc[i,0]
            v = mat.iloc[i,1]
            if (v > vthres):
                if (i == 0):
                    ycrit.append(y)
                else:
                    ycrit.append(interpolate(yprev,y,vprev,v,vthres))
                found = True
                break
            yprev = y
            vprev = v
        if (not found):
            ycrit.append(y)
    else: # from above
        for i in range(n):
            y = mat.iloc[i,0]
            v = mat.iloc[i,1]
            if (v < vthres):
                if (i == 0):
                    ycrit.append(y)
                else:
                    ycrit.append(interpolate(yprev,y,vprev,v,vthres))
                found = True
                break
            yprev = y
            vprev = v
        if (not found):
            ycrit.append(y)
            
ycrit = np.asarray(ycrit)

# Output result
nx = xs.shape[0]
with open(out_file, 'w') as writer:
    for i in range(nx):
        writer.write("{:g} {:g}\n".format(xs[i],ycrit[i]))
