# time_average.py

import sys
import math

args = sys.argv

if (len(args) != 8):
    print("Usage: time_average.py time_col value_col",
          "t_start t_end t_inc data_file output_file")
    sys.exit(1)

time_col = int(args.pop(1)) # Time column
value_col = int(args.pop(1)) # Data column
t_start = int(args.pop(1)) # First sampled point
t_end = int(args.pop(1))
t_inc = int(args.pop(1))
data_file = args.pop(1) # Data file name
output_file = args.pop(1) # Output file name

# Read data file and store average
avg = 0.0
avgSq = 0.0
n = 0

with open(data_file, "r") as f:
    for line in f:
        # Ignore comments
        if (line.startswith("#")):
            continue
        data = line.strip().split()
        
        # Ignore any empty lines
        if (data == []):
            continue
        
        time = int(data[time_col])
        
        # Ignore lines before tstart and after tend
        if (time > t_end): break
        if (time < t_start or (time-t_start) % t_inc != 0):
            continue
        n += 1
        value = float(data[value_col])
        avg += value
        avgSq += value*value

n = float(n)
avg /= max(n,1)
avgSq /= max(n,1)

# Compute error (use unbiased estimate for standard deviation)
var = 0.0
sigma = 0.0
error = 0.0

if (n > 1):
    var = n / (n-1) * (avgSq - avg*avg)
    if (var < 0.0):
        print("Negative variance: var = %.5f" % var)
    sigma = math.sqrt(abs(var)) 
    error = sigma / math.sqrt(n)

writer = open(output_file, "w")
output = "{:.5e} {:.5e} {:.5e}\n".format(avg, sigma, error)

writer.write(output)
writer.close()
