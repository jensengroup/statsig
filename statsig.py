
import errors as e

import sys
import numpy as np

import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    exit("usage: python example.py example_input.csv")

filename = sys.argv[1]
f = open(filename, "r")
data = np.genfromtxt(f, delimiter=',', names=True)
f.close()

ref = data['REF']
n = len(ref)

methods = data.dtype.names
methods = methods[1:]
nm = len(methods)

rmse = []
lower_error = []
upper_error = []

for method in methods:
    mdata = data[method]
    mrmse, mle, mue = e.rmse(mdata, ref)
    rmse.append(mrmse)
    lower_error.append(mle)
    upper_error.append(mue)


print "Method_A   Method_B      RMSE_A   RMSE_B   A-B      Comp Err   same?"
ps = "{:10s} "*2 +  "{:8.3f} "*4 + "     {:}"


for i in xrange(nm):
    for j in xrange(i+1, nm):

        m_i = methods[i]
        m_j = methods[j]

        rmse_i = rmse[i]
        rmse_j = rmse[j]

        r_ij = np.corrcoef(data[m_i], data[m_j])[0][1]

        if rmse_i > rmse_j:
            lower = lower_error[i]
            upper = upper_error[j]
        else:
            lower = lower_error[j]
            upper = upper_error[i]

        comp_error = np.sqrt(upper**2 + lower**2 - 2.0*r_ij*upper*lower)
        significance = abs(rmse_i - rmse_j) < comp_error

        print ps.format(m_i, m_j, rmse_i, rmse_j, rmse_i-rmse_j, comp_error, significance)


# Create x-axis
x = range(len(methods))

# Errorbar (upper and lower)
asymmetric_error = [lower_error, upper_error]

# Add errorbar for RMSE
plt.errorbar(x, rmse, yerr=asymmetric_error, fmt='o')

# change x-axis to method names and rotate the ticks 30 degrees
plt.xticks(x, methods, rotation=30)

# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)

# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)

# Add grid to plot
plt.grid(True)

# Save plot to PNG format
plt.savefig('example_plot.png')

