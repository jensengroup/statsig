# usage "python statsig.py xxx.csv"

import sys
import numpy as np

filename = sys.argv[1]
file = open(filename, "r")

data = np.genfromtxt(file, delimiter=',', names=True)

ref = data['REF']
n = len(ref)
l_factor = 1-np.sqrt(1-1.96*np.sqrt(2)/np.sqrt(n-1))
u_factor = np.sqrt(1+1.96*np.sqrt(2)/np.sqrt(n-1))-1

#print data.dtype.names

print "Method_1", "Method_2", "RMSE_1", "RMSE_2", "RMSE_1-RMSE_2", "Composite Error", "Same/Different" 

for i,name_i in enumerate(data.dtype.names):
    if i == 0: continue
    method_i = data[name_i]
    rmse_i = np.sqrt(np.mean((method_i-ref)**2))
    for j,name_j in enumerate(data.dtype.names):
        if j == 0: continue
        method_j = data[name_j]
        rmse_j = np.sqrt(np.mean((method_j-ref)**2))
        if i < j:
            r_ij = np.corrcoef(method_i,method_j)[0][1]
            if rmse_i > rmse_j:
               lower = rmse_i*l_factor
               upper = rmse_j*u_factor
            else:
               lower = rmse_j*l_factor
               upper = rmse_i*u_factor
            comp_error = np.sqrt(upper**2 + lower**2 - 2.0*r_ij*upper*lower)
            if abs(rmse_i - rmse_j) > comp_error: significance = "different"
            if abs(rmse_i - rmse_j) < comp_error: significance = "same"
            print name_i,name_j,rmse_i,rmse_j,rmse_i - rmse_j,comp_error,significance
 
