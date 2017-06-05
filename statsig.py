import numpy as np

check = "rmse"
#check = "pearson"

def correl(X,Y):
    N, = X.shape

    if N < 9:
        print "not enough points. {} datapoints given. at least 9 is required".format(n)
        return

    r = np.corrcoef(X,Y)[0][1]
    r_sig = 1.96/np.sqrt(n-2+1.96**2)
    F_plus = 0.5*np.log((1+r)/(1-r))+r_sig
    F_minus = 0.5*np.log((1+r)/(1-r))-r_sig
    le = r - (np.exp(2*F_minus)-1)/(np.exp(2*F_minus)+1)
    ue = (np.exp(2*F_plus)-1)/(np.exp(2*F_plus)+1) - r

    return r, le, ue


def rmse(X, Y):
    """
    Root-Mean-Square Error

    Lower Error = RMSE \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    Upper Error = RMSE \left(    \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} } - 1 \right )

    This only works for N >= 8.6832, otherwise the lower error will be
    imaginary.

    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats

    Returns:
    rmse -- Root-mean-square error between X and Y
    le -- Lower error on the RMSE value
    ue -- Upper error on the RMSE value
    """

    N, = X.shape

    if N < 9:
        print "Not enough points. {} datapoints given. At least 9 is required".format(N)
        return

    diff = X - Y
    diff = diff**2
    rmse = np.sqrt(diff.mean())

    le = rmse * (1.0 - np.sqrt(1-1.96*np.sqrt(2.0)/np.sqrt(N-1)))
    ue = rmse * (np.sqrt(1 + 1.96*np.sqrt(2.0)/np.sqrt(N-1))-1)

    return rmse, le, ue


def mae(X, Y):
    """
    Mean Absolute Error (MAE)

    Lower Error =  MAE_X \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    Upper Error =  MAE_X \left(  \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} }-1  \right )

    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats

    Returns:
    mae -- Mean-absolute error between X and Y
    le -- Lower error on the MAE value
    ue -- Upper error on the MAE value
    """

    N, = X.shape

    mae = np.abs(X - Y)
    mae = mae.mean()

    le =  mae * (1 - np.sqrt(1 - 1.96*np.sqrt(2)/np.sqrt(N-1) ) )
    ue =  mae * (    np.sqrt(1 + 1.96*np.sqrt(2)/np.sqrt(N-1) ) -1 )

    return mae, le, ue


def me(X, Y):
    """
    mean error (ME)

    L_X = U_X =  \frac{1.96 s_N}{\sqrt{N}}
    where sN is the standard population deviation (e.g. STDEVP in Excel).

    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats

    Returns:
    mae -- Mean error between X and Y
    e   -- Upper and Lower error on the ME
    """

    N, = X.shape

    error = X - Y
    me = error.mean()

    s_N = stdevp(error, me, N)
    e = 1.96*s_N/np.sqrt(N)

    return me, e


def stdevp(X, X_hat, N):
    """
    Parameters:
    X -- One dimensional Numpy array of floats
    X_hat -- Float
    N -- Integer

    Returns:

    Calculates standard deviation based on the entire population given as
    arguments. The standard deviation is a measure of how widely values are
    dispersed from the average value (the mean).
    """
    return np.sqrt(np.sum((X-X_hat)**2)/N)


if __name__ == '__main__':

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

    rmse_list = []
    rmse_lower = []
    rmse_upper = []

    mae_list = []
    mae_lower = []
    mae_upper = []

    me_list = []
    me_lower = []
    me_upper = []

    r_list = []
    r_lower = []
    r_upper = []

    for method in methods:
        mdata = data[method]

        # RMSE
        mrmse, mle, mue = rmse(mdata, ref)
        rmse_list.append(mrmse)
        rmse_lower.append(mle)
        rmse_upper.append(mue)

        # MAD
        mmae, maele, maeue = mae(mdata, ref)
        mae_list.append(mmae)
        mae_lower.append(maele)
        mae_upper.append(maeue)

        # ME
        mme, mmee = me(mdata, ref)
        me_list.append(mme)
        me_lower.append(mmee)
        me_upper.append(mmee)

        # r
        r, rle, rue = correl(mdata, ref)
        r_list.append(r)
        r_lower.append(rle)
        r_upper.append(rue)


    print "Method_A   Method_B      RMSE_A   RMSE_B   RMSE_A-RMSE_B  Comp Err  same?"
    ps = "{:10s} "*2 +  "{:8.3f} "*2 + "{:8.3f}" + "{:15.3f}" + "     {:}"


    if check == "pearson":
        measure = r_list
        upper_error = r_upper
        lower_error = r_lower
    else:
        measure = rmse_list
        upper_error = rmse_upper
        lower_error = rmse_lower

    for i in xrange(nm):
        for j in xrange(i+1, nm):

            m_i = methods[i]
            m_j = methods[j]

            rmse_i = measure[i]
            rmse_j = measure[j]

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

    print 
    print "\\begin{table}[]"
    print "\centering"
    print "\caption{}"
    print "\label{}"
    centering = "\\begin{tabular}{l"
    for i in range(nm):
        centering += "c"
    centering +="}"
    print centering
    print "\midrule"

    print "&",
    for i in range(nm-1):
        print methods[i],"&",
    print methods[-1],"\\\\"
    print "\midrule"
#   for i in xrange(nm-1):
#       print '%.1f $\pm$ %.1f/%.1f &'%(rmse_list[i],lower_error[i],rmse_upper[i]),
#   print '%.1f $\pm$ %.1f/%.1f'%(rmse_list[-1],lower_error[-1],rmse_upper[-1])
    print "RMSE &",
    for i in xrange(nm-1):
        print '%.1f &'%(rmse_list[i]),
    print '%.1f \\\\'%(rmse_list[-1])
    print "95 \% conf &",
    for i in xrange(nm-1):
        print '$\pm$ %.1f/%.1f &'%(rmse_upper[i],lower_error[i]),
    print '$\pm$ %.1f/%.1f \\\\'%(rmse_upper[-1],lower_error[-1])
    print "ME &",
    for i in xrange(nm-1):
        print '%.1f $\pm$ %.1f &'%(me_list[i],me_upper[i]),
    print '%.1f $\pm$ %.1f \\\\'%(me_list[-1],me_upper[-1])
    print "$r$ &",
    for i in xrange(nm-1):
        print '%.2f &'%(r_list[i]),
    print '%.2f \\\\'%(r_list[-1])
    print "95 \% conf &",
    for i in xrange(nm-1):
        print '$\pm$ %.2f/%.2f &'%(r_upper[i],r_lower[i]),
    print '$\pm$ %.2f/%.2f \\\\'%(r_upper[-1],r_lower[-1])
    print "\midrule"
    print "\end{tabular}"
    print "\end{table}"
    



    # Create x-axis
    x = range(len(methods))

    # Errorbar (upper and lower)
    asymmetric_error = [rmse_lower, rmse_upper]

    # Add errorbar for RMSE
    plt.errorbar(x, rmse_list, yerr=asymmetric_error, fmt='o')

    # change x-axis to method names and rotate the ticks 30 degrees
    plt.xticks(x, methods, rotation=30)

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)

    # Add grid to plot
    plt.grid(True)

    # Set plot title
    plt.title('Root-mean-sqaure error')

    # Save plot to PNG format
    plt.savefig('example_rmsd.png')

    # Clear figure
    plt.clf()

    # MAE plot
    asymmetric_error = [mae_lower, mae_upper]
    plt.errorbar(x, mae_list, yerr=asymmetric_error, fmt='o')
    plt.xticks(x, methods, rotation=30)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.title('Mean Absolute Error')
    plt.savefig('example_mae.png')

    # Clear figure
    plt.clf()

    # ME plot
    asymmetric_error = [me_lower, me_upper]
    plt.errorbar(x, me_list, yerr=asymmetric_error, fmt='o')
    plt.xticks(x, methods, rotation=30)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.title('Mean Error')
    plt.savefig('example_me.png')
