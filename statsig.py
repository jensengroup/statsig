import numpy as np

def rmse(X, Y):
    """
    Root-mean-square error

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

    # Lower = RMSE \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    le = rmse * (1.0 - np.sqrt(1-1.96*np.sqrt(2.0)/np.sqrt(N-1)))

    # Upper =  RMSE \left(  \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} }-1  \right )
    ue = rmse * (np.sqrt(1 + 1.96*np.sqrt(2.0)/np.sqrt(N-1))-1)

    return rmse, le, ue


def mae(X, Y):
    """
    mean absolute error (MAE)

    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats

    Returns:
    mae -- Mean-absolute error between X and Y
    le -- Lower error on the MAE value
    ue -- Upper error on the MAE value
    """

    # L_X =  MAE_X \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    # U_X =  MAE_X \left(  \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} }-1  \right )

    N, = X.shape

    mae = np.abs(X - Y)
    mae = mae.mean()

    le =  mae * (1 - np.sqrt(1 - 1.96*np.sqrt(2)/np.sqrt(N-1) ) )
    ue =  mae * (    np.sqrt(1 + 1.96*np.sqrt(2)/np.sqrt(N-1) ) -1 )

    return mae, le, ue


def me(X, Y):
    """
    mean error (ME)

    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats

    Returns:
    mae -- Mean error between X and Y
    e   -- Upper and Lower error on the ME
    """

    N, = X.shape

    me = X - Y
    me = me.mean()

    # TODO 
    # L_X = U_X =  \frac{1.96 s_N}{\sqrt{N}}
    # where sN is the standard population deviation (e.g. STDEVP in Excel).
    e = 0

    return me, e



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
    lower_error = []
    upper_error = []

    mae_list = []
    mae_lower = []
    mae_upper = []

    for method in methods:
        mdata = data[method]

        # RMSE
        mrmse, mle, mue = rmse(mdata, ref)
        rmse_list.append(mrmse)
        lower_error.append(mle)
        upper_error.append(mue)

        # MAD
        mmae, maele, maeue = mae(mdata, ref)
        mae_list.append(mmae)
        mae_lower.append(maele)
        mae_upper.append(maeue)

        # ME
        # TODO


    print "Method_A   Method_B      RMSE_A   RMSE_B   RMSE_A-RMSE_B  Comp Err  same?"
    ps = "{:10s} "*2 +  "{:8.3f} "*2 + "{:8.3f}" + "{:15.3f}" + "     {:}"

    for i in xrange(nm):
        for j in xrange(i+1, nm):

            m_i = methods[i]
            m_j = methods[j]

            rmse_i = rmse_list[i]
            rmse_j = rmse_list[j]

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
    plt.errorbar(x, rmse_list, yerr=asymmetric_error, fmt='o')

    # change x-axis to method names and rotate the ticks 30 degrees
    plt.xticks(x, methods, rotation=30)

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)

    # Add grid to plot
    plt.grid(True)

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
    plt.savefig('example_mae.png')

