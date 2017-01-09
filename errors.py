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
        print "Not enough points. Only {} points given, at least 9 is required".format(N)
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

    mae = np.abs(X - Y)
    mae = mae.mean()

    # L_X =  MAE_X \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    le = 0

    # U_X =  MAE_X \left(  \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} }-1  \right )
    ue = 0

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

    X = range(10)
    X = np.array(X)
    Y = X + 5

    print rmse(X, Y)

    print mae(X, Y)

    print me(X, Y)

