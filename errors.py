import numpy as np

"""

http://proteinsandwavefunctions.blogspot.ch/2016/11/which-method-is-more-accurate-or-errors.html

Comparing two methods

If errorX is some measure of the error, RMSE, MAE, etc, and errorA>errorB then
the difference is statistically significant only if

errorA - errorB > L2A+U2B - 2rABLAUB

where rAB is the Pearson's r value of method A compared to B, not to be
confused with rA which compares A to the reference value.  Conversely, if this
condition is not satisfied then you cannot say that method B is not more
accurate than method A with 95% confidence because the error bars are too
large.

Note also that if there is a high degree of correlation between the predictions
(rAB ~ 1) and the error bars are similar in size LA ~ UB then even small
differences in error could be significant.

Usually one can assume that rAB>0 so if errorA- errorB > L2A+U2B or
errorA-errorB>LA+UB then the difference is statistically significant, but it is
better to evaluate rAB to be sure.

The meaning of 95% confidence

Say you compute errors for some property for 50
molecules using method A (errorA) and B (errorB) and observe that Eq 11 is
true.

Assuming no prior knowledge on the performance of A and B, if you repeat this
process an additional 40 times using all new molecules each time then in 38
cases (38/40 = 0.95) the errors observed for method A will likely be between
errorA-LA and errorA+UA and similarly for method B. For one of the remaining
two cases the error is expected to be larger than this range, while for the
other remaining case it is expected to be smaller. Furthermore, in 39 of the 40
cases errorA is likely larger than errorB, while errorA is likely smaller than
errorB in the remaining case.

"""


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
        print "Not enough data points. Only {} data points given".format(N)
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

    me = X - Y
    me = me.mean()

    # L_X = U_X =  \frac{1.96 s_N}{\sqrt{N}}
    e = 0

    return me, e


def correlation(X, Y):
    """

    """

    r = 0
    p = 0
    r, p

    # L_X = r_X - \frac{e^{2F_-}-1}{e^{2F_-}+1}
    # U_X =  \frac{e^{2F_+}-1}{e^{2F_+}+1} - r_X

    # where
    # F_{\pm} = \frac{1}{2} \ln \frac{1+r_X}{1-r_X} \pm r_{significant}

    lx = 0
    ux = 0
    return r, p, lx, ux


def compare(error_a, error_b, la, lb, ua, ub):
    """

    error_A - error_B > \sqrt {L_A^2 + U_B^2 - 2{r_{AB}}{L_A}{U_B}}

    """

    # error_A - error_B > \sqrt {L_A^2 + U_B^2 - 2{r_{AB}}{L_A}{U_B}}

    return error_a - error_b, np.sqrt()


if __name__ == '__main__':

    X = range(10)
    X = np.array(X)
    Y = X + 5

    print rmse(X, Y)

    print mae(X, Y)

    print me(X, Y)

    print correlation(X, Y)

    print X == Y

