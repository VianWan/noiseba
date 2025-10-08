import numpy as np

def rhoesi(vp):
    """Nafe-Drake's correlation for density."""
    nafe_drake_poly = np.poly1d([0.000106, -0.0043, 0.0671, -0.4721, 1.6612, 0.0])
    return 1.471 * vp**0.25
    # return nafe_drake_poly(vp)

def vs2vp(vs):
    # return vs * ((1. - poisson) / (0.5 - poisson))**0.5
    return np.array(vs) * 4