import numpy as np


def rho_from_vp(vp, *, constant=1.8, method="constant"):
    """
    Estimate bulk density (g/cm³) from P-wave velocity (vp, km/s).

    Parameters
    ----------
    vp : array_like
        P-wave velocity in km/s.
    constant : float, optional
        Constant density value when method='constant' (g/cm³).
    method : {'constant', 'nafe-drake', 'gardner', 'brocher'}, optional
        Empirical relation to use.

    Returns
    -------
    rho : ndarray
        Bulk density in g/cm³.

    Raises
    ------
    ValueError
        If an unknown method is given.
    """
    vp = np.asarray(vp, dtype=float)

    if method == "constant":
        return np.full_like(vp, constant)

    elif method == "nafe-drake":
        # Nafe–Drake polynomial (Vp in km/s → ρ in g/cm³)
        # Valid ~1.5 < Vp < 8.5 km/s
        coef = [0.000106, -0.0043, 0.0671, -0.4721, 1.6612, 0.0]
        return np.polyval(coef, vp)

    elif method == "gardner":
        # Gardner et al. 1974: ρ = 0.31 * Vp**0.25  (Vp in km/s → ρ in g/cm³)
        return 0.31 * vp**0.25

    elif method == "brocher":
        # Brocher 2005, simplified least-squares fit
        # Vp in km/s → ρ in g/cm³
        return 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5

    else:
        raise ValueError("Unknown method. Choose from 'constant', 'nafe-drake', 'gardner', 'brocher'.")


def vpvsv(v, mode="s2p", lith="mudrock", *, constant=4.0):
    """
    Empirical Vp↔Vs converter for various lithologies.

    Parameters
    ----------
    v : array_like
        Input velocity in km/s.
        When mode='p2s'  --> Vp (km/s)
        When mode='s2p'  --> Vs (km/s)
    mode : {'p2s', 's2p'}, optional
        Conversion direction.
    lith : str, optional
        Lithology keyword. Built-in choices:
            'mudrock','shale','sandstone','limestone','dolomite',
            'salt','basalt','granite','brocher','global'.
             'custom' - uses the *constant* argument.
    constant : float, optional
        Only used when lith='custom'. Vp = constant * Vs.
        (Typical value 1.73 for a global average, or any user-defined ratio).

    Returns
    -------
    out : ndarray
        Converted velocity in km/s.

    Raises
    ------
    ValueError
        If unknown lithology, illegal mode, or missing constant when required.
    """
    v = np.asarray(v, dtype=float)

    if mode not in {"p2s", "s2p"}:
        raise ValueError("mode must be either 'p2s' or 's2p'.")

    # ------------------------------------------------------------------
    # 1) Custom constant ratio (user-defined, e.g. Vp = 1.8 * Vs)
    # ------------------------------------------------------------------
    if lith == "custom":
        if constant is None:
            raise ValueError("For lith='custom' you must supply a numeric constant (Vp/Vs ratio).")
        if mode == "s2p":
            return constant * v
        else:  # p2s
            return v / constant

    # ------------------------------------------------------------------
    # 2) Built-in linear relations: Vp = a * Vs + b
    # ------------------------------------------------------------------
    linear = {
        "mudrock": (1.16, 1.36),
        "shale": (1.16, 1.36),
        "sandstone": (1.47, 0.39),
        "limestone": (1.76, -0.40),
        "dolomite": (1.81, -0.85),
        "salt": (1.15, 1.50),
        "basalt": (1.64, 0.05),
        "granite": (1.69, -0.05),
        "global": (1.73, 0.0),  # simple 1.73× approximation
    }

    if lith in linear:
        a, b = linear[lith]
        if mode == "s2p":
            return a * v + b
        else:  # p2s
            return (v - b) / a

    # ------------------------------------------------------------------
    # 3) Brocher 2005 quadratic: Vp = a*Vs² + b*Vs + c
    # ------------------------------------------------------------------
    if lith == "brocher":
        a, b, c = 0.9409, 2.0977, -1.4837
        if mode == "s2p":
            return a * v**2 + b * v + c
        else:  # solve quadratic for Vs
            disc = b**2 - 4 * a * (c - v)
            if np.any(disc < 0):
                raise ValueError("Brocher quadratic yields no real Vs for the given Vp.")
            vs1 = (-b + np.sqrt(disc)) / (2 * a)
            vs2 = (-b - np.sqrt(disc)) / (2 * a)
            return np.maximum(vs1, vs2)  # pick physically meaningful root

    # ------------------------------------------------------------------
    # 4) Unknown lithology
    # ------------------------------------------------------------------
    raise ValueError(f"Unknown lithology '{lith}'.")

# Note:
# vs < 1000 m/s, sandstone formula works well
# vs > 1000 m/s, limestone, dolomite, salt,