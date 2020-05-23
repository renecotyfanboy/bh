"""
Numba boosted functions for black hole imaging
"""

import numpy as np
from numba import jit_module
from numba import vectorize, float64

c_i = np.array(
    [
        0,
        1 / 18,
        1 / 12,
        1 / 8,
        5 / 16,
        3 / 8,
        59 / 400,
        93 / 200,
        5490023248 / 9719169821,
        13 / 20,
        1201146811 / 1299019798,
        1,
        1,
    ]
)
a_i_j = np.transpose(
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / 48, 1 / 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / 32, 0, 3 / 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5 / 16, 0, -75 / 64, 75 / 64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3 / 80, 0, 0, 3 / 16, 3 / 20, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                29443841 / 614563906,
                0,
                0,
                77736538 / 692538347,
                -28693883 / 1125000000,
                23124283 / 1800000000,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                16016141 / 946692911,
                0,
                0,
                61564180 / 158732637,
                22789713 / 633445777,
                545815736 / 2771057229,
                -180193667 / 1043307555,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                39632708 / 573591083,
                0,
                0,
                -433636366 / 683701615,
                -421739975 / 2616292301,
                100302831 / 723423059,
                790204164 / 839813087,
                800635310 / 3783071287,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                246121993 / 1340847787,
                0,
                0,
                -37695042795 / 15268766246,
                -309121744 / 1061227803,
                -12992083 / 490766935,
                6005943493 / 2108947869,
                393006217 / 1396673457,
                123872331 / 1001029789,
                0,
                0,
                0,
                0,
            ],
            [
                -1028468189 / 846180014,
                0,
                0,
                8478235783 / 508512852,
                1311729495 / 1432422823,
                -10304129995 / 1701304382,
                -48777925059 / 3047939560,
                15336726248 / 1032824649,
                -45442868181 / 3398467696,
                3065993473 / 597172653,
                0,
                0,
                0,
            ],
            [
                185892177 / 718116043,
                0,
                0,
                -3185094517 / 667107341,
                -477755414 / 1098053517,
                -703635378 / 230739211,
                5731566787 / 1027545527,
                5232866602 / 850066563,
                -4093664535 / 808688257,
                3962137247 / 1805957418,
                65686358 / 487910083,
                0,
                0,
            ],
            [
                403863854 / 491063109,
                0,
                0,
                -5068492393 / 434740067,
                -411421997 / 543043805,
                652783627 / 914296604,
                11173962825 / 925320556,
                -13158990841 / 6184727034,
                3936647629 / 1978049680,
                -160528059 / 685178525,
                248638103 / 1413531060,
                0,
                0,
            ],
        ]
    )
)
b_8 = np.array(
    [
        14005451 / 335480064,
        0,
        0,
        0,
        0,
        -59238493 / 1068277825,
        181606767 / 758867731,
        561292985 / 797845732,
        -1041891430 / 1371343529,
        760417239 / 1151165299,
        118820643 / 751138087,
        -528747749 / 2220607170,
        1 / 4,
    ]
)
b_7 = np.array(
    [
        13451932 / 455176623,
        0,
        0,
        0,
        0,
        -808719846 / 976000145,
        1757004468 / 5645159321,
        656045339 / 265891186,
        -3867574721 / 1518517206,
        465885868 / 322736535,
        53011238 / 667516719,
        2 / 45,
        0,
    ]
)


def theta_d_p(i, alpha):

    return np.arccos(
        -np.sin(alpha) * np.cos(i) / np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(i) ** 2)
    )


def theta_d_s(i, alpha):

    return (
        np.arccos(
            -np.sin(alpha)
            * np.cos(i)
            / np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(i) ** 2)
        )
        + np.pi
    )


def f(Z):

    Z_dot = np.zeros((2,))
    Z_dot[0] = Z[1]
    Z_dot[1] = 3 / 2 * Z[0] ** 2 - Z[0]

    return Z_dot


def calc_r_p(i, alpha, b):
    """
    Unvectorized computation for the primary image radius

    Parameters
    ----------
    i : float
        Observer angle in radian
    alpha : float
        Angle of the pixel
    b : float
        Radius of the pixel

    Returns
    -------
    r_p : float
        Disk radius observed on the pixel
    """

    u_c = 3 * 3 ** 0.5 / 2 / b
    Y0 = np.zeros((2,))
    Y0[1] = u_c * 2 / (3 * 3 ** 0.5)
    theta_f = theta_d_p(i, alpha)
    Y = ode87(Y0, theta_f)

    return 1 / Y[0]


def calc_r_s(i, alpha, b):

    u_c = 3 * 3 ** 0.5 / 2 / b
    Y0 = np.zeros((2,))
    Y0[1] = u_c * 2 / (3 * 3 ** 0.5)
    theta_f = theta_d_s(i, alpha)
    Y = ode87(Y0, theta_f)

    return 1 / Y[0]


def flux(r):

    return (
        r ** (-5 / 2)
        / (r - 3 / 2)
        * (
            r ** 0.5
            - 3 ** 0.5
            + np.sqrt(3 / 8)
            * np.log(
                (np.sqrt(2) - 1)
                / (np.sqrt(2) + 1)
                * (np.sqrt(r) + np.sqrt(3 / 2))
                / (np.sqrt(r) - np.sqrt(3 / 2))
            )
        )
    )


def redshift(r, i, alpha, b):

    return (
        1
        / np.sqrt(1 - 3 / (2 * r))
        * (
            1
            + np.cos(i)
            * np.cos(alpha)
            * b
            / (3 * 3 ** 0.5 / 2)
            * (3 / (2 * r)) ** (3 / 2)
        )
    )


def ode87(Y, theta_max, n_points=10):

    step_pow = 1 / 8
    h = theta_max / n_points
    n_reject = 0
    updated = False
    tol = 1e-12

    # Minimal step size
    h_min = 16 * np.spacing(1)
    h_max = h

    c_time = 0

    while 0 < (theta_max - c_time):

        # If next step bring the solution beyond the final time
        if ((c_time + h) - theta_max) > 0:
            h = theta_max - c_time

        updated = False

        while not updated:

            c_Y = np.copy(Y)
            p_n_i = np.zeros((13, 2))

            for i in range(13):

                Y_n_i = c_Y + h * np.dot(a_i_j[:, i], p_n_i)
                p_n_i[i, :] = f(Y_n_i)

            Y_8 = c_Y + h * np.dot(b_8, p_n_i)
            Y_7 = c_Y + h * np.dot(b_7, p_n_i)
            error_step = np.sqrt(np.dot(Y_8 - Y_7, Y_8 - Y_7))
            tau = tol * max(np.max(np.abs(c_Y)), 1.0)

            # update if solution is precise enough
            if error_step < tau:

                if Y_8[0] < 0 or Y_8[1] > 1e20:
                    return Y

                Y = np.copy(Y_8)
                c_time += h
                updated = True

            else:

                n_reject += 1

            # STEP CONTROL
            if error_step == 0.0:
                error_step = 10 * np.spacing(1)

            h = min(h_max, abs(0.9 * h * (tau / error_step) ** step_pow))
            h = max(h_min, abs(h))

    return Y


jit_module(nopython=True, error_model="numpy",cache=True)


@vectorize([float64(float64, float64, float64)], target="parallel",cache=True)
def rp_map(i, alpha, b):

    R_p = calc_r_p(i, alpha, b)

    return R_p

@vectorize([float64( float64, float64, float64)],
           target="parallel",
    cache=True)
def rs_map(i, alpha, b):

    R_s = calc_r_s(i, alpha, b)

    return R_s

@vectorize([float64(float64, float64, float64)], target="parallel",cache=True)
def rn_map(i, alpha, b):

    R_n = b / np.sin(theta_d_p(i, alpha))

    return R_n

@vectorize([float64(float64, float64, float64, float64, float64)], 
    target="parallel",
    cache=True)
def r_map(r_min, r_max, i, alpha, b):

    R_p = calc_r_p(i, alpha, b)

    if R_p > r_min and R_p < r_max:

        return R_p

    R_s = calc_r_s(i, alpha, b)

    if R_s > r_min and R_s < r_max:

        return R_s

    return 0

@vectorize(
    [float64(float64, float64, float64, float64, float64, float64)],
    target="parallel",
    cache=True
)
def img_value(r, r_min, r_max, inclinaison, alpha, b):

    if r > r_min and r < r_max:

        return flux(r) / (redshift(r, inclinaison, alpha, b)) ** 4

    else:

        return 0


@vectorize(
    [float64(float64, float64, float64, float64, float64, float64)], 
    target="parallel",
    cache=True
)
def red_value(r, r_min, r_max, inclinaison, alpha, b):

    if r > r_min and r < r_max:

        return redshift(r, inclinaison, alpha, b) - 1

    else:

        return np.nan
