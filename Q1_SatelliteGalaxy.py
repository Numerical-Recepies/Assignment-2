# imports
import numpy as np
import matplotlib.pyplot as plt


def n(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float | ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return 0  # insert your function


##### Integrator block #####


# Below we provide a template for romberg integration
# You can implement this or use another integration method based on some form of Richardson extrapolation
def romberg_integrator(
    func: callable, bounds: tuple, order: int = 5, err: bool = False, args: tuple = ()
) -> float | tuple[float, float]:
    """
    Romberg integration method

    Parameters
    ----------
    func : callable
        Function to integrate.
    bounds : tuple
        Lower- and upper bound for integration.
    order : int, optional
        Order of the integration.
        The default is 5.
    err : bool, optional
        Whether to retun first error estimate.
        The default is False.
    args : tuple, optional
        Arguments to be passed to func.
        The default is ().

    Returns
    -------
    float
        Value of the integral. If err=True, returns the tuple
        (value, err), with err a first estimate of the (relative)
        error.
    """
    # TODO: implement Romberg integration method

    if err:
        return 0.0, 0.0  # (value, error)
    return 0.0


#### Sampler block ####


def sampler(
    dist: callable,
    min: float,
    max: float,
    Nsamples: int,
    args: tuple = (),
) -> np.ndarray:
    """
    Sample a distribution using sampling method of your choice

    Parameters
    dist : callable
        Distribution to sample
    min :
        Minimum value for sampling
    max : float
        Maximum value for sampling
    Nsamples : int
        Number of samples
    args : tuple, optional
        Arguments of the distribution to sample, passed as args to dist

    Returns
    -------
    sample: ndarray
        Values sampled from dist, shape (Nsamples,)
    """

    return np.zeros(Nsamples)


#### Sorting block ####


def sort_array(
    arr: np.ndarray,
    inplace: bool = False,
) -> np.ndarray:
    """
    Sort a 1D array using a sorting algorithm of your choice

    Parameters
    ----------
    arr : ndarray
        Input array to be sorted
    inplace : bool, optional
        If True, sort the array in-place
        If False, return a sorted copy

    Returns
    -------
    sorted_arr : ndarray
        Sorted array (same shape as arr)

    """
    if inplace:
        sorted_arr = arr
    else:
        sorted_arr = arr.copy()

    # TODO: sort sorted_arr in-place here

    return sorted_arr


def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    # TODO: Implement your choice function here, e.g. by using Fisher-Yates shuffling
    return arr[:size].copy()


##### Derivative block #####


def dn_dx(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Analytical derivative of number density provide

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    # TODO: Write the analytical derivative of n(x) here
    return 0.0


def finite_difference(
    function: callable, x: float | np.ndarray, h: float
) -> float | np.ndarray:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float | ndarray
        Derivative at x
    """
    # TODO: Implement finite difference method
    return 0.0


def compute_derivative(
    function: callable,
    x: float | np.ndarray,
    h_init: float,
    # For Ridders use parameters below:
    # d: float, # Factor by which to decrease h_init every iteration
    # eps: float, # Relative error
    # max_iters: int = 10, 3 Maximum number of iterations before exiting
) -> float | np.ndarray:
    """
    Function to compute derivative

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference

    Returns
    -------
    df : float | ndarray
        Derivative at x
    """
    # TODO: Implement derivative
    return 0.0


def main():

    # Values from the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (0, 5)
    xmin, xmax = 10**-4, 5
    N_generate = 10000
    xx = np.linspace(xmin, xmax, N_generate)

    integrand = lambda x, a, b, c: 0.0  # insert the correct function
    integral, err = romberg_integrator(
        integrand, bounds, order=2, args=(a, b, c), err=True
    )

    # Normalisation
    A = 1.0  # to be computed
    with open("Calculations/satellite_A.txt", "w") as f:
        f.write(f"{A:.12g}\n")
    integrand = lambda x, a, b, c: 0.0  # replace by the correct function
    integrated_Nsat = (
        0.0  # replace by the correct integral, e.g. by calling your integrator
    )

    p_of_x = (
        lambda x: 0.0
    )  # replace by the normalised distribution of satellite galaxies as a function of x

    # Numerically determine maximum to normalize p(x) for sampling
    pmax = 0.0  # replace by taking the maximum value of p_of_x

    p_of_x_norm = lambda x: 0.0  # replace by the normalised distribution
    random_samples = np.zeros(
        N_generate
    )  # replace by your sampler(p_of_x_norm, min=xmin, max=xmax, Nsamples=N_generate, args=())

    edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 21)

    hist = np.histogram(
        xmin + np.sort(np.random.rand(N_generate)) * (xmax - xmin), bins=edges
    )[
        0
    ]  # replace!
    hist_scaled = (
        1e-3 * hist
    )  # replace; this is NOT what you should be plotting, this is just a random example to get a plot with reasonable y values (think about how you *should* scale hist)

    fig = plt.figure()
    relative_radius = edges.copy()  # replace!
    analytical_function = edges.copy()  # replace

    fig1b, ax = plt.subplots()
    ax.stairs(
        hist_scaled, edges=edges, fill=True, label="Satellite galaxies"
    )  # just an example line, correct this!
    plt.plot(
        relative_radius, analytical_function, "r-", label="Analytical solution"
    )  # correct this according to the exercise!
    ax.set(
        xlim=(xmin, xmax),
        ylim=(10 ** (-3), 10),  # you may or may not need to change ylim
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("Plots/my_solution_1b.png", dpi=600)

    # Cumulative plot of the chosen galaxies (1c)
    chosen = xmin + np.sort(np.random.rand(Nsat)) * (xmax - xmin)  # replace!
    fig1c, ax = plt.subplots()
    ax.plot(chosen, np.arange(100))
    ax.set(
        xscale="log",
        xlabel="Relative radius",
        ylabel="Cumulative number of galaxies",
        xlim=(xmin, xmax),
        ylim=(0, 100),
    )
    plt.savefig("Plots/my_solution_1c.png", dpi=600)

    x_to_eval = 1
    func_to_eval = lambda x: n(x, A, Nsat, a, b, c)
    dn_dx_numeric = 0.0  # replace by your derivative, e.g. compute_derivative(func_to_eval, x_to_eval, h_init=0.1)
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)
    with open("Calculations/satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.12g}\n")

    with open("Calculations/satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.12g}\n")


if __name__ == "__main__":
    main()
