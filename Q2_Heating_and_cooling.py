import numpy as np

# Constants (mind the units!)

psi = 0.929
Tc = 1e4 # K
Z = 0.015
k = 1.38e-16  # erg/K
aB = 2e-13  # cm^3 / s
A = 5e-10
xi = 1e-15


class NewtonRaphsonRootFinder:
    def __init__(
        self,
        func: callable,
        func_kwargs: dict,
        derivative: callable,
        derivative_kwargs: dict,
        gamma: float = 1.0,
    ):
        """
        Implementation of the Newton-Raphson method for root finding.
        
        P.S: I would have used autodiff for the derivative of the function which makes
        it more general but I was lazy to implement autodiff from scratch and we can't use 
        external libraries ;( 
        
        :param func: The function for which we want to find the root.
        :param func_kwargs: A dictionary of keyword arguments to pass to the function.
        :param derivative: The derivative of the function.
        :param derivative_kwargs: A dictionary of keyword arguments to pass to the derivative function.
        :param gamma: A damping factor to control the step size (default is 1.0).
        """
        self._func = func
        self._func_kwargs = func_kwargs
        self._derivative = derivative
        self._derivative_kwargs = derivative_kwargs
        self._gamma = gamma

    @property
    def history(self):
        """
        Returns the history of guesses,
        function values, derivative values, and step sizes
        during the root finding process.
        """
        if hasattr(self, "_history"):
            return self._history
        else:
            print("No history found. Please run estimate_root with data_logging=True to log the history of guesses.")
            return {}

    def estimate_root(
        self,
        guess: float,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        maximum_iterations: int = 100,
        data_logging: bool = False,
        log_every_n_iterations: int = 1,
    ):
        """
        Estimate the root of the function using the Newton-Raphson method.
        The method iteratively updates the guess for the root using the formula:
            new_guess = guess - gamma * f(guess) / f'(guess)
        The iteration continues until either the absolute error or the
        relative error is less than the specified tolerance.
        
        :param guess: Initial guess for the root.
        :param atol: Absolute tolerance for convergence (default is 1e-6).
        :param rtol: Relative tolerance for convergence (default is 1e-6).
        :param maximum_iterations: Maximum number of iterations to perform (default is 100).
        :param data_logging: Whether to log the history of guesses,
        function values, derivative values, and step sizes (default is False).
        :param log_every_n_iterations: Log data every n iterations if data_logging is True (default is 1).
        :return: A tuple containing the estimated root, absolute error, and relative error.
        """
        initial_guess_arr = np.array(guess)

        iteration_count = 0
        self._history = {}
        while True:
            # Break if the maximum number of iterations is reached
            if iteration_count >= maximum_iterations:
                print("Maximum number of iterations reached without convergence.")
                break
            iteration_count += 1

            new_guess_arr: np.ndarray = (
                initial_guess_arr - self._step(initial_guess_arr, self._gamma)
            )

            # Calculating the difference between each guess (absolute error)
            aerr = (np.abs(new_guess_arr - initial_guess_arr)).max()

            # Calculating the relative error
            rerr = aerr / np.abs(new_guess_arr).max()

            # Change the initial guess to the current guess
            initial_guess_arr = new_guess_arr

            if data_logging:
                if iteration_count % log_every_n_iterations == 0:
                    self._history = self._logging(initial_guess_arr, history=self._history, itteration_count=iteration_count)

            # Break if either the absolute error or the
            # relative error is less than the threshold
            if self._check_convergence(aerr, rerr, atol, rtol, iteration_count):
                break
        return initial_guess_arr, aerr, rerr
    
    def plot_history(self):
        """
        Plot the history of guesses, function values,
        derivative values, and step sizes during the root finding process.
        """
        import matplotlib.pyplot as plt

        self._history = self.history

        iterations = self._history["iteration"]
        guesses = self._history["guess"]
        function_values = self._history["function_value"]
        derivative_values = self._history["derivative_value"]
        step_sizes = self._history["step_size"]

        fig, axs = plt.subplots(2, 2, figsize=(10, 20))

        axs[0, 0].plot(iterations, guesses, marker='o')
        axs[0, 0].set_title('Guess vs Iteration')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Guess')

        axs[0, 1].plot(iterations, function_values, marker='o')
        axs[0, 1].set_title('Function Value at Guess vs Iteration')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Function Value at Guess')

        axs[1, 0].plot(iterations, derivative_values, marker='o')
        axs[1, 0].set_title('Derivative Value at Guess vs Iteration')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Derivative Value at Guess')

        axs[1, 1].plot(iterations, step_sizes, marker='o')
        axs[1, 1].set_title('Step Size vs Iteration')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Step Size')

        plt.show()

    def _step(self, guess_arr: np.ndarray, gamma: float) -> np.ndarray:
        """
        Calculate the step size for the Newton-Raphson method.
        The step size is given by the function value at the current guess
        divided by the derivative value at the current guess, multiplied by the damping factor gamma.
            step = gamma * f(guess) / f'(guess)

        :param guess_arr: The current guess for the root as a numpy array.
        :param gamma: The damping factor for the step size.
        :return: The step size to update the guess.
        """
        return gamma * self._func(guess_arr, **self._func_kwargs) / self._derivative(guess_arr, **self._derivative_kwargs)
    
    def _check_convergence(
            self,
            aerr: float,
            rerr: float,
            atol: float,
            rtol: float,
            iteration_count: int
        ) -> bool:
        """
        Check if the convergence criteria are met based on the absolute error and relative error.
        
        :param aerr: The absolute error of the current guess.
        :param rerr: The relative error of the current guess.
        :param atol: The absolute tolerance for convergence.
        :param rtol: The relative tolerance for convergence.
        :param iteration_count: The current iteration count.
        :return: True if convergence is successful, False otherwise.
        """
        if aerr < atol or rerr < rtol:
            print(f"Convergence successful at iteration {iteration_count}!")
            return True
        return False

    def _logging(self, guess_arr: np.ndarray, history: dict = {}, itteration_count: int = 0):
        func_value = self._func(guess_arr, **self._func_kwargs)
        derivative_falue = self._derivative(guess_arr, **self._derivative_kwargs)
        step_size = self._step(guess_arr)
        
        # Initialize the history dictionary if it doesn't exist
        if "iteration" not in history:
            history["iteration"] = []
        if "guess" not in history:
            history["guess"] = []
        if "function_value" not in history:
            history["function_value"] = []
        if "derivative_value" not in history:
            history["derivative_value"] = []
        if "step_size" not in history:
            history["step_size"] = []

        # Append the current iteration data to the history
        history["iteration"].append(itteration_count)
        history["guess"].append(guess_arr)
        history["function_value"].append(func_value)
        history["derivative_value"].append(derivative_falue)
        history["step_size"].append(step_size)

        print("-----------------------------")
        print(f"Iteration {itteration_count}:")
        print(f"Current guess: {guess_arr}")
        print(f"Function value at current guess: {func_value}")
        print(f"Derivative value at current guess: {derivative_falue}")
        print(f"Step size: {step_size}")
        print("-----------------------------")
        return history


# There's no need for nH nor ne as they cancel out
def equilibrium1(T, Z, Tc, psi):
    return psi * Tc * k - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T * k

def equilibrium1_deriv(T, Z):
    # (d/dT) (psi * Tc * k - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T * k) =
    # (d/dT) (- (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T * k)
    # Let f(T) = - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z)))
    # Then d/dT (f(T) * T) = f'(T) * T + f(T)
    f_T = (-0.6424 + 0.0416 * np.log(T / (1e4 * Z * Z))) * k
    f_T_deriv = 0.0416 / T * k
    return f_T_deriv * T + f_T

def equilibrium2(T, Z, Tc, psi, nH, A, xi, aB):
    return (
        (
            psi * Tc
            - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T
            - 0.54 * (T / 1e4) ** 0.37 * T
        )
        * k
        * nH
        * aB
        + A * xi
        + 8.9e-26 * (T / 1e4)
    )

def equilibrium2_deriv(T, Z, nH, aB):
    # (d/dT) (- (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T)
    # Let f(T) = - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z)))
    # Then d/dT (f(T) * T) = f'(T) * T + f(T)
    f_T = -0.6424 + 0.0416 * np.log(T / (1e4 * Z * Z))
    f_T_deriv = 0.0416 / T
    f_T_times_T_deriv = f_T_deriv * T + f_T

    # (d/dT) (- 0.54 * (T / 1e4) ** 0.37 * T)
    # Let g(T) = - 0.54 * (T / 1e4) ** 0.37
    # Then d/dT (g(T) * T) = g'(T) * T + g(T)
    g_T = -0.54 * (T / 1e4) ** 0.37
    g_T_deriv = -0.54 * 0.37 * (T / 1e4) ** -0.63 / 1e4
    g_T_times_T_deriv = g_T_deriv * T + g_T

    last_term_deriv = 8.9e-26 / 1e4
    
    # Full derivative
    # (d/dT)(
    #     psi * Tc
    #     - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T
    #     - 0.54 * (T / 1e4) ** 0.37 * T
    # )
    # * k
    # * nH
    # * aB
    # + A * xi
    # + 8.9e-26 * (T / 1e4) = d/dT(
    #     psi * Tc
    #     + f(T) * T
    #     + g(T) * T
    # ) * k * nH * aB + A * xi + last_term)
    # Thus, the derivative is:
    # (
    #     d/dT (f(T) * T)
    #     + d/dT (g(T) * T)
    # ) * k * nH * aB + d/dT (last_term)
    return (f_T_times_T_deriv + g_T_times_T_deriv) * k * nH * aB + last_term_deriv



#### root finder ####


def root_finder(
    func: callable,
    derivative: callable,
    initial_guess: float,
    func_kwargs: dict,
    derivative_kwargs: dict,
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
    plot_history: bool = False,
) -> tuple[float, float, float]:
    """
    Find a root of a function

    Parameters
    ----------
    func : callable
        Function to find root of
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        Teh default is 100

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error
    rerr : float
        Relative error
    """
    nr_root_finder = NewtonRaphsonRootFinder(
        func=func,
        func_kwargs=func_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        gamma=1.0 # Controls the step size, can be tuned for better convergence.
    )

    root, aerr, rerr = nr_root_finder.estimate_root(
        guess=[initial_guess],
        atol=atol,
        rtol=rtol,
        maximum_iterations=max_iters,
        data_logging=True,
        log_every_n_iterations=1,
    )

    if plot_history:
        nr_root_finder.plot_history()
    return root.item(), aerr.item(), rerr.item()


def main():

    # Initial bracket
    bracket = (1, 1e7)

    root, aerr, rerr = root_finder(
        func=equilibrium1,
        derivative=equilibrium1_deriv,
        initial_guess=(bracket[0] + bracket[1]) / 2, # Starting with the midpoint of the bracket as the initial guess
        func_kwargs={"Z": Z, "Tc": Tc, "psi": psi},
        derivative_kwargs={"Z": Z},
        bracket=bracket,
        atol=1e-6,
        rtol=1e-6,
        max_iters=100,
        plot_history=True,
    )  # replace with your root finder
    

    with open("Calculations/equilibrium_temp_simple.txt", "w") as f:
        f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e}")
    #### 2b ####

    # Initial bracket
    bracket = (1, 1e15)

    for nH in [1e-4, 1, 1e4]:

        root, aerr, rerr = root_finder(
            func=equilibrium2,
            derivative=equilibrium2_deriv,
            initial_guess=(bracket[0] + bracket[1]) / 2, # Starting with the midpoint of the bracket as the initial guess
            func_kwargs={"Z": Z, "Tc": Tc, "psi": psi, "nH": nH, "A": A, "xi": xi, "aB": aB},
            derivative_kwargs={"Z": Z, "nH": nH, "aB": aB},
            bracket=bracket,
            atol=1e-10,
            rtol=1e-10,
            max_iters=100,
            plot_history=True,
        )
        if nH == 1e-4:
            with open("Calculations/equilibrium_low_density.txt", "w") as f:
                f.write(f"{root:.12g}")
        elif nH == 1:
            with open("Calculations/equilibrium_mid_density.txt", "w") as f:
                f.write(f"{root:.12g}")
        elif nH == 1e4:
            with open("Calculations/equilibrium_high_density.txt", "w") as f:
                f.write(f"{root:.12g}")


if __name__ == "__main__":
    main()
