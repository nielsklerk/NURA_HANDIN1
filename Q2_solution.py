import os
import sys
import timeit

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.size"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20


def load_data():
    """
    Function to load the data from Vandermonde.txt.

    Returns
    ------------
    x (np.ndarray): Array of x data points.

    y (np.ndarray): Array of y data points.
    """
    data = np.genfromtxt(
        os.path.join(sys.path[0], "Vandermonde.txt"), comments="#", dtype=np.float64
    )
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def construct_vandermonde_matrix(x: np.ndarray) -> np.ndarray:
    """
    Construct the Vandermonde matrix V with V[i,j] = x[i]^j.

    Parameters
    ----------
    x : np.ndarray, x-values.

    Returns
    -------
    V : np.ndarray, Vandermonde matrix.
    """
    # Create an exponent array with 2 axes
    j = np.arange(len(x))[None, :]

    return x[:, None] ** j

def LU_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Perform LU decomposition.

    The lower-triangular matrix (L) is stored in the lower part of A (the diagonal elements are assumed =1),
    while the upper-triangular matrix (U) is stored on and above the diagonal of A.

    Parameters
    ----------
    A : np.ndarray
        Matrix to decompose.

    Returns
    -------
    A : np.ndarray
        Decomposed array.
    """

    def swap_rows(A: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Swaps two rows of an array

        Parameters
        ----------
        M: np.ndarray
            Matrix of which rows need to be swapped
        i: int
            Row index 1
        j: int
            Row index 2   

        Returns
        -------
        A : np.ndarray
            Matrix with rows i and j swapped

        """
        temp = A[i].copy()
        A[i] = A[j].copy()
        A[j] = temp

        return A

    # Find the largest value in each row and calculate their inverse
    biggest_in_row = np.max(np.abs(A), axis=1)
    if 0 in biggest_in_row:
        raise ValueError('This matrix is singular')
    inverse = 1/biggest_in_row

    A = A.copy()
    N = A.shape[0]
    index_array = np.arange(N)

    # Loop over k
    for k in range(N):
        pivot = 0
        i_max = None

        # Loop over the rows to find the pivot
        for i in range(k, N):
            if np.abs(A[i, k] * inverse[i]) > np.abs(pivot * inverse[k]):
                pivot = A[i, k].copy()
                i_max = i

        # Swap rows if the pivot is not yet in row k
        if i_max != k:
            A = swap_rows(A, i_max, k)
            index_array = swap_rows(index_array, i_max, k)
            inverse = swap_rows(inverse, i_max, k)

        # Divide by the pivot and subtract to find the components of the LU matrix
        A[k+1:N, k] /= pivot
        A[k+1:N, k+1:] -= A[k+1:N, k:k+1] * A[k:k+1, k+1:]

    return A, index_array

def forward_substitution_unit_lower(LU: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve L*y = b using forward substitution,
    where L is the lower-triangular matrix.

    Parameters
    ----------
    LU : np.ndarray
        LU matrix from LU_decomposition.
    b : np.ndarray
        Right-hand side.

    Returns
    -------
    y : np.ndarray
        Solution vector.
    """

    # Loop over the elements of b to perform the forward substitution
    for i in range(1, len(b)):
        b[i] -= LU[i, :i] @ b[:i]

    return b


def backward_substitution_upper(LU: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve U*c = y using backward substitution,
    where U is the upper-triangular matrix.

    Parameters
    ----------
    LU : np.ndarray
        LU matrix from LU_decomposition.
    y : np.ndarray
        Right-hand side.

    Returns
    -------
    c : np.ndarray
        Solution vector.
    """
    N = len(y)

    # Loop backwards over the elements of y to perform the forward substitution
    for i in range(N):
        if i != 0:
            y[-1 - i] -= LU[-1 - i, N-i:] @ y[N-i:]
        y[-1 - i] /= LU[-1 - i, -1 - i]

    return y


def vandermonde_solve_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve for polynomial coefficients c from data (x,y) using the Vandermonde matrix.

    Parameters
    ----------
    x : np.ndarray
        x-values.
    y : np.ndarray
        y-values.

    Returns
    -------
    c : np.ndarray
        Polynomial coefficients.
    """
    vandermonde_matrix = construct_vandermonde_matrix(x)
    LU, index_array = LU_decomposition(vandermonde_matrix)
    b = forward_substitution_unit_lower(LU, y[index_array])

    return backward_substitution_upper(LU, b)


def evaluate_polynomial(c: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate y(x) = sum_j c[j] * x^j.

    Parameters
    ----------
    c : np.ndarray
        Polynomial coefficients.
    x_eval : np.ndarray
        Evaluation points.

    Returns
    -------
    y_eval : np.ndarray
        Polynomial values.
    """
    # Create an exponent array with 2 axes
    j = np.arange(len(c))[None, :]

    # Turning the x^j part of the sum into a matrix M such that y = M @ c
    matrix = x_eval[:, None] ** j

    return matrix @ c


def neville(x_data: np.ndarray, y_data: np.ndarray, x_interp: float) -> float:
    """
    Function that applies Nevilles algorithm to calculate the function value at x_interp.

    Parameters
    ------------
    x_data (np.ndarray): Array of x data points.
    y_data (np.ndarray): Array of y data points.
    x_interp (float): The x value at which to interpolate.

    Returns
    ------------
    float: The interpolated y value at x_interp.
    """
    M = len(x_data)
    new_values = y_data.copy()

    # Loop over the number of iterations to be performed
    for k in range(M - 1):
        new_values[:M-k-1] = ((x_data[1 + k:M] - x_interp) * new_values[:M-k-1] + (x_interp - x_data[:M-k-1]) * new_values[1:M-k])/(x_data[1 + k:M] - x_data[:M-k-1])
    
    return new_values[0]


# you can merge the function below with LU_decomposition to make it more efficient
def run_LU_iterations(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = 11,
    coeffs_output_path: str = "Coefficients_per_iteration.txt",
):
    """
    Iteratively improves computation of coefficients c.

    Parameters
    ----------
    x : np.ndarray
        x-values.
    y : np.ndarray
        y-values.
    iterations : int
        Number of iterations.
    coeffs_output_path : str
        File to write coefficient values per iteration.

    Returns
    -------
    coeffs_history :
        List of coefficient vectors.
    """
    vandermonde_matrix = construct_vandermonde_matrix(x)
    LU, index_array = LU_decomposition(vandermonde_matrix)
    delta_y = y.copy()
    coeffs_history = np.zeros(iterations, dtype=np.ndarray)
    
    # Looping over iterations
    for i in range(iterations):
        delta_b = forward_substitution_unit_lower(LU, delta_y[index_array])
        delta_b = backward_substitution_upper(LU, delta_b)

        # For the first iteration store the base estimate for c
        if i == 0:
            c = delta_b.copy()

        # For the other iterations improve the estimate of c using delta_b
        else:
            c -= delta_b
        
        delta_y = vandermonde_matrix @ c - y

        # Storing the new coefficients
        coeffs_history[i] = c.copy()
    
    # Writing the coefficients of each iteration to .t
    with open(coeffs_output_path, "w", encoding="utf-8") as f:
        for coeffs in coeffs_history:
            for i, coef in enumerate(coeffs):
                f.write(f"c$_{i+1}$ = {coef}, ")
            f.write("\n")
        
    return coeffs_history


def plot_part_a(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_c: np.ndarray,
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (a) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_c : np.ndarray
        Polynomial coefficients c.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = evaluate_polynomial(coeffs_c, xx)
    y_at_data = evaluate_polynomial(coeffs_c, x_data)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linewidth=3)
    axs[0].set_xlim(np.floor(xx[0])-0.01*(xx[-1]-xx[0]), np.ceil(xx[-1])+0.01*(xx[-1]-xx[0]))
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via LU decomposition"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linewidth=3)

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2a.pdf"))
    plt.close()


def plot_part_b(
    x_data: np.ndarray,
    y_data: np.ndarray,
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (b) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = np.array([neville(x_data, y_data, x) for x in xx], dtype=np.float64)
    y_at_data = np.array([neville(x_data, y_data, x) for x in x_data], dtype=np.float64)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linestyle="dashed", linewidth=3)
    axs[0].set_xlim(np.floor(xx[0])-0.01*(xx[-1]-xx[0]), np.ceil(xx[-1])+0.01*(xx[-1]-xx[0]))
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via Neville's algorithm"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linestyle="dashed", linewidth=3)

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2b.pdf"))
    plt.close()


def plot_part_c(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_history: list[np.ndarray],
    iterations_num: list[int] = [0, 1, 10],
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (c) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_history : list[np.ndarray]
        Coefficients per iteration.
    iterations_num : list[int]
        Iteration numbers to plot.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """

    linstyl = ["solid", "dashed", "dotted"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    xx = np.linspace(x_data[0], x_data[-1], 1001)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0, color="black", label="data")

    for i, k in enumerate(iterations_num):
        if k >= len(coeffs_history):
            continue
        c = coeffs_history[k]
        yy = evaluate_polynomial(c, xx)
        y_at_data = evaluate_polynomial(c, x_data)
        diff = np.abs(y_at_data - y_data)

        axs[0].plot(
            xx,
            yy,
            linestyle=linstyl[i],
            color=colors[i],
            linewidth=3,
            label=f"Iteration {k}",
        )
        axs[1].plot(x_data, diff, linestyle=linstyl[i], color=colors[i], linewidth=3)

    axs[0].set_xlim(np.floor(xx[0])-0.01*(xx[-1]-xx[0]), np.ceil(xx[-1])+0.01*(xx[-1]-xx[0]))
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2c.pdf"))
    plt.close()


def main():
    os.makedirs("Plots", exist_ok=True)
    x_data, y_data = load_data()

    # compute times
    number = 10

    t_a = (
        timeit.timeit(
            stmt=lambda: vandermonde_solve_coefficients(x_data, y_data),
            number=number,
        )
        / number
    )

    xx = np.linspace(x_data[0], x_data[-1], 1001)
    t_b = (
        timeit.timeit(
            stmt=lambda: np.array(
                [neville(x_data, y_data, x) for x in xx], dtype=np.float64
            ),
            number=number,
        )
        / number
    )

    t_c = (
        timeit.timeit(
            stmt=lambda: run_LU_iterations(x_data, y_data, iterations=11),
            number=number,
        )
        / number
    )

    # write all timing
    with open("Execution_times.txt", "w", encoding="utf-8") as f:
        f.write(f"\\item Execution time for part (a): {t_a:.5f} seconds\n")
        f.write(f"\\item Execution time for part (b): {t_b:.5f} seconds\n")
        f.write(f"\\item Execution time for part (c): {t_c:.5f} seconds\n")
    c_a = vandermonde_solve_coefficients(x_data, y_data)
    plot_part_a(x_data, y_data, c_a)

    formatted_c = [f"{coef:.3e}" for coef in c_a]
    with open("Coefficients_output.txt", "w", encoding="utf-8") as f:
        for i, coef in enumerate(formatted_c):
            f.write(f"c$_{i+1}$ = {coef}, ")
    plot_part_b(x_data, y_data)

    coeffs_history = run_LU_iterations(
        x_data,
        y_data,
        iterations=11,
        coeffs_output_path="Coefficients_per_iteration.txt",
    )
    plot_part_c(x_data, y_data, coeffs_history, iterations_num=[0, 1, 10])


if __name__ == "__main__":
    main()
