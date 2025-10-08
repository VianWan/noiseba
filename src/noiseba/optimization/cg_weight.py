# import os

# # constrian thread as threadnum
# # run before import numpy
# threadnum = '36'
# os.environ['OMP_NUM_THREADS'] = threadnum
# os.environ['MKL_NUM_THREADS'] = threadnum
# os.environ['OPENBLAS_NUM_THREADS'] = threadnum

import numpy as np
from scipy.sparse.linalg import LinearOperator
from typing import  Optional, Union


def cg_weight(
    A_op: LinearOperator,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    reg_lambda: Union[float, int] = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
    norm: float = 1.0,
) -> np.ndarray:
    """
    Solves the Lp-norm regularized least squares problem using an iterative reweighted conjugate gradient method.
    The algorithm approximately solves min ||A*x - b||_p + lambda*||x||_p, p belong to [1, 2].

    This implementation correctly handles complex data.

    Parameters:
    ----------
    A_op : scipy.sparse.linalg.LinearOperator
        The linear operator. Must implement the .matvec() (forward) and .rmatvec() (Hermitian transpose) methods.
    b : np.ndarray
        The observation data vector (can be complex).
    x0 : np.ndarray, optional
        Initial guess for the solution vector. If None, starts from a zero vector.
    reg_lambda: float, optional
        Regularization parameter, 0 for no regularization.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance based on the weighted gradient norm.
    norm : float,
        Lp norm order.
    Returns:
    -------
    x : np.ndarray
        The computed solution vector.
    """
    # --- 1. Initialization ---
    if x0 is None:
        x = np.zeros(A_op.shape[1], dtype=b.dtype)
    else:
        x = x0.copy()
    
    # define hermitian transpose opeartor
    if isinstance(A_op, np.ndarray):
        A = LinearOperator(
            shape=A_op.shape,
            matvec=lambda v: A_op @ v,
            rmatvec=lambda v: A_op.conj().T @ v,
            dtype=A_op.dtype,
        )
    else:
        A = A_op
    
    # Compute the initial residual
    # r_k = b - A @ x_k
    r = b - A @ x

    # Compute initial weights
    epsilon = np.percentile(np.abs(r), 2)
    # epsilon = np.max(np.abs(r)) / 100

    # Compute the initial weighted gradient (residual of the normal equation)
    # s_k = Wm * A.H * (Wr * r_k)
    s = A.H @ r

    # Initialize the search direction, also conjugate gradient
    p = s.copy()

    # gamma_old = ||s_k||^2
    gamma_old = np.vdot(s, s).real

    # --- 2. Iteration Loop ---
    for iteration in range(max_iter):
        # Compute q_k = A @ p_k
        Ap = A @ p

        # Compute step size alpha_k = ||s_k||^2 / ||A*p_k||^2
        alpha = gamma_old / np.vdot(Ap, Ap).real

        # Update the solution: x_{k+1} = x_k + alpha_k * p_k
        x = x + alpha * p

        # Update the residual of the original problem: r_{k+1} = r_k - alpha_k * A*p_k
        r = r - alpha * Ap

        # --- Core Step: Update weights and weighted gradient ---
        # Update weights; L1 norm below
        # werid, most time Wm is |m|^-1
        wr = (np.maximum(np.abs(r), epsilon))**((norm-2) / 2) # W_r = ||r_i||^((p-2)/2)
        wm = np.abs(x)**(0.5)                           # W_m = ||m_i||^((2-p)/2)

        wm_final = (1 - reg_lambda) * 1.0 + reg_lambda * wm

        # Update the weighted gradient: s_{k+1} = Wm * A.H * (Wr * r_{k+1})
        s = wm_final * (A.H @ (wr * r))

        # Compute gamma_new = ||s_{k+1}||^2
        gamma_new = np.vdot(s, s).real

        # Check for convergence
        grad_norm = np.sqrt(gamma_new)

        if grad_norm < tol:
            # print(f"\nConverged after {iteration + 1} iterations.")
            return x

        # Compute beta_{k+1} = ||s_{k+1}||^2 / ||s_k||^2
        beta = gamma_new / gamma_old

        # Update the search direction: p_{k+1} = s_{k+1} + beta_{k+1} * p_k
        p = s + beta * p

        # Update gamma_old for the next iteration
        gamma_old = gamma_new

    return x



# --- Usage Example ---
if __name__ == '__main__':
    # 1. Create a simulated complex problem
    m, n = 10000, 5000
    # Create a random complex matrix
    A_matrix = np.random.randn(m, n) + 1j * np.random.randn(m, n)

    # Wrap it into a LinearOperator
    def matvec(v: np.ndarray) -> np.ndarray:
        return A_matrix @ v

    def rmatvec(v: np.ndarray) -> np.ndarray:
        # rmatvec must implement the Hermitian transpose
        return A_matrix.conj().T @ v

    A_op = LinearOperator(
        shape=(m, n),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.complex128
    )

    # 2. Create a sparse true solution
    x_true = np.zeros(n, dtype=np.complex128)
    x_true[10] = 1 + 2j
    x_true[25] = -3 - 1j
    x_true[40] = 5j

    # 3. Generate observation data with some noise
    b_true = A_op @ x_true
    noise = (np.random.randn(m) + 1j * np.random.randn(m)) * 0.1
    b_data = b_true + noise

    # 4. Call the weighted CGLS solver
    print("--- Solving... ---")
    x_solution = cg_weight(
        A_op,
        b_data,
        reg_lambda=1,
        max_iter=100,
        tol=1e-5,
    )

    # 5. Compare the results
    error = np.linalg.norm(x_solution - x_true) / np.linalg.norm(x_true)
    print("\n--- Results ---")
    print(f"Relative error between solution and true solution: {error:.4f}")
