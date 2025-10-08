import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse import diags
def irls_cg(A, b, reg_lambda: float = 1, p: float = 1.0, maxiter: int = 100, tol: float = 1e-4):
    """
    Optimized Iteratively Reweighted Least Squares (IRLS). This algorithm handles complex data.
    
    Parameters:
    -----------
    A (`np.ndarray`):
        Input matrix (m x n), can be complex.
    b (`np.ndarray`):
        Target vector (m,), can be complex.
    reg_lambda (`float`):
        Regularization parameter.
    p (`float`):
        Norm parameter (1 =< p <= 2).
    maxiter (`int`):
        Maximum number of IRLS iterations.
    tol (`float`):
        Convergence tolerance.
    
    Returns:
    --------
    x np.ndarray
        Solution vector (n,), can be complex.
    """
    m, n = A.shape
    # Initial solution using LinearOperator for (A^H A + lambda I)x = A^H b
    def AtA_plus_reg(v):
        return A.conj().T @ (A @ v) + reg_lambda * v
    
    AtA_op = LinearOperator(shape=(n, n), matvec=AtA_plus_reg, dtype=A.dtype)
   
   # gradient opeartor
    # data = [-np.ones(n), np.ones(n-1)]
    # offsets = [0, -1]
    # G = diags(data, offsets, shape=(n, n)).toarray()
    # G[-1][-1] = 1

    # initial guass
    x, _ = cg(AtA_op, A.conj().T @ b, maxiter=20)
    
    # ensure p is valid
    p = max(min(p, 2.0), 1.)

    for _ in range(maxiter):
        x_old = x.copy()
        r = b - A @ x  # Residual
        epsilon = max(np.percentile(np.abs(r), 2), 1e-6)
        
        # Compute weights (complex-safe with np.abs)
        Wm_diag = np.maximum(np.abs(x), 1e-6) ** (p - 2)
        # Wr_norm2 = np.maximum(np.abs(r), epsilon) ** (-1)
        Wr_diag = np.ones_like(r)   # according my experience, Wr is not necessary or redundant
        
        # Define LinearOperator for A^H W_r A + lambda W_m
        def weighted_system(v):
            Wr_Av = Wr_diag * (A @ v)          # W_r @ A
            At_Wr_Av = A.conj().T @ Wr_Av      # A.T @ Wr @ A
            Wm_v = reg_lambda * Wm_diag * v    # lambda * W_m
            return At_Wr_Av + Wm_v
        
        weighted_op = LinearOperator(shape=(n, n), matvec=weighted_system, dtype=A.dtype)
        
        # Right-hand side: A^H W_r b
        rhs = A.conj().T @ (Wr_diag * b)
        
        x, _ = cg(weighted_op, rhs, x0=x_old, maxiter=100)
        
        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break
    
    return x

