import numpy as np

'''
Two methods are provided to calculate c-eigenpairs of a symmetric matrix.
Theoretically (from G. Beylkin and L. Monz´on, Applied and Computational Harmonic Analysis 19, 17 (2005)), Takagi's factorization should be used to control the error of Prony's method.
In practice, however, we are solving an overdetermined problem and we find SVD can always give equally good approximations.
'''

def takagi_fact(A):
    n = A.shape[0]
    err = n * 1.e-15
    assert A.ndim == 2 and A.shape[1] == n
    assert np.linalg.norm(A.T - A) < err
    
    if np.linalg.norm(A.imag) < err:
        A = A.real
        u, s, vh = np.linalg.svd(A, hermitian=True)
        assert np.linalg.norm(u.imag) < err and np.linalg.norm(vh.imag) < err
        v = vh.T
        idx = np.argmax(u, axis=0)
        ratio = u[idx, np.arange(n)] / v[idx, np.arange(n)]
        assert np.linalg.norm(np.abs(ratio) - 1, ord=np.inf) < 1.e-15
        S = s
        U = np.array([v[:, i] if np.abs(ratio[i] - 1.0) < 1.e-12 else 1.0j * v[:, i] for i in range(n)]).T
    else:
        B = A.real
        C = A.imag
        F = np.vstack((np.hstack((B, C)), np.hstack((C, -B))))
        u, s, vh = np.linalg.svd(F, hermitian=True)
        assert np.linalg.norm(u.imag) < err and np.linalg.norm(vh.imag) < err
        u = u[:, ::2]
        v = vh.T[:, ::2]
        idx = np.argmax(u, axis=0)
        ratio = u[idx, np.arange(n)] / v[idx, np.arange(n)]
        assert np.linalg.norm(np.abs(ratio) - 1, ord=np.inf) < 1.e-15
        v1 = v[:n, :]
        v2 = v[n:, :]
        S = s[::2]
        U = np.array([v1[:, i] - 1.0j * v2[:, i] if np.abs(ratio[i] - 1.0) < 1.e-12 else v2[:, i] + 1.0j * v1[:, i] for i in range(n)]).T
    
    assert np.linalg.norm(A @ U - np.conjugate(U) @ np.diag(S)) < err
    
    return S, U

def svd(A):
    n = A.shape[0]
    err = n * 1.e-15
    assert A.ndim == 2 and A.shape[1] == n
    assert np.linalg.norm(A.T - A) < err
    
    if np.linalg.norm(A.imag) < err:
        u, s, vh = np.linalg.svd(A.real, hermitian=True)
    else:
        u, s, vh = np.linalg.svd(A)
        
    S = s
    U = np.conjugate(vh.T)
    
    return S, U
