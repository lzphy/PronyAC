import numpy as np
from c_eigenpair import *

class PronyApprox:
    '''
    Stable Prony's approximation which works for both real-valued and complex-valued damping functions.
    '''
    
    def __init__(self, h_k, a = 0, b = 1):
        '''
        Initialize with an odd number of function values sampled on a uniform grid from a to b.
        '''
        if h_k.size % 2 != 1:
            raise Exception("number of sampling points must be odd!")
        self.N = h_k.size // 2
        self.a = a
        self.b = b
        self.x_k = np.linspace(a, b, 2 * self.N + 1)
        if np.max(np.abs(h_k.imag)) < 1.e-15:
            self.type = float
            self.h_k = np.array(h_k.real)
        else:
            self.type = complex
            self.h_k = np.array(h_k)
        
        self.H = np.zeros((self.N + 1, self.N + 1), dtype=self.h_k.dtype)
        for row in range(self.N + 1):
            self.H[row, :] = self.h_k[row : (row + self.N + 1)]
        
        S, V = svd(self.H)
        self.S = S
        self.V = V
    
    def find_idx_with_err(self, err):
        '''
        Find the index of the corresponding singular value for the given error tolerance.
        '''
        idx = 0
        for i in range(self.S.size):
            if self.S[i] < err:
                idx = i
                break
        if self.S[idx] >= err:
            raise Exception("err is set to be too small!")
        
        return idx
    
    def find_idx_with_exp_decay(self):
        '''
        Find the maximum index for the exponentially decaying region.
        '''
        n_max = min(3 * int(np.log(1.e12)), int(0.8 * self.S.size))
        idx_fit = np.arange(int(0.8 * n_max), n_max)
        val_fit = self.S[idx_fit]
        
        A = np.vstack((idx_fit, np.ones_like(idx_fit))).T
        a, b = np.linalg.pinv(A) @ np.log(val_fit)
        self.S_approx = np.exp(a * np.arange(n_max) + b)
        idx = sum(self.S[:n_max] > 5.0 * self.S_approx) + 1
        
        return idx
    
    def find_v_with_idx(self, idx):
        '''
        Find the c-eigenpair for the given index idx.
        '''
        if idx >= self.S.size:
            raise Exception("index is invalid!")
        
        self.idx = idx
        self.sigma = self.S[idx]
        self.v = self.V[:, idx]
    
    def find_approx(self, full = False, cutoff = 1.0, keep_largest = True):
        '''
        Find Prony's approximation.
        If full is False, only nodes with modulus smaller than cutoff will be considered; otherwise, all nodes will be considered. It is suggested to keep it False.
        '''
        self.find_gamma(full=full, cutoff=cutoff)
        self.find_omega()
        if keep_largest:
            idx_sort = np.argsort(np.abs(self.omega))[::-1][:self.idx]
            self.omega = self.omega[idx_sort]
            self.gamma = self.gamma[idx_sort]
        h_k_approx = self.get_value(self.x_k)
        self.err_max = np.abs(h_k_approx - self.h_k).max()
        self.err_ave = np.abs(h_k_approx - self.h_k).mean()
    
    def find_approx_opt(self, full = False, cutoff = 1.0):
        '''
        Find the optimal Prony's approximation.
        If full is False, only nodes with modulus smaller than cutoff will be considered; otherwise, all nodes will be considered. It is suggested to keep it False.
        '''
        idx_exp = self.find_idx_with_exp_decay()
        err_exp = self.S[idx_exp]
        idx_list = np.arange(self.find_idx_with_err(1000 * err_exp), min(idx_exp + 10, self.S.size))
        err_list = np.zeros(idx_list.shape, dtype=np.float64)
        
        for i in range(idx_list.size):
            idx = idx_list[i]
            self.find_v_with_idx(idx)
            self.find_approx(full=full, cutoff=cutoff, keep_largest=True)
            err_list[i]  = self.err_ave
        
        idx_f = idx_list[np.argmin(err_list)]
        self.find_v_with_idx(idx_f)
        self.find_approx(full=full, cutoff=cutoff, keep_largest=True)
    
    def find_roots(self, u):
        '''
        Find roots of a polyminal with coefficients u. Note the convention difference so that u should be inversed.
        '''
        return np.roots(u[::-1])
    
    def find_gamma(self, full = False, cutoff = 1.0):
        '''
        Find nodes gamma.
        It is suggested that full is always set to be False, so that Prony's approximation always keeps stable. 
        However, this requires that the function to be approximated has decaying magnitude. For our method, this condition is always satisfied.
        '''
        import warnings
        self.gamma = self.find_roots(self.v)
        if not full:
            self.gamma = self.gamma[np.abs(self.gamma) < cutoff]
        else:
            warnings.warn("Prony method is unstable when calculating weights of full nodes and may fail starting from around sigma = 10^-7")
    
    def find_omega(self):
        '''
        Find weights of corresponding nodes gamma.
        '''
        A = np.zeros((self.h_k.size, self.gamma.size), dtype=np.complex128)
        for i in range(A.shape[0]):
            A[i, :] = self.gamma ** i

        self.omega = np.dot(np.linalg.pinv(A), self.h_k)
    
    def get_value(self, x):
        '''
        Get the approximated function value at point x.
        '''
        x0 = (x - self.a) / (self.b - self.a)
        if np.any(x0 < -1.e-12) or np.any(x0 > 1.0 + 1.e-12):
            raise Exception("Prony approximation only has error control for x in [x_min, x_max]!")
        
        if np.isscalar(x0):
            A = self.gamma ** (2 * self.N * x0)
            value = np.dot(A, self.omega)
        else:
            A = np.zeros((x0.size, self.omega.size), dtype=np.complex128)
            for i in range(A.shape[0]):
                A[i, :] = self.gamma ** (2 * self.N * x0[i])
            value = np.dot(A, self.omega)
        return (value.real if self.type == float else value)
