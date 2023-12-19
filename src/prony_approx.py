import numpy as np
from c_eigenpair import *

class PronyApprox:
    '''
    Stable Prony's approximation which works for both real-valued and complex-valued damping functions.
    '''
    
    def __init__(self, h_k, a = 0, b = 1):
        '''
        Initialize with an odd number of function values sampled on the uniform grid from a to b.
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
        self.S = S[:min(3 * int(np.log(1.e12)), int(0.8 * S.size))]
        self.V = V[:, :min(3 * int(np.log(1.e12)), int(0.8 * S.size))]
    
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
        from scipy.optimize import curve_fit
        
        n_max = self.S.size
        idx_fit = np.arange(int(0.8 * n_max), n_max)
        val_fit = self.S[idx_fit]
        
        try:
            param, param_cov = curve_fit(lambda x, a, b : a * x + b, idx_fit, val_fit)
            self.S_approx = param[0] * np.arange(n_max) + param[1]
            idx = sum(self.S > 10.0 * self.S_approx) - 1
        except:
            idx = self.S.size - 1
        
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
    
    def find_approx(self, full = False, f = None, x_min = 0, x_max = 0, y_min = 0, y_max = 0):
        '''
        Find Prony's approximation. 
        If full is False, only nodes strictly inside the unit disk will be considered; otherwise, all nodes will be considered. It is suggested to keep it False. 
        If function f is provided, then only nodes gamma with f(gamma).real in (x_min, x_max) and f(gamma).imag in (y_min, y_max) will be considered.
        '''
        self.find_gamma(full)
        if f is not None:
            self.cut_gamma_with_map(f, x_min, x_max, y_min, y_max)
        self.find_omega()
    
    def find_approx_opt(self, full = False, f = None, x_min = 0, x_max = 0, y_min = 0, y_max = 0):
        '''
        Find the optimal Prony's approximation.
        If full is False, only nodes strictly inside the unit disk will be considered; otherwise, all nodes will be considered. It is suggested to keep it False.
        If function f is provided, then only nodes gamma with f(gamma).real in (x_min, x_max) and f(gamma).imag in (y_min, y_max) will be considered.
        '''
        idx_exp = self.find_idx_with_exp_decay()
        err_exp = self.S[idx_exp]
        idx_list = np.arange(self.find_idx_with_err(1000 * err_exp), min(idx_exp + 10, self.S.size))
        var_list = np.zeros(idx_list.shape, dtype=np.float64)
        pole_list = np.zeros(idx_list.shape, dtype=np.int32)

        for i in range(idx_list.size):
            idx = idx_list[i]
            self.find_v_with_idx(idx)
            self.find_approx()
            approx = self.get_value(self.x_k)
            var_list[i]  = np.sqrt(np.var(np.abs(approx - self.h_k)))
            pole_list[i] = self.omega.size

        idx_c = pole_list < idx_list + 20
        idx_cut = idx_list[idx_c]
        var_cut = var_list[idx_c]

        idx_f = idx_cut[np.argmin(var_cut)]
        self.find_v_with_idx(idx_f)
        self.find_approx(full=full, f=f, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    
    def find_roots(self, u):
        '''
        Find roots of a polyminal with coefficients u. Note the convention difference so that u should be inversed.
        '''
        return np.roots(u[::-1])
    
    def find_gamma(self, full = False):
        '''
        Find nodes gamma. 
        It is suggested that full is always set to be False, so that Prony's approximation always keeps stable. 
        However, this requires that the function to be approximated has decaying magnitude. For our method, this condition is always satisfied.
        '''
        import warnings
        self.gamma = self.find_roots(self.v)
        if not full:
            self.gamma = self.gamma[np.abs(self.gamma) < 1.0]
        else:
            warnings.warn("Prony method is unstable when calculating weights of full nodes and may fail starting from around sigma = 10^-7")
    
    def cut_gamma_with_map(self, f, x_min, x_max, y_min, y_max):
        '''
        Do cutoff for nodes gamma so that f(gamma).real in (x_min, x_max) and f(gamma).imag in (y_min, y_max).
        '''
        assert x_min < x_max and y_min < y_max
        idx_x = np.logical_and(f(self.gamma).real > x_min, f(self.gamma).real < x_max)
        idx_y = np.logical_and(f(self.gamma).imag > y_min, f(self.gamma).imag < y_max)
        self.gamma = self.gamma[np.logical_and(idx_x, idx_y)]
    
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
        if np.isscalar(x0):
            A = self.gamma ** (2 * self.N * x0)
            value = np.dot(A, self.omega)
        else:
            A = np.zeros((x0.size, self.omega.size), dtype=np.complex128)
            for i in range(A.shape[0]):
                A[i, :] = self.gamma ** (2 * self.N * x0[i])
            value = np.dot(A, self.omega)
        return (value.real if self.type == float else value)
