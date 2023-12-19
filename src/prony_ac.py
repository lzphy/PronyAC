import numpy as np
import scipy.integrate as integrate
from prony_approx import *
from con_map import *
import warnings
'''
In practice, we always want to get numerical integration as accurate as possible. 
Occasionally, if we set the error tolerance to be too small, some integration warnings will be raised. 
But it still gives the best possible result in the current situation. 
For every case we tested, we found the integration warning is always not an issue. 
So we simply ignore this warning. 
If you do not like this, please just comment out the following single line.
'''
warnings.filterwarnings("ignore")

class PronyAC:
    '''
    A universal analytic continuation program working for fermionic, bosonic, diagonal, off-diagonal, discrete, continuous, noiseless and noisy cases.
    '''
    def __init__(self, G_w, w, optimize = True, n_k = 1001, x_min = -10, x_max = 10, y_min = -10, y_max = 10, err = None):
        '''
        G_w is a 1-d array containing the Matsubara data, w is the corresponding sampling grid;
        n_k is the maximum number of contour integrals. 
        Only poles located within the rectangle  x_min < x < x_max, y_min < y < y_max are recovered. Our method is not sensitive to these four paramters. So the range can be set to be relatively large. 
        If the error tolerance err is given, the continuation will be carried out in this tolerance;
        else if optimize is True, the tolerance will be chosen to be the optimal one;
        else the tolerance will be chosen to be the last singular value in the exponentially decaying range.
        For noisy data, it is highly suggested to set optimize to be True (and please do not provide err). Although the simulation speed will slow down (will take about several minutes), this will highly improve the noise resistance and the accuracy of final results. 
        For noiseless data, setting optimize to be True will still give best results. However, setting err=1.e-12 will be much faster and give nearly optimal results.
        '''
        assert G_w.size == w.size
        assert w.size >= 100
        assert np.linalg.norm(np.diff(np.diff(w)), ord=np.inf) < 1.e-10
        
        #the number of input data points for Prony's approximation must be odd
        N_odd  = w.size if w.size % 2 == 1 else w.size - 1
        self.G_w = G_w[:N_odd]
        self.w = w[:N_odd]
        
        self.optimize = optimize
        self.err = err
        
        #perform the first Prony's approximation to approximate Matsubara data
        self.p_o = PronyApprox(self.G_w, self.w[0], self.w[-1])
        if self.err is not None:
            idx = self.p_o.find_idx_with_err(self.err)
            self.p_o.find_v_with_idx(idx)
            self.p_o.find_approx()
        elif self.optimize is False:
            idx = self.p_o.find_idx_with_exp_decay()
            self.p_o.find_v_with_idx(idx)
            self.p_o.find_approx()
        else:
            self.p_o.find_approx_opt()
        
        #get the corresponding conformal mapping
        w_m = 0.5 * (self.w[0] + self.w[-1])
        dw_h = 0.5 * (self.w[-1] - self.w[0])
        self.con_map = ConMapGeneric(w_m, dw_h)

        #calculate contour integrals
        self.cal_hk_generic(n_k)

        #apply the second Prony's approximation to recover poles
        self.find_poles(x_min, x_max, y_min, y_max)
    
    def cal_hk_generic(self, n_k = 1001):
        '''
        Calculate the contour integrals. Cutoff is set to be much smaller than the predetermined error tolerance.
        '''
        cutoff = 0.02 * self.p_o.sigma
        err = max(0.0001 * cutoff, 1.e-14)
        
        self.h_k = np.zeros((n_k,), dtype=np.complex128)        
        for k in range(n_k):
            if k % 2 == 0:
                int_r = integrate.quad(lambda x: self.p_o.get_value(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).real * np.sin((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                int_i = integrate.quad(lambda x: self.p_o.get_value(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).imag * np.sin((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                self.h_k[k] = (1.0j / np.pi) * (int_r + 1.0j * int_i)
            else:
                int_r = integrate.quad(lambda x: self.p_o.get_value(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).real * np.cos((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                int_i = integrate.quad(lambda x: self.p_o.get_value(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).imag * np.cos((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                self.h_k[k] = (1.0 / np.pi) * (int_r + 1.0j * int_i)
            if np.abs(self.h_k[k]) < cutoff:
                break
        
        if k % 2 == 0:
            self.h_k = self.h_k[:(k+1)]
        else:
            self.h_k = self.h_k[:k]

    def find_poles(self, x_min, x_max, y_min, y_max):
        '''
        Recover poles from contour integrals h_k.
        '''
        #apply the second Prony's method
        self.p_f = PronyApprox(self.h_k)
        if self.err is not None:
            idx = self.p_f.find_idx_with_err(self.err)
            self.p_f.find_v_with_idx(idx)
            self.p_f.find_approx(f=self.con_map.z, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        elif self.optimize is False:
            idx = self.p_f.find_idx_with_exp_decay()
            self.p_f.find_v_with_idx(idx)
            self.p_f.find_approx(f=self.con_map.z, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        else:
            self.p_f.find_approx_opt(f=self.con_map.z, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        
        #tranform poles from w-plane to z-plane
        weight   = (self.p_f.omega * self.con_map.dz(self.p_f.gamma))
        location = self.con_map.z(self.p_f.gamma)
        
        #discard poles with negligible weights
        idx1 = np.absolute(weight) > 1.e-8
        weight = weight[idx1]
        location = location[idx1]
        
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx2 = np.argsort(location.real)
        self.pole_weight   =  weight[idx2]
        self.pole_location =  location[idx2]
