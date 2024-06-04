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
    def __init__(self, G_w, w, optimize = False, n_k = 301, x_min = -10, x_max = 10, y_min = -10, y_max = 10, err = None, impose_sym = False, G_approx=None, pole_real=False):
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
        assert np.linalg.norm(np.diff(np.diff(w)), ord=np.inf) < 1.e-10
        
        #the number of input data points for Prony's approximation must be odd
        N_odd  = w.size if w.size % 2 == 1 else w.size - 1
        #N_even = w.size if w.size % 2 == 0 else w.size - 1
        #if impose_sym is False:
        self.G_w = G_w[:N_odd]
        self.w = w[:N_odd]
        #else:
            #self.G_w = G_w[:N_even]
            #self.w = w[:N_even]
        
        self.optimize = optimize
        self.err = err
        self.impose_sym = impose_sym
        self.G_approx = G_approx
        self.pole_real = pole_real
        
        #perform the first Prony's approximation to approximate Matsubara data
        if G_approx is None:
            if impose_sym is False:
                self.p_o = PronyApprox(self.G_w, self.w[0], self.w[-1])
            else:
                self.p_o = PronyApprox(np.insert(self.G_w, 0, np.conjugate(self.G_w[0])), -self.w[0], self.w[-1])
        
            if self.err is not None:
                idx = self.p_o.find_idx_with_err(self.err)
                self.p_o.find_v_with_idx(idx)
                self.p_o.find_approx(full=False, cutoff=1.0 + 0.5 / self.p_o.N)
            elif self.optimize is False:
                idx = self.p_o.find_idx_with_exp_decay()
                self.p_o.find_v_with_idx(idx)
                self.p_o.find_approx(full=False, cutoff=1.0 + 0.5 / self.p_o.N)
            else:
                self.p_o.find_approx_opt(full=False, cutoff=1.0 + 0.5 / self.p_o.N)
        
        #get the corresponding conformal mapping
        if impose_sym is False:
            w_m = 0.5 * (self.w[0] + self.w[-1])
            dw_h = 0.5 * (self.w[-1] - self.w[0])
            self.con_map = ConMapGeneric(w_m, dw_h)
        else:
            #w_m = 0.0
            #dw_h = self.w[-1]
            self.con_map = ConMapGapless(self.w[0])

        #calculate contour integrals
        if impose_sym is False:
            self.cal_hk_generic(n_k)
        else:
            #n_k = min(n_k, G_w.size)
            #self.cal_hk_gapped(n_k, G_approx)
            self.cal_hk_gapless(n_k, G_approx)

        #apply the second Prony's approximation to recover poles
        self.find_poles() #x_min, x_max, y_min, y_max)
    
    def cal_hk_generic(self, n_k = 1001):
        '''
        Calculate the contour integrals. Cutoff is set to be much smaller than the predetermined error tolerance.
        '''
        cutoff = 0.1 * self.p_o.sigma
        #cutoff = self.err
        err = 0.01 * cutoff
        
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
        
        #the rank of h_k should not be greater than that of G_w
        if self.h_k.size > self.G_w.size:
            self.h_k = self.h_k[:self.G_w.size]

    def cal_hk_gapless(self, n_k, G_approx):
        '''
        Calculate the contour integrals. Cutoff is set to be much smaller than the predetermined error tolerance.
        '''
        #cutoff = 0.1 * self.p_o.sigma
        #cutoff = 0.1 * self.err
        cutoff = self.err
        err = 0.01 * cutoff
        
        if G_approx is None:
            G_approx = lambda x: self.p_o.get_value(x) if x >= 0.0 else np.conjugate(self.p_o.get_value(-x))
        
        self.h_k = np.zeros((n_k,), dtype=np.float_)
        for k in range(n_k):
            self.h_k[k] = self.cal_hk_gapless_indiv(G_approx, k, err)
            if k >= 1:
                if np.abs(self.h_k[k]) < cutoff and np.abs(self.h_k[k - 1]) < cutoff:
                    break
        
        if k % 2 == 0:
            self.h_k = self.h_k[:(k + 1)]
        else:
            self.h_k = self.h_k[:k]
    
    def cal_hk_gapless_indiv(self, G_approx, k, err):
        theta0 = 1.e-12
        if k % 2 == 0:
            result =   (-2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).imag * np.sin((k + 1) * x), theta0, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
        else:
            result =    (2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).real * np.cos((k + 1) * x), theta0, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
        return result
    
    def cal_hk_gapped(self, n_k, G_approx):
        '''
        Calculate the contour integrals. Cutoff is set to be much smaller than the predetermined error tolerance.
        '''
        #cutoff = 0.1 * self.p_o.sigma
        #cutoff = 0.1 * self.err
        cutoff = self.err
        err = 0.01 * cutoff
        
        if G_approx is None:
            G_approx = lambda x: self.p_o.get_value(x) if x >= 0.0 else np.conjugate(self.p_o.get_value(-x))
        self.h_k = np.zeros((n_k,), dtype=np.complex128)
        for k in range(n_k):
            if k % 2 == 0:
                int_r = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).real * np.sin((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                int_i = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).imag * np.sin((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                self.h_k[k] = (1.0j / np.pi) * (int_r + 1.0j * int_i)
            else:
                int_r = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).real * np.cos((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                int_i = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).imag * np.cos((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
                self.h_k[k] = (1.0 / np.pi) * (int_r + 1.0j * int_i)
            if k >= 1:
                if np.abs(self.h_k[k]) < cutoff and np.abs(self.h_k[k-1]) < cutoff:
                    break
        
        if k % 2 == 0:
            self.h_k = self.h_k[:(k+1)]
        else:
            self.h_k = self.h_k[:k]
        self.h_k = self.h_k.real
        
        #the rank of h_k should not be greater than that of G_w
        #if self.h_k.size > self.G_w.size:
        #    self.h_k = self.h_k[:self.G_w.size]
    
    def find_poles(self): #, x_min, x_max, y_min, y_max):
        '''
        Recover poles from contour integrals h_k.
        '''
        #apply the second Prony's method
        self.p_f = PronyApprox(self.h_k)
        if self.err is not None:
            idx = self.p_f.find_idx_with_err(self.err)
            self.p_f.find_v_with_idx(idx)
            self.p_f.find_approx()
        elif self.optimize is False:
            idx = self.p_f.find_idx_with_exp_decay()
            self.p_f.find_v_with_idx(idx)
            self.p_f.find_approx()
        else:
            self.p_f.find_approx_opt()
        
        #tranform poles from w-plane to z-plane
        if self.impose_sym:
            if self.pole_real:
                self.p_f.gamma = self.p_f.gamma[np.abs(self.p_f.gamma.imag) < 1.e-3]
                self.p_f.find_omega()
            omega = self.p_f.omega#[np.abs(self.p_f.omega) > 1.e-7]
            gamma = self.p_f.gamma#[np.abs(self.p_f.omega) > 1.e-7]
        else:
            omega = self.p_f.omega
            gamma = self.p_f.gamma
        weight   = omega * self.con_map.dz(gamma)
        location = self.con_map.z(gamma)
        
        #discard poles with negligible weights
        idx1 = np.absolute(weight) > 10 * self.err #1.e-8
        weight = weight[idx1]
        location = location[idx1]
        
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx2 = np.argsort(location.real)
        self.pole_weight   =  weight[idx2]
        self.pole_location =  location[idx2]
        '''
        if self.impose_sym is True:
            #xl = self.pole_location[np.logical_and(self.pole_location != np.inf, np.abs(self.pole_location.imag) < 1.e-3)]
            xl = self.pole_location[self.pole_location != np.inf]
            A = np.zeros((self.w.size, xl.size), dtype=np.complex_)
            for i in range(xl.size):
                A[:, i] = 1.0 / (1j * self.w - xl[i])
            Al = np.linalg.pinv(A) @ self.G_w
            self.pole_weight   =  Al
            self.pole_location =  xl
        '''
     
def cal_G(wn, Al, xl):
    if wn < 0:
        return np.conjugate(cal_G(-wn, Al, xl))
    #G_z = np.zeros(wn.shape, dtype=np.complex128)
    G_z = 0.0
    for i in range(Al.size):
        G_z += Al[i] / (1j * wn - xl[i])
    return G_z

def PronyACReal(G_w, w, err = None, pole_real=False):
    ac1 = PronyAC(G_w, w, err=err, impose_sym=False)
    return PronyAC(G_w, w, err=ac1.p_o.sigma, impose_sym=True, G_approx = lambda x: cal_G(x, ac1.pole_weight, ac1.pole_location), pole_real=pole_real)
