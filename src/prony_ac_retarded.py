import numpy as np
import scipy.integrate as integrate
from prony_approx import *
from con_map import *

class PronyACRetarded:
    def __init__(self, G_R, eta = 0.0, w_min = -10, w_max = 10, err = None, n_k = 201):
        '''
        G_R is an analytic function of the retarded Green's function;
        eta is the broadening parameter for which G_R is evaluated;
        w_min and w_max are cutoffs of the real frequency;
        err is the error tolerance for which the continuation will be carried out;
        n_k is the maximum number of contour integrals.
        '''
        assert eta >= 0.0
        assert w_min < w_max
        assert err is not None
        assert n_k % 2 == 1
        
        self.G_R = G_R
        self.eta = eta
        self.err = err
        self.n_k = n_k
        
        w_m  = 0.5 * (w_max + w_min)
        dw_h = 0.5 * (w_max - w_min)
        self.con_map = ConMapGeneric(w_m, dw_h)

        #calculate contour integrals
        self.cal_hk_generic(self.G_R, n_k)
        
        #apply the Prony's approximation to recover poles
        self.find_poles()
    
    def cal_hk_generic(self, G_approx, n_k = 1001):
        '''
        Calculate the contour integrals. Cutoff is set to be much smaller than the predetermined error tolerance.
        '''
        self.h_k = np.zeros((n_k,), dtype=np.complex128)
        for idx in range(n_k):
            k = idx
            self.h_k[idx] = self.cal_hk_generic_indiv(G_approx, k, 0.01 * self.err)
            if np.abs(self.h_k[idx]) < self.err:
                break
        
        if idx % 2 == 0:
            self.h_k = self.h_k[:(idx+1)]
        else:
            self.h_k = self.h_k[:idx]
    
    def cal_hk_generic_indiv(self, G_approx, k, err):
        if k % 2 == 0:
            int_r = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).real * np.sin((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
            int_i = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).imag * np.sin((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
            return (1.0j / np.pi) * (int_r + 1.0j * int_i)
        else:
            int_r = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).real * np.cos((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
            int_i = integrate.quad(lambda x: G_approx(self.con_map.w_m +  self.con_map.dw_h * np.sin(x)).imag * np.cos((k + 1) * x), -0.5 * np.pi, 0.5 * np.pi, epsabs=err, epsrel=err, limit=1000000)[0]
            return (1.0 / np.pi) * (int_r + 1.0j * int_i)

    def find_poles(self):
        '''
        Recover poles from contour integrals h_k.
        '''
        #apply the Prony's method
        self.p_f = PronyApprox(self.h_k)
        idx = self.p_f.find_idx_with_err(self.err)
        self.p_f.find_v_with_idx(idx)
        self.p_f.find_approx()
        
        #tranform poles from w-plane to z-plane
        weight   = (self.p_f.omega * self.con_map.dz(self.p_f.gamma))
        location = self.con_map.z(self.p_f.gamma)
        
        weight = -1j * weight
        location = -1j * location + 1j * self.eta

        #discard poles with negligible weights
        idx1 = np.absolute(weight) > 5 * self.err
        weight   = weight[idx1]
        location = location[idx1]
        
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx2 = np.argsort(location.real)
        self.pole_weight   =  weight[idx2]
        self.pole_location =  location[idx2]
