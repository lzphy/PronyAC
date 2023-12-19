import numpy as np

class ConMapGeneric:
    '''
    Generic holomorphic mapping which works for any cases.
    '''
    def __init__(self, w_m, dw_h):
        '''
        Initialize the class with w_m and dw_h, which correspond to $\omega_{\rm m}$ and $\Delta \omega_{\rm h}$ in the paper, respectively.
        '''
        assert dw_h > 0.0
        assert w_m  - dw_h > -1.e-15
        self.w_m = w_m
        self.dw_h = dw_h
    
    def cal_z(self, w):
        '''
        Intermediate function of z(w) which only works for a single point.
        '''
        assert np.abs(w) < 1.0 + 1.e-15
        if w == 0.0:
            return np.inf
        else:
            return 0.5 * self.dw_h * (w - 1.0 / w) + 1j * self.w_m
        
    def cal_w(self, z):
        '''
        Intermediate function of w(z) which only works for a single point.
        '''
        x = (z - 1j * self.w_m) / self.dw_h
        w = x - np.sqrt(x ** 2.0 + 1.0)
        if np.absolute(w) > 1.0:
            w = 2.0 * x - w
        return w
    
    def cal_dz(self, w):
        '''
        Intermediate function of dz(w) which only works for single point.
        '''
        assert np.abs(w) < 1.0 + 1.e-15
        if w == 0.0:
            return np.inf
        else:
            return 0.5 * self.dw_h * (1.0 + 1.0 / w ** 2.0)
    
    def z(self, w):
        '''
        Calculate z from w.
        '''
        return np.vectorize(self.cal_z)(w)
    
    def w(self, z):
        '''
        Calculate w from z.
        '''
        return np.vectorize(self.cal_w)(z)
    
    def dz(self, w):
        '''
        Calculate dz/dw at value w.
        '''
        return np.vectorize(self.cal_dz)(w)
