import numpy as np
import scipy.integrate as integrate
from prony_approx import *
from con_map import *

class PronyAC:
    '''
    A universal analytic continuation program working for fermionic, bosonic, diagonal, off-diagonal, discrete, continuous, noiseless and noisy cases.
    '''
    def __init__(self, G_w, w, err = None, optimize = False, symmetry = False, pole_real = False, reduce_pole = True, plane = None, k_max = 999, x_range = [-np.inf, np.inf], y_range = [-np.inf, np.inf]):
        '''
        G_w is a 1-d array containing the Matsubara data, w is the corresponding sampling grid;
        If the error tolerance err is given, the continuation will be carried out in this tolerance;
        else if optimize is True, the tolerance will be chosen to be the presumably optimal one;
        else the tolerance will be chosen to be the last singular value in the exponentially decaying range;
        It is suggested to always provide err (>=1.e-12). For noisy data, err should be fine-tuned to improve the noise resistance and the accuracy of final results;
        symmetry determines whether to preserve the up-down symmetry;
        pole_real determines whether to restrict the poles exactly on the real axis when symmetry is True;
        reduce_pole determines whether to discard the poles whose weights are smaller than the error tolerance;
        plane decides whehter to use the original plane (z plane) or mapped plane (w plane) to compute pole weights;
        k_max is the maximum number of contour integrals;
        Only poles located within the rectangle  x_range[0] < x < x_rang[1], y_range[0] < y < y_range[1] are retained.
        '''
        assert G_w.size == w.size
        assert w[0] >= 0.0
        assert np.linalg.norm(np.diff(np.diff(w)), ord=np.inf) < 1.e-10
        
        #the number of input data points for Prony's approximation must be odd
        N_odd = w.size if w.size % 2 == 1 else w.size - 1
        self.G_w = G_w[:N_odd]
        self.w = w[:N_odd]
        self.optimize = optimize
        self.err = err
        self.symmetry = symmetry
        self.pole_real = pole_real
        self.reduce_pole = reduce_pole
        if plane is not None:
            self.plane = plane
        elif self.symmetry is False:
            self.plane = "z"
        else:
            self.plane = "w"
        assert self.plane in ["z", "w"]
        self.k_max = k_max
        self.x_range = x_range
        self.y_range = y_range
        
        if self.symmetry is False:
            #perform the first Prony's approximation to approximate Matsubara data
            self.p_o = PronyApprox(self.G_w, self.w[0], self.w[-1])
            self.S = self.p_o.S
            if self.err is not None:
                idx = self.p_o.find_idx_with_err(self.err)
                self.p_o.find_v_with_idx(idx)
                self.p_o.find_approx(cutoff=1.0 + 0.5 / self.p_o.N)
            elif self.optimize is False:
                idx = self.p_o.find_idx_with_exp_decay()
                self.p_o.find_v_with_idx(idx)
                self.p_o.find_approx(cutoff=1.0 + 0.5 / self.p_o.N)
            else:
                self.p_o.find_approx_opt(cutoff=1.0 + 0.5 / self.p_o.N)
            self.G_approx = self.p_o.get_value
            self.err_max = self.p_o.err_max
            self.err_ave = self.p_o.err_ave
            #get the corresponding conformal mapping
            w_m = 0.5 * (self.w[0] + self.w[-1])
            dw_h = 0.5 * (self.w[-1] - self.w[0])
            self.con_map = ConMapGeneric(w_m, dw_h)
            #calculate contour integrals
            self.cal_hk_generic(self.G_approx, k_max)
        else:
            #use complex poles to approximate Matsubara data in [1j * w[0], +inf)
            p = PronyAC(G_w, w, optimize=optimize, err=err, k_max=k_max)
            self.S = p.p_o.S
            self.G_approx = lambda x: self.cal_G(x, p.pole_weight, p.pole_location)
            self.const = p.const
            G_w_approx = self.G_approx(self.w)
            self.err_max = np.abs(G_w_approx + self.const - self.G_w).max()
            self.err_ave = np.abs(G_w_approx + self.const - self.G_w).mean()
            #get the corresponding conformal mapping
            self.con_map = ConMapGapless(self.w[0])
            #calculate contour integrals
            self.cal_hk_gapless(self.G_approx, k_max)
        
        #apply the second Prony's approximation to recover poles
        self.find_poles()
        self.cut_pole(self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1])
    
    def cal_hk_generic(self, G_approx, k_max = 999):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err_max
        err = 0.01 * cutoff
        
        self.h_k = np.zeros((k_max,), dtype=np.complex_)
        for k in range(k_max):
            self.h_k[k] = self.cal_hk_generic_indiv(G_approx, k, err)
            if k >= 1:
                if np.abs(self.h_k[k]) < cutoff and np.abs(self.h_k[k - 1]) < cutoff:
                    break
        self.h_k = self.h_k[:(2 * (k // 2) + 1)]
    
    def cal_hk_generic_indiv(self, G_approx, k, err):
        if k % 2 == 0:
            return (1.0j / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_m + self.con_map.dw_h * np.sin(x)), -0.5 * np.pi, 0.5 * np.pi, weight="sin", wvar=k + 1, complex_func=True, epsabs=err, epsrel=err, limit=10000)[0]
        else:
            return (1.0  / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_m + self.con_map.dw_h * np.sin(x)), -0.5 * np.pi, 0.5 * np.pi, weight="cos", wvar=k + 1, complex_func=True, epsabs=err, epsrel=err, limit=10000)[0]
    
    def cal_hk_gapless(self, G_approx, k_max = 999):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err_max
        err = 0.01 * cutoff
        
        self.h_k = np.zeros((k_max,), dtype=np.float_)
        for k in range(k_max):
            self.h_k[k] = self.cal_hk_gapless_indiv(G_approx, k, err)
            if k >= 1:
                if np.abs(self.h_k[k]) < cutoff and np.abs(self.h_k[k - 1]) < cutoff:
                    break
        self.h_k = self.h_k[:(2 * (k // 2) + 1)]
    
    def cal_hk_gapless_indiv(self, G_approx, k, err):
        theta0 = 1.e-6
        if k % 2 == 0:
            return (-2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).imag, theta0, 0.5 * np.pi, weight="sin", wvar=k + 1, epsabs=err, epsrel=err, limit=10000)[0]
        else:
            return (+2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).real, theta0, 0.5 * np.pi, weight="cos", wvar=k + 1, epsabs=err, epsrel=err, limit=10000)[0]
    
    def find_poles(self):
        '''
        Recover poles from contour integrals h_k.
        '''
        #apply the second Prony's method
        self.p_f = PronyApprox(self.h_k)
        if self.err is not None:
            idx_max = self.p_f.find_idx_with_err(self.err_max)
            for idx in np.arange(idx_max, -1, -1):
                self.p_f.find_v_with_idx(idx)
                self.p_f.find_approx()
                if self.p_f.err_max <= 10.0 * self.err_max:
                    break
        elif self.optimize is False:
            idx = self.p_f.find_idx_with_exp_decay()
            self.p_f.find_v_with_idx(idx)
            self.p_f.find_approx()
        else:
            self.p_f.find_approx_opt()
        
        if self.symmetry and self.pole_real:
            self.p_f.gamma = self.p_f.gamma[np.abs(self.p_f.gamma.imag) < 1.e-3]
            self.p_f.find_omega()
        
        #tranform poles from w-plane to z-plane
        location = self.con_map.z(self.p_f.gamma)
        weight = self.p_f.omega * self.con_map.dz(self.p_f.gamma)
        
        if self.symmetry is False:
            G_w_approx = self.cal_G(self.w, weight, location)
            const = (self.G_w - G_w_approx).mean()
            self.const = const if np.abs(const) > 100.0 * self.err_max else 0.0
        
        if self.plane == "z":
            A = np.zeros((self.w.size, location.size), dtype=np.complex_)
            for i in range(location.size):
                A[:, i] = 1.0 / (1j * self.w - location[i])
            weight = np.linalg.pinv(A) @ (self.G_w - self.const)
        
        if self.reduce_pole:
            #discard poles with negligible weights
            idx1 = np.absolute(weight) > self.err_max
            weight   = weight[idx1]
            location = location[idx1]
        
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx2 = np.argsort(location.real)
        self.pole_weight   = weight[idx2]
        self.pole_location = location[idx2]
    
    def cut_pole(self, x_min, x_max, y_min, y_max):
        '''
        Only keep poles located within (x_min, x_max) and (y_min, y_max).
        '''
        idx_x = np.logical_and(self.pole_location.real > x_min, self.pole_location.real < x_max)
        idx_y = np.logical_and(self.pole_location.imag > y_min, self.pole_location.imag < y_max)
        self.pole_weight   = self.pole_weight[np.logical_and(idx_x, idx_y)]
        self.pole_location = self.pole_location[np.logical_and(idx_x, idx_y)]
    
    @staticmethod
    def cal_G(wn, Al, xl):
        assert np.all(wn > 0)
        G_z = np.zeros(wn.shape, dtype=np.complex_)
        for i in range(Al.size):
            G_z += Al[i] / (1j * wn - xl[i])
        return G_z
    
    def check_valid(self):
        import matplotlib.pyplot as plt
        
        #check svd of the first Prony
        plt.figure()
        plt.semilogy(self.S, ".")
        plt.semilogy([0, self.S.size - 1], [self.err_max, self.err_max], color="tab:green")
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\sigma_n$")
        plt.title("SVD of the first Prony")
        plt.show()
        
        #check the first approximation
        G_w1 = self.G_approx(self.w) if self.symmetry is False else self.G_approx(self.w) + self.const
        plt.figure()
        plt.semilogy(self.w, np.abs(G_w1.real - self.G_w.real), ".-" , label="diff real")
        plt.semilogy(self.w, np.abs(G_w1.imag - self.G_w.imag), ".--", label="diff imag")
        plt.semilogy([self.w[0], self.w[-1]], [self.err_max, self.err_max], color="tab:green")
        plt.legend()
        plt.xlabel(r"$\omega_n$")
        plt.ylabel(r"$|\hat{G}(i\omega_n) - G(i\omega_n)|$")
        plt.title("First approximation")
        plt.show()
        
        #check h_k
        #part 1
        plt.figure()
        plt.semilogy(np.abs(self.h_k), '.')
        plt.semilogy([0, self.h_k.size - 1], [self.err_max, self.err_max], color="tab:green")
        plt.xlabel(r"$k$")
        plt.ylabel(r"$h_k$")
        plt.title("Contour integrals: value")
        plt.show()
        #part 2
        plt.figure()
        plt.semilogy(self.p_f.S, ".")
        plt.semilogy([0, self.p_f.S.size - 1], [self.err_max, self.err_max], color="tab:green")
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\sigma_n$")
        plt.title("Contour integrals: SVD")
        plt.show()
        #part 3
        plt.figure()
        plt.semilogy(np.abs(self.p_f.get_value(np.linspace(0, 1, self.h_k.size)) - self.h_k), '.')
        plt.semilogy([0, self.h_k.size - 1], [self.err_max, self.err_max], color="tab:green")
        plt.xlabel(r"$k$")
        plt.ylabel(r"$|\hat{h}_k - h_k|$")
        plt.title("Contour integrals: approximation")
        plt.show()
        
        #check the final approximation
        G_w2 = self.cal_G(self.w, self.pole_weight, self.pole_location) + self.const
        plt.figure()
        plt.semilogy(self.w, np.abs(G_w2.real - self.G_w.real), ".-" , label = "diff real")
        plt.semilogy(self.w, np.abs(G_w2.imag - self.G_w.imag), ".--", label = "diff imag")
        plt.semilogy([self.w[0], self.w[-1]], [self.err_max, self.err_max], color="tab:green")
        plt.legend()
        plt.xlabel(r"$\omega_n$")
        plt.ylabel(r"$|\hat{G}(i\omega_n) - G(i\omega_n)|$")
        plt.title("Final approximation")
        plt.show()
        
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 1, 1), (0, 0, 1)] #(R, G, B) tuples for white and blue
        n_bins = 100 #Discretize the interpolation into bins
        cmap_name = "WtBu"
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins) #Create the colormap
        
        #check pole locations
        pts = self.pole_location
        scatter = plt.scatter(pts.real, pts.imag, c=np.abs(self.pole_weight), vmin=0, vmax=1, cmap=cmap)
        cbar = plt.colorbar(scatter)
        cbar.set_label('weight')
        x_max = np.abs(self.pole_location.real).max() * 1.2
        y_max = max(np.abs(self.pole_location.imag).max() * 1.2, 1.0)
        plt.xlim([-x_max, x_max])
        plt.ylim([-y_max, y_max])
        plt.xlabel(r"Real($z$)")
        plt.ylabel(r"Imag($z$)")
        plt.show()
        
        #check mapped pole locations
        theta = np.arange(1001) * 2.0 * np.pi / 1000
        pts = self.con_map.w(self.pole_location)
        plt.plot(np.cos(theta), np.sin(theta), color="tab:orange")
        scatter = plt.scatter(pts.real, pts.imag, c=np.abs(self.pole_weight), vmin=0, vmax=1, cmap=cmap)
        cbar = plt.colorbar(scatter)
        cbar.set_label('weight')
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])
        plt.xlabel(r"Real($w$)")
        plt.ylabel(r"Imag($w$)")
        plt.show()
