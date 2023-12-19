#!/usr/bin/env python3
import numpy as np
import argparse
import h5py
import scipy.integrate as integrate
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.joinpath("../../src").resolve()))
from green_func import *
from prony_ac import *
from spectrum_example import *

#########################################
#input parameter
parser = argparse.ArgumentParser()
parser.add_argument("--system", type=str, default="discrete")
args = parser.parse_args()
singular_type = args.system.lower()
assert singular_type in ["discrete", "continuous"]
#########################################

statistics = 'F'
beta = 200
N = 1000
A_i = np.array([0.52, 0.48])
x_i = np.array([0, 1])
if singular_type == "discrete":
    gf = GreenFunc(statistics, beta, "discrete", A_i=A_i, x_i=x_i)
    n0 = 30
    err_list = [2.554e-4, 7.984e-6, 1.834e-7, 5.754e-9, 1.604e-10, 8.264e-13]
else:
    gf = GreenFunc(statistics, beta, "continuous", A_x=lambda x: 0.3 * gaussian(x, mu=0.0, sigma=0.2) + 0.5 * gaussian(x, mu=1.0, sigma=0.5) + 0.2 * gaussian(x, mu=-2.0, sigma=0.5), x_min=-np.inf, x_max=np.inf)
    n0 = 0
    err_list = [2.364e-3, 1.894e-5, 7.704e-7, 1.044e-9, 3.544e-11, 2.144e-13]
dn = 1
N_tot = 2 * N + 1
n_sample = np.arange(n0, n0 + N_tot * dn, dn)

gf.get_matsubara(n_sample.max() + 1)
w   = gf.w[n_sample]
G_w = gf.G_w[n_sample]

data = h5py.File("err_ctrl_" + singular_type + ".h5", "w")
data["statistics"] = statistics
data["beta"] = beta
data["N"] = N
data["n0"] = n0
data["dn"] = dn

count = 0
for err in err_list:
    print("error tolerance = ", "{:.2e}".format(err))
    ac = PronyAC(G_w, w, optimize=False, err=err)
    
    if singular_type == "discrete":
        pole_weight = ac.pole_weight.real
        pole_location = ac.pole_location.real
    else:
        pole_weight = ac.pole_weight
        pole_location = ac.pole_location
    data[str(count) + "/sigma"] = ac.p_o.sigma
    data[str(count) + "/A_i"] = pole_weight
    data[str(count) + "/x_i"] = pole_location
    gf2 = GreenFunc(statistics, beta, "discrete", A_i=pole_weight, x_i=pole_location)
    x0 = np.linspace(-10, 10, 1000000)
    y0 = np.abs(gf2.get_spectral(x0, epsilon = 0.01) - gf.get_spectral(x0,  epsilon = 0.01))
    err = integrate.trapezoid(y0, x0)
    data[str(count) + "/err"] = err
    print("done!")
    count += 1
data["count"] = count
data.close()
