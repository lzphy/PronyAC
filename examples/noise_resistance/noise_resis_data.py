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

input_data = h5py.File("input_" + singular_type + ".h5", "r")
noise_list = []
w_list = []
G_w_list = []
for i in range(5):
    noise_list.append(np.copy(input_data[str(i) + "/noise"]))
    w_list.append(np.copy(input_data[str(i) + "/w"]))
    G_w_list.append(np.copy(input_data[str(i) + "/G_w"]))
input_data.close()

statistics = 'F'
beta = 200
N = 1000
A_i = np.array([0.6, 0.4])
x_i = np.array([-1, 1])
if singular_type == "discrete":
    gf = GreenFunc(statistics, beta, "discrete", A_i=A_i, x_i=x_i)
    n0 = 0
else:
    gf = GreenFunc(statistics, beta, "continuous", A_x=lambda x: 0.5 * lorentzian(x, w0=0, half_width=0.5) + 0.3 * lorentzian(x, w0=-2.5, half_width=0.8) + 0.2 * gaussian(x, mu=2.5, sigma=0.8), x_min=-np.inf, x_max=np.inf)
    n0 = 10

dn = 1
N_tot = 2 * N + 1
n_sample = np.arange(n0, n0 + N_tot * dn, dn)
gf.get_matsubara(n_sample.max() + 1)

data = h5py.File("noise_resis_" + singular_type + ".h5", "w")
data["statistics"] = statistics
data["beta"] = beta
data["N"] = N
data["n0"] = n0
data["dn"] = dn

count = 0
for i in range(5):
    noise = noise_list[i]
    print("noise = ", noise)
    w = w_list[i]
    G_w = G_w_list[i]

    ac = PronyAC(G_w[n_sample], w[n_sample], optimize=True)

    if singular_type == "discrete":
        pole_weight = ac.pole_weight.real
        pole_location = ac.pole_location.real
    else:
        pole_weight = ac.pole_weight
        pole_location = ac.pole_location
    data[str(count) + "/noise"] = noise
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
