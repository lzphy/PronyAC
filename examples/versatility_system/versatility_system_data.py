#!/usr/bin/env python3
import numpy as np
import h5py
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.joinpath("../../src").resolve()))
from green_func import *
from prony_ac import *
from spectrum_example import *

beta = 200
N = 1000
dn = 1
N_tot = 2 * N + 1

data = h5py.File("versatility_system_data.h5", "w")

#Fer off-diag continuous
print("Fer off-diag continuous")
statistics = "F"
singular_type = "continuous"
system = statistics + "_" + singular_type
n0 = 0
n_sample = np.arange(n0, n0 + N_tot * dn, dn)
gf_fer_offc = GreenFunc(statistics, beta, singular_type, A_x=lambda x: 0.5 * gaussian(x, mu=3.0, sigma=0.5) - 0.1 * gaussian(x, mu=1.0, sigma=1.0) - 0.5 * gaussian(x, mu=-3.0, sigma=0.5) + 0.1 * gaussian(x, mu=-1.0, sigma=1.0), x_min=-np.inf, x_max=np.inf)
gf_fer_offc.get_matsubara(n_sample.max() + 1)
w_fer_offc   = gf_fer_offc.w[n_sample]
G_w_fer_offc = gf_fer_offc.G_w[n_sample]
ac = PronyAC(G_w_fer_offc, w_fer_offc, optimize=False, err=1.e-12)

if singular_type == "discrete":
    pole_weight = ac.pole_weight.real
    pole_location = ac.pole_location.real
else:
    pole_weight = ac.pole_weight
    pole_location = ac.pole_location
data[system + "/statistics"] = statistics
data[system + "/singular_type"] = singular_type
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = pole_weight
data[system + "/x_i"] = pole_location
print("done!")

#Fer off-diag discrete
print("Fer off-diag discrete")
statistics = "F"
singular_type = "discrete"
system = statistics + "_" + singular_type
A_i = np.array([-0.06842167,  0.30803739,  0.09215373, -0.28216683, -0.09213708, 0.04253446])
x_i = np.array([-4.63614978, -2.60745562,  0.86189661,  1.61810008,  3.33810339, 4.02550532])
n0 = 0
n_sample = np.arange(n0, n0 + N_tot * dn, dn)
gf_fer_offd = GreenFunc(statistics, beta, singular_type, A_i=A_i, x_i=x_i)
gf_fer_offd.get_matsubara(n_sample.max() + 1)
w_fer_offd   = gf_fer_offd.w[n_sample]
G_w_fer_offd = gf_fer_offd.G_w[n_sample]
ac = PronyAC(G_w_fer_offd, w_fer_offd, optimize=False, err=1.e-12)

if singular_type == "discrete":
    pole_weight = ac.pole_weight.real
    pole_location = ac.pole_location.real
else:
    pole_weight = ac.pole_weight
    pole_location = ac.pole_location
data[system + "/statistics"] = statistics
data[system + "/singular_type"] = singular_type
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = pole_weight
data[system + "/x_i"] = pole_location
print("done!")

#Bos diag continuous
print("Bos diag continuous")
statistics = "B"
singular_type = "continuous"
system = statistics + "_" + singular_type
n0 = 0
n_sample = np.arange(n0, n0 + N_tot * dn, dn)
gf_bos_diagc = GreenFunc(statistics, beta, singular_type, A_x=lambda x: 0.5 * gaussian(x, mu=1.0, sigma=1) - 0.5 * gaussian(x, mu=-1.0, sigma=1), x_min=-np.inf, x_max=np.inf)
gf_bos_diagc.get_matsubara(n_sample.max() + 1)
w_bos_diagc   = gf_bos_diagc.w[n_sample]
G_w_bos_diagc = gf_bos_diagc.G_w[n_sample]
ac = PronyAC(G_w_bos_diagc, w_bos_diagc, optimize=False, err=1.e-12)

if singular_type == "discrete":
    pole_weight = ac.pole_weight.real
    pole_location = ac.pole_location.real
else:
    pole_weight = ac.pole_weight
    pole_location = ac.pole_location
data[system + "/statistics"] = statistics
data[system + "/singular_type"] = singular_type
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = pole_weight
data[system + "/x_i"] = pole_location
print("done!")

#Bos offdiag discrete
print("Bos offdiag discrete")
statistics = "B"
singular_type = "discrete"
system = statistics + "_" + singular_type
A_i = np.array([-0.1, -0.3, 0.2, 0.4, -0.4, -0.2, 0.3, 0.1])
x_i = np.array([-3, -2, -1, -0.02, 0.02, 1, 2, 3])
n0 = 10
n_sample = np.arange(n0, n0 + N_tot * dn, dn)
gf_bos_offd = GreenFunc(statistics, beta, singular_type, A_i=A_i, x_i=x_i)
gf_bos_offd.get_matsubara(n_sample.max() + 1)
w_bos_offd   = gf_bos_offd.w[n_sample]
G_w_bos_offd = gf_bos_offd.G_w[n_sample]
ac = PronyAC(G_w_bos_offd, w_bos_offd, optimize=False, err=1.e-12)

if singular_type == "discrete":
    pole_weight = ac.pole_weight.real
    pole_location = ac.pole_location.real
else:
    pole_weight = ac.pole_weight
    pole_location = ac.pole_location
data[system + "/statistics"] = statistics
data[system + "/singular_type"] = singular_type
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = pole_weight
data[system + "/x_i"] = pole_location
data.close()
print("done!")
