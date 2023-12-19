#!/usr/bin/env python3
import numpy as np
import h5py
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.joinpath("../../src").resolve()))
from green_func import *
from prony_ac import *
from spectrum_example import *

statistics = "F"
beta = 200
N = 1000
dn = 1
N_tot = 2 * N + 1

#read input for square and triangle lattices
systems = ["square_lattice_w_NNN", "triangle_lattice_anisotropic"]
input_data = h5py.File("input_tight_binding.h5", "r")
w_square = np.copy(input_data[systems[0] + "/w"])
G_square = np.copy(input_data[systems[0] + "/G_w"])
w_triangle = np.copy(input_data[systems[1] + "/w"])
G_triangle = np.copy(input_data[systems[1] + "/G_w"])
input_data.close()

#begin simulation
data = h5py.File("versatility_spectrum_data.h5", "w")

#1. Square Lattice with next-nearest-neighbor interaction
n0 = 0
system = "square_lattice_w_NNN"
print(system)

ac = PronyAC(G_square, w_square, optimize=True)

data[system + "/statistics"] = statistics
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = ac.pole_weight
data[system + "/x_i"] = ac.pole_location
print("done!")

#2. Bethe Lattice
n0 = 0
n_sample = np.arange(n0, n0 + N_tot * dn, dn)
w_sample = (2 * n_sample + 1) * np.pi / beta
G_w = bethe_lattice_G(w_sample, t = 1.0)
system = "bethe_lattice"
print(system)

ac = PronyAC(G_w, w_sample, optimize=True)

data[system + "/statistics"] = statistics
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = ac.pole_weight
data[system + "/x_i"] = ac.pole_location
print("done!")

#3. Anisotropic triangular lattice
n0 = 0
system = "triangle_lattice_anisotropic"
print(system)

ac = PronyAC(G_triangle, w_triangle, optimize=True)

data[system + "/statistics"] = statistics
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = ac.pole_weight
data[system + "/x_i"] = ac.pole_location
print("done!")

#4. Mixed
n0 = 10
n_sample = np.arange(n0, n0 + N_tot * dn, dn)

gf_mix1 = GreenFunc(statistics, beta, "continuous", A_x=lambda x: 0.2 * gaussian(x, mu=2.0, sigma=0.5) + 0.2 * gaussian(x, mu=-2.0, sigma=0.5), x_min=-np.inf, x_max=np.inf)
gf_mix2 = GreenFunc(statistics, beta, "discrete", A_i=np.array([0.6]), x_i=np.array([0]))
gf_mix1.get_matsubara(n_sample.max() + 1)
gf_mix2.get_matsubara(n_sample.max() + 1)
w_sample = gf_mix1.w[n_sample]
G_w = gf_mix1.G_w[n_sample] + gf_mix2.G_w[n_sample]
system = "mixed"
print(system)

ac = PronyAC(G_w, w_sample, optimize=True)

data[system + "/statistics"] = statistics
data[system + "/beta"] = beta
data[system + "/N"] = N
data[system + "/n0"] = n0
data[system + "/dn"] = dn
data[system + "/sigma"] = ac.p_o.sigma
data[system + "/A_i"] = ac.pole_weight
data[system + "/x_i"] = ac.pole_location
data.close()
print("done!")
