import matplotlib.pyplot as plt
%matplotlib inline

import healpy as hp
import numpy as np
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
import os.path
from os.path import exists as file_exists
import pymaster as nmt

z, dndz = np.loadtxt("dndz_bin0.txt", unpack=True)

import pyccl as ccl

#need to add halo model stuff to get Pks


my_tracer = ccl.Tracer()

cosmo = ccl.CosmologyVanillaLCDM()
z_arr = z
print(z_arr)
nz_arr = dndz
print(nz_arr)

# Galaxy clustering
t_g= ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_arr, nz_arr), bias=(z_arr, np.ones_like(z_arr)))
print(t_g)

def try_kernel(alpha,z):
        kernel = (1+z)**alpha
        maybe_this = [z,kernel]
        return maybe_this

notsure = try_kernel(2,z) #kernel needs to be a tuple of 2 arrays. Not sure what the second array should be.
print(notsure)


bpw_edges = [0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3071]
b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
ells = b.get_effective_ells()


UGRB = my_tracer.add_tracer(cosmo, kernel = notsure, transfer_a = None)
cl_theory = ccl.angular_cl(cosmo, UGRB, t_g, ells, p_of_k = p_of_k)
#need p_of_k from halo model 
