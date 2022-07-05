import numpy as np
import healpy as hp
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
import os.path
from os.path import exists as file_exists
import argparse
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser()

parser.add_argument('map_gal', type = str, help = 'Path to galaxy map')
parser.add_argument('map_gamma', type = str, help = 'Path to gamma ray map')
parser.add_argument('mask_gal', type = str, help = 'Path to galaxy mask')
parser.add_argument('mask_gamma', type = str, help = 'Path to gamma ray mask')
parser.add_argument('--nside', type = int, default = 1024, help = 'Nside to use')
parser.add_argument('--regions_name', type = str,default = 'None' ,help = 'Name JK regions file')
parser.add_argument('--njk', type = int,default = 100,help = '# JK regions')
parser.add_argument('--namefile', type = str, help = 'Filename to save data as')
args = parser.parse_args()

print("Reading maps")
sys.stdout.flush()

gammamap_read = hp.ud_grade(np.load(args.map_gamma), nside_out =args.nside)
overdensity_read = hp.ud_grade(hp.read_map(args.map_gal), nside_out =args.nside)
gammamask_read = hp.ud_grade(hp.read_map(args.mask_gamma), nside_out =args.nside)
overdensitymask_read = hp.ud_grade(hp.read_map(args.mask_gal), nside_out =args.nside)

#apodize masks? to smoothing it out? 1 degree
apdzd_galmask = nmt.mask_apodization(gammamask_read, 1. , apotype = "Smooth")
apdzd_gammask = nmt.mask_apodization(overdensitymask_read, 1., apotype = "Smooth")

#nmt does mask*map?
f_gal = nmt.NmtField(apdzd_galmask, overdensity_read, spin = None)
f_gam = nmt.NmtField(apdzd_gammask, gammamap_read, spin None)

fsky = np.mean(apdzd_galmask*apdzd_gammask)
ell_per_band = 1/fsky
#ell_per_band i.e second argument of nmtBin.from_nside_linear is the correlation length?
b = nmt.NmtBin.from_nside_linear(args.nside, ell_per_band)

#computes Cl (doing hp.anafast)?
cl_galagam = nmt.compute_full_namaster(f_gal,f_gam, b)





