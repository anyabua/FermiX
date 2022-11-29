import numpy as np
import healpy as hp
import os.path
from os.path import exists as file_exists 
import argparse
import matplotlib.pyplot as plt
import sys
import pymaster as nmt

parser = argparse.ArgumentParser()
parser.add_argument('map_gal', type=str, help='Path to galaxy map')
parser.add_argument('map_gamma', type=str, help='Path to gamma ray map')
parser.add_argument('mask_gal', type=str, help='Path to galaxy mask')
parser.add_argument('mask_gamma', type=str, help='Path to gamma ray mask')
parser.add_argument('--nside', type=int,
                    default=1024, help='Nside to use')
parser.add_argument('--namefile' , type=str, help ='File name to save as')
parser.add_argument('--use_namaster', default=False, action='store_true',
                    help='Use namaster instead of anafast')
args = parser.parse_args()

if args.use_namaster:
    print("Creating bins")
    sys.stdout.flush()
    bpw_edges = [30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073]
    b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
    ells = b.get_effective_ells()
else:
    ells = np.arange(3*args.nside)

#inputs (i.e maps and masks) to be used entered here
print("Reading maps")
sys.stdout.flush()
gammamap_read = hp.ud_grade(np.load(args.map_gamma), nside_out=args.nside)
overdensity_read = hp.ud_grade(hp.read_map(args.map_gal), nside_out=args.nside)
gammamask_read = hp.ud_grade(hp.read_map(args.mask_gamma), nside_out=args.nside)
overdensitymask_read = hp.ud_grade(hp.read_map(args.mask_gal), nside_out=args.nside)


#Gives total mask for two masks 
def get_total_mask(mask1,mask2):
    total_mask = mask1*mask2
    return total_mask


print("Total mask")
sys.stdout.flush()
mask_full = get_total_mask(gammamask_read, overdensitymask_read)
               
#computing the data points i.e calculates Cl
def calculate_cl(mp1, mp2,mask_full,return_bpw=False):
    if args.use_namaster:
        fsky = np.mean(mask_full)
        f_gal = nmt.NmtField(mask_full,[mp1], n_iter=0)
        f_gam = nmt.NmtField(mask_full, [mp2], n_iter=0)
        PCL_galgam = nmt.compute_coupled_cell(f_gal,f_gam)/fsky
        PCL_galgal = nmt.compute_coupled_cell(f_gal,f_gal)/fsky
        PCL_gamgam = nmt.compute_coupled_cell(f_gam,f_gam)/fsky
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f_gal, f_gam, b)
        cl = w.decouple_cell(PCL_galgam)
        if return_bpw:
            bpw = w.get_bandpower_windows()
    else:
        PCL = hp.anafast(mp1*mask_full, mp2*mask_full)
        fsky  = np.mean(mask_full)
        ell = len(PCL)
        cl = PCL/fsky
        if return_bpw:
            bpw = np.eye(3*args.nside)
    if return_bpw:
        return cl, bpw
    else:
        return cl,PCL_galgam, PCL_galgal, PCL_gamgam, f_gal,f_gam

def calculate_gausscov(f_gal,f_gam,PCL_galgam,PCL_galgal,PCL_gamgam):
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f_gal,f_gam,b)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f_gal,f_gam,flb1 = None,flb2 = None)
    full_cov = nmt.gaussian_covariance(cw,0,0,0,0,[PCL_galgal],[PCL_galgam],[PCL_galgam],[PCL_gamgam],w,wb = w, coupled =True).reshape([ells,1,ells,1])
    return full_cov
        

PCL_fskydivided,PCL_galgam,PCL_galgal, PCL_gamgam, f_gal, f_gam = calculate_cl(gammamap_read,overdensity_read,mask_full, return_bpw=False) #This is the Cl with no jackknife regions removed
print(type(f_gal))
gauss_cov = calculate_gausscov(f_gal,f_gam,PCL_galgam,PCL_galgal,PCL_gamgam)

print("Gaussian Covariance calculated")
sys.stdout.flush()
name = args.namefile
filename = "%s.npz" % name
np.savez(filename, ells=ells, PCL_fskydivided = PCL_fskydivided, gauss_cov = gauss_cov,bpw=bpw)

exit(1)


