import numpy as np
import healpy as hp 
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
args = parser.parse_args()


print("Creating bins")
sys.stdout.flush()
bpw_edges = [30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073]
b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
ells = b.get_effective_ells()


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
               

def calculate_cl(mp1, mp2,mask1,mask2):
    fsky = np.mean(mask1*mask2)
    f_gal = nmt.NmtField(mask1,[mp1], n_iter=0)
    f_gam = nmt.NmtField(mask2, [mp2], n_iter=0)
    PCL_galgam = nmt.compute_coupled_cell(f_gal,f_gam)/fsky
    PCL_galgal = nmt.compute_coupled_cell(f_gal,f_gal)/fsky
    PCL_gamgam = nmt.compute_coupled_cell(f_gam,f_gam)/fsky
    return PCL_galgam, PCL_galgal, PCL_gamgam, f_gal,f_gam

#calculating the gaussian covariance. Create two workspaces where one stores the coupling coefficients of the two fields and the other computes the mode coupling matrix. 
def calculate_gausscov(f_gal,f_gam,PCL_galgam,PCL_galgal,PCL_gamgam):
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f_gal,f_gam,b)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f_gal,f_gam,flb1 = None,flb2 = None,lmax = (3*args.nside)-1)
    gauss_cov = nmt.gaussian_covariance(cw,0,0,0,0,PCL_galgal,PCL_galgam,PCL_galgam,PCL_gamgam,w,wb = w)
    return gauss_cov
        

PCL_galgam,PCL_galgal, PCL_gamgam, f_gal, f_gam = calculate_cl(overdensity_read,gammamap_read,mask_full,mask_full) 
print("Gaussian Covariance")
gauss_cov = calculate_gausscov(f_gal,f_gam,PCL_galgam,PCL_galgal,PCL_gamgam)
print("Gaussian Covariance calculated")
sys.stdout.flush()
name = args.namefile
filename = "%s.npz" % name
np.savez(filename, ells=ells,PCL_galgam = PCL_galgam, PCL_galgal = PCL_galgal,PCL_gamgam = PCL_gamgam, gauss_cov = gauss_cov)

exit(1)


