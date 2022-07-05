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
parser.add_argument('map_gal', type=str, help='Path to galaxy map')
parser.add_argument('map_gamma', type=str, help='Path to gamma ray map')
parser.add_argument('mask_gal', type=str, help='Path to galaxy mask')
parser.add_argument('mask_gamma', type=str, help='Path to gamma ray mask')
parser.add_argument('--nside', type=int,
                    default=1024, help='Nside to use')
parser.add_argument('--regions_name', type=str, default='None',
                    help='Nside to use')
parser.add_argument('--njk', type=int,
                    default=100, help='# JK regions')
parser.add_argument('--namefile' , type=str, help ='File name to save as')
args = parser.parse_args()


#inputs (i.e maps and masks) to be used entered here
print("Reading maps")
sys.stdout.flush()
gammamap_read = hp.ud_grade(np.load(args.map_gamma), nside_out=args.nside)
overdensity_read = hp.ud_grade(hp.read_map(args.map_gal), nside_out=args.nside)
gammamask_read = hp.ud_grade(hp.read_map(args.mask_gamma), nside_out=args.nside)
overdensitymask_read = hp.ud_grade(hp.read_map(args.mask_gal), nside_out=args.nside)
n_jk_regions = args.njk


#Gives total mask for two masks 
def get_total_mask(mask1,mask2):
    total_mask = mask1*mask2
    return total_mask


print("Total mask")
sys.stdout.flush()
mask_full = get_total_mask(gammamask_read, overdensitymask_read)

def get_regions(mask, n_regions, unassigned=hp.UNSEEN):
    npix = len(mask)
    nside = hp.npix2nside(npix)
    ipix = np.arange(npix) #integers for pixels
    ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
    goodpix = mask > 0
    km = kmeans_sample(np.array([ra[goodpix], dec[goodpix]]).T,
                       n_regions, maxiter=100, tol=1.0e-5,
                       verbose=False)
    map_ids = np.full(npix, unassigned)
    map_ids[ipix[goodpix]] = km.labels 
    return map_ids

#unassigned are setting the unseen pixels in the mask, sets it equal to -1

#check whether the jackknife region mask already exists
def get_jk_ids(mask, filename):
    if file_exists(filename):
        data = np.load(filename)
        jk_id_num = data['jk_id_num']
    else:
        jk_id_num = get_regions(mask, n_jk_regions, unassigned = -1)
        if filename != 'None':
            np.savez(filename, jk_id_num = jk_id_num)
    return jk_id_num


print("Jackknife IDs")
sys.stdout.flush()
jk_id = get_jk_ids(mask_full, args.regions_name)

print("JackKnife IDs calculated")
#function removes the jackknife region from the mask so that we can apply the pseudo CL estimator to the rest of the map
def removing_region(jk_id, label, mask_full): 
    mask_jk = mask_full.copy()
    mask_jk[jk_id == label] = 0
    return mask_jk


#need the resolution to be the same because then mask_jk[jk_id_num==label]= 0 won't be 1-to-1

               
#computing the data points i.e calculates Cl
def calculate_cl(mp1, mp2, msk1, msk2):
    PCL = hp.anafast(mp1*msk1, mp2*msk2)
    fsky  = np.mean(msk1*msk2)
    ell = len(PCL)
    return PCL/fsky

#calculates Cl with the desired jackknife region removed                      
def calculate_jkcl(PCL_fskydivided,jk_id):
    jkcl = []
    for i in range(n_jk_regions):
        print(i, n_jk_regions)
        effectivemask = removing_region(jk_id,i, mask_full)
        print(len(effectivemask))
        Cl = calculate_cl(gammamap_read,overdensity_read, effectivemask, effectivemask)
        jkcl.append(Cl)
    return jkcl
        

PCL_fskydivided = calculate_cl(gammamap_read,overdensity_read, mask_full, mask_full) #This is the Cl with no jackknife regions removed
JKCL = calculate_jkcl(PCL_fskydivided, jk_id) #This is the jackknife Cl, i.e with each jackknife region removed
print("JKCL Calculated")
sys.stdout.flush()
#with JKCL, we can calculate the jackknife error bars
def calculate_errorbars(JKCL):
    n = len(JKCL)
    mean_Cl = np.mean(JKCL, axis=0)
    jk_error = np.sqrt((n-1)*np.mean((JKCL-mean_Cl)**2, axis=0))
    return jk_error

jack_error = calculate_errorbars(JKCL)

print("Jackknife errors calculated")
sys.stdout.flush()
name = args.namefile
filename = "%s.npz" % name
np.savez(filename,JKCL = JKCL, PCL_fskydivided = PCL_fskydivided, jack_error = jack_error)

exit(1)


