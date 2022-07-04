import numpy as np
import healpy as hp
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
import os.path
from os.path import exists as file_exists 

#inputs (i.e maps and masks) to be used entered here
gammamap_read = hp.readmap(input("Input gamma map"))
overdensity_read =hp.read_map(input("Input overdensity map"))
gammamask_read = hp.read_map(input("Input gamma map mask"))
overdensitymask_read = hp.read_map(input("Input overdensity mask"))
n_jk_regions = input("Input number of regions to apply Jackknife resampling")



#Gives total mask for two masks 
def get_total_mask(mask1,mask2):
    total_mask = mask1*mask2
    return total_mask


mask_full = get_total_mask(gammamask_read, overdensitymask_read)


#get jk_ids (this function divides the mask into jackknife regions and labels them)

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
def check_existence(filename):
    if file_exists(filename) == False: 
        jk_id_num = get_regions(filename, n_jk_regions, unassigned = -1)
        np.savez(filename, jk_id_num = jk_id_num)
    else:
        jk_id_num = np.load(filename)

jk_id = check_existence("...")
    
#function removes the jackknife region from the mask so that we can apply the pseudo CL estimator to the rest of the map
def removing_region(jk_id, label, mask_full): 
    mask_jk = mask_full.copy()
    mask_jk[jk_id == label] = 0
    return mask_jk


msk = hp.ud_grade(mask_full, nside_out = len(mask_full)) 
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
        effectivemask = removing_region(jk_id_num[i], label, mask_full) 
        Cl = calculate_cl(gammamap_read,overdensity_read, effectivemask, effectivemask)
        jkcl.append(Cl)
    return jkcl
    
    
    
PCL_fskydivided = calculate_cl(gammamap_read,overdensity_read, total_mask, total_mask) #This is the Cl with no jackknife regions removed
JKCL = calculate_jkcl(PCL_fskydivided, jk_id) #This is the jackknife Cl, i.e with each jackknife region removed


#with JKCL, we can calculate the jackknife error bars
def calculate_errorbars(JKCL):
    n = len(JKCL)
    mean_Cl = np.mean(JKCL)
    jk_error = np.sqrt((n-1)*np.mean(JKCL-mean_cl)*np.mean(JKCL-mean_cl))
    return jk_error

jack_error = calculate_errorbars(JKCL)



def plot_power_spectrum_theory(filename, cls_data):
    data =[]
    datum = np.loadtxt(filename, unpack=True)
    ls = datum[0]
    cls = datum[1:]
    for i in range(len(cls)):
        fig, ax = plt.subplots()
        ax.plot(ls, cls[i], 'k-')
        ax.plot(ls, cls_data[i], 'r-')
        ax.set_xlabel('l')
        ax.set_ylabel('Cl')
        ax.set_xscale('log')
        ax.set_yscale('log')

def calculate_cl_multi(maps, masks):
    PCL_fsky = []
    nmaps = len(maps)
    for i1, (mp1, mask1) in enumerate(zip(maps, masks)):
        for mp2, mask2 in zip(maps[i1:], masks[i1:]):
            PCL_divided_byfsky = calculate_cl(mp1, mp2, mask1, mask2)
            nside = hp.npix2nside(len(mp1))
            PCL_fsky.append(PCL_divided_byfsky)
    return PCL_fsky