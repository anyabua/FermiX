import matplotlib.pyplot as plt
%matplotlib inline
import healpy as hp
import numpy as np
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
import os.path
from os.path import exists as file_exists
import pymaster as nmt
import matplotlib.cm as cm
import pyccl as ccl
from scipy.special import erf




z, dndz = np.loadtxt("dndz_bin1.txt", unpack=True)

cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96, transfer_function="eisenstein_hu")

a = 1./(1+z)

ells = np.geomspace(2,3000,150)
k_arr = np.geomspace(1E-4,100,256)
#k =  (ells + 0.5)/chi
a_arr = (a)[::-1] #makes scale factor monotonically increasing
chi = ccl.comoving_radial_distance(cosmo,a_arr)


########## HALO MODEL STUFF FROM GITHUB ###########

# We will use a mass definition with Delta = 200 times the matter density
hmd_200m = ccl.halos.MassDef200m()

# The Duffy 2008 concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(hmd_200m)

# The Tinker 2008 mass function
nM = ccl.halos.MassFuncTinker08(cosmo, mass_def=hmd_200m)

# The Tinker 2010 halo bias
bM = ccl.halos.HaloBiasTinker10(cosmo, mass_def=hmd_200m)

# The NFW profile to characterize the matter density around halos
pM = ccl.halos.profiles.HaloProfileNFW(cM)

class HaloProfileHOD(ccl.halos.HaloProfileNFW):
            def __init__(self, c_M_relation,lMmin=12.02, lMminp=-1.34, lM0=6.6, lM0p=-1.43, lM1=13.27, lM1p=-0.323):
                        self.lMmin=lMmin
                        self.lMminp=lMminp
                        self.lM0=lM0
                        self.lM0p=lM0p
                        self.lM1=lM1
                        self.lM1p=lM1p
                        self.a0 = 1./(1+0.65)
                        self.sigmaLogM = 0.4
                        self.alpha = 1.
                        super(HaloProfileHOD, self).__init__(c_M_relation)
                        self._fourier = self._fourier_analytic_hod
                            
            def _lMmin(self, a):
                        return self.lMmin + self.lMminp * (a - self.a0)
                                  
            def _lM0(self, a):
                        return self.lM0 + self.lM0p * (a - self.a0)

            def _lM1(self, a):
                        return self.lM1 + self.lM1p * (a - self.a0)

            def _Nc(self, M, a):
                        # Number of centrals
                        Mmin = 10.**self._lMmin(a)
                        return 0.5 * (1 + erf(np.log(M / Mmin) / self.sigmaLogM))

            def _fourier_analytic_hod(self, cosmo, k, M, a, mass_def):
                        M_use = np.atleast_1d(M)
                        k_use = np.atleast_1d(k)
                        
                        Nc = self._Nc(M_use, a)
                        Ns = self._Ns(M_use, a)
                        # NFW profile
                        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

                        prof = Nc[:, None] * (1 + Ns[:, None] * uk)
                        
                        if np.ndim(k) == 0:
                        prof = np.squeeze(prof, axis=-1)
                        if np.ndim(M) == 0:
                        prof = np.squeeze(prof, axis=0)
                        return prof
                                    
            def _fourier_variance(self, cosmo, k, M, a, mass_def):
                        # Fourier-space variance of the HOD profile
                        M_use = np.atleast_1d(M)
                        k_use = np.atleast_1d(k)
                        
                        Nc = self._Nc(M_use, a)
                        Ns = self._Ns(M_use, a)
                        # NFW profile
                        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

                        prof = Ns[:, None] * uk
                        prof = Nc[:, None] * (2 * prof + prof**2)

                        if np.ndim(k) == 0:
                        prof = np.squeeze(prof, axis=-1)
                        if np.ndim(M) == 0:
                        prof = np.squeeze(prof, axis=0)
                        return prof

pg = HaloProfileHOD(cM)
hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd_200m)

#galaxy x galaxy power spectrum

class Profile2ptHOD(ccl.halos.Profile2pt):
            def fourier_2pt(self, prof, cosmo, k, M, a, prof2=None, mass_def=None):
                        return prof._fourier_variance(cosmo, k, M ,a, mass_def)

HOD2pt = Profile2ptHOD()

pk_gg = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, 1.,pg, prof_2pt=HOD2pt, normprof1=True)


pk_ggf = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof_2pt=HOD2pt, normprof1=True, lk_arr= np.log(k_arr), a_arr=a_arr)
                        

######## END OF HALO MODEL STUFF #########





GRB_tracer = ccl.Tracer()


# Galaxy clustering
t_g = ccl.NumberCountsTracer(cosmo, has_rsd = False, dndz=(z,dndz), bias=(z, np.ones_like(z)))

#g_kernel = np.squeeze(t_g.get_kernel(chi))

#print(g_kernel)

g_kernel = ccl.get_density_kernel(cosmo, (z,dndz))


plt.plot(chi,g_kernel[1])
plt.xlabel('$\\chi\\,[{\\rm Mpc}]$',fontsize=14)
plt.ylabel('$q_\\delta(\\chi)$',fontsize=14)
plt.show()

def try_kernel(alpha,chi,z):
            kernel = (1+z)**alpha #missing proportional
            gamma_kernel_and_chi = [chi,kernel]
            return gamma_kernel_and_chi

gamma_kernel = try_kernel(6,chi,z)



plt.plot(gamma_kernel[0],gamma_kernel[1])
plt.xlabel('$\\chi\\,[{\\rm Mpc}]$',fontsize=14)
plt.ylabel('$q_\\gamma(\\chi)$',fontsize=14)
plt.show()

GRB_tracer.add_tracer(cosmo, kernel = gamma_kernel)

cl_theory = ccl.angular_cl(cosmo, GRB_tracer , t_g, ells, p_of_k_a = pk_ggf) #uses Limber approx

plt.figure()
plt.plot(ells, cl_theory, 'y-')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('$C_\\ell$', fontsize=14)
plt.xlabel('$\\ell$')
                                    
                        
