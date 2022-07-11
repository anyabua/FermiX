
import pyccl as ccl

z, dndz = np.loadtxt("dndz_bin1.txt", unpack=True)

#bpw_edges = [0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3071]
#b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
#ells = b.get_effective_ells()

#print(len(ells))
ells = np.geomspace(2,1000,150)

#need to add halo model stuff to get Pks
cosmo = ccl.CosmologyVanillaLCDM()
#mass_def = ccl.halos.MassDef(200, 'critical')
#hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)
#hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mass_def)
#cm = ccl.halos.ConcentrationDuffy08(mass_def)
#hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mass_def)

#g_prof = ccl.halos.HaloProfileHOD(cm)

#class Profile2ptHOD(ccl.halos.Profile2pt):
            #def fourier_2pt(self, prof, cosmo, k, M, a,prof2=None, mass_def=None):
                        #return prof._fourier_variance(cosmo, k, M ,a, mass_def)
    
#HOD2pt = Profile2ptHOD()

#galaxy x galaxy power spectrum
#pk_gg = ccl.halos.halomod_power_spectrum(cosmo, hmc, k, 1.,pg, prof_2pt=HOD2pt,normprof1=True)


my_tracer = ccl.Tracer()

a = 1./(1+z)
#print(a)
chi = ccl.comoving_radial_distance(cosmo,a)
#k_arr = (ells+0.5)/chi

# Galaxy clustering
t_g = ccl.NumberCountsTracer(cosmo, has_rsd = False, dndz=(z,dndz), bias=(z, np.ones_like(z)))

#g_kernel = np.squeeze(t_g.get_kernel(chi))

#print(g_kernel)
g_kernel = ccl.get_density_kernel(cosmo, (z,dndz)) 


plt.plot(chi,g_kernel[1])
plt.xlabel('$\\chi\\,[{\\rm Mpc}]$',fontsize=14)
plt.ylabel('$q(\\chi)$',fontsize=14)
plt.show()

def try_kernel(alpha,chi,z):
            kernel = (1+z)**alpha #missing proportionality constants
            gamma_kernel_and_chi = [chi,kernel]
            return gamma_kernel_and_chi

gamma_kernel = try_kernel(3,chi,z) #kernel needs to be a tuple of 2 arrays. Not sure what the second array should be.
#print(notsure)

            
plt.plot(notsure[0],notsure[1])
plt.xlabel('$\\chi\\,[{\\rm Mpc}]$',fontsize=14)
plt.ylabel('$q(\\chi)$',fontsize=14)
plt.show()

UGRB = my_tracer.add_tracer(cosmo, kernel = gamma_kernel, transfer_a = None)
cl_theory = ccl.angular_cl(cosmo, UGRB, g_kernel, ells, p_of_k = pk_gg)
#need p_of_k from halo model 
