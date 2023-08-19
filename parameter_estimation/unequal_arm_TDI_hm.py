# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys, os, math,copy,corner
from time import *
path ='/disk1/home/wrjx'
sys.path.append(path)
from LitePIG.signal.gensignal import *
from LitePIG.noise.TDInoise import *

#sys.path.append('/disk1/home/wangrj/LDC/MLDC-master/software/LDCpipeline/scripts')
from pycbc import types,fft,frame
from pycbc.filter import highpass,lowpass_fir, matched_filter, matched_filter_core,sigmasq
from pycbc.psd import welch, interpolate
from pycbc.psd.read import from_numpy_arrays
from pycbc.conversions import q_from_mass1_mass2,mchirp_from_mass1_mass2,primary_mass,secondary_mass
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q
from pycbc.distributions import Uniform, JointDistribution, SinAngle,UniformAngle,UniformLog10,CosAngle
from pycbc.inference import  sampler

import Cosmology
import LISAConstants as LC
from pyFDresponse import *

from matplotlib.ticker import LinearLocator
from astropy import units as u
from astropy.coordinates import SkyCoord



# # Cartesian SSB components of spacecraft positions
# p1 = np.fromfile('/disk1/home/wangrj/higherorder/signal/LISA_orbit_data/SCP1_1yr.dat',sep=' ')
# p2 = np.fromfile('/disk1/home/wangrj/higherorder/signal/LISA_orbit_data/SCP2_1yr.dat',sep=' ')
# p3 = np.fromfile('/disk1/home/wangrj/higherorder/signal/LISA_orbit_data/SCP3_1yr.dat',sep=' ')
# p1= p1.reshape(int(len(p1)/3),3)
# p2= p2.reshape(int(len(p2)/3),3)
# p3= p3.reshape(int(len(p3)/3),3)

# c1= SkyCoord(x=p1[:,0],y=p1[:,1],z=p1[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
# c2= SkyCoord(x=p2[:,0],y=p2[:,1],z=p2[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
# c3= SkyCoord(x=p3[:,0],y=p3[:,1],z=p3[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
# #print(c1)
# #transfrom to Ecliptic system of coordinate
# p1_SSB= c1.heliocentrictrueecliptic
# p2_SSB= c2.heliocentrictrueecliptic
# p3_SSB= c3.heliocentrictrueecliptic
# p1_SSB.representation_type='cartesian'
# p2_SSB.representation_type='cartesian'
# p3_SSB.representation_type='cartesian'

# trajdict_numerical_orbits = {
#     'funcp0':funcp0_SSB,
#     'funcp': funcp_SSB,
#     'funcL': funcL_SSB,
#     'p1_SSB': p1_SSB,
#     'p2_SSB': p2_SSB,
#     'p3_SSB': p3_SSB,
#     }








#generate the template waveform
##MBHBs parameters
z=1
m1s=2e6
m2s=2e5

#dimensionless spins: a=cos(theta_s)*s
chi1 = 0.0
chi2 = 0.0


#t0= np.random.uniform(0.0,5.0)
t0=0.05
#Ecliptic Longitude, Latitude
theta=1.0
phi=1.5
lambd =  phi 
beta = np.pi/2 -theta
print('lambd,beta',lambd,beta)

inc = 0.5
DL = Cosmology.DL(z, w=0)[0] # in Mpc
print('Dl,inc',DL,inc)

#phi0 = np.random.uniform(0.0, 2.0*np.pi)
#psi = np.random.uniform(0.0, 2.0*np.pi)   #pols=psi
psi=0.8
phi0=0.0
print('t0,phi0,psi',t0,phi0,psi)

# Equal arm orbit: MLCD trajectories 
trajdict = trajdict_MLDC 
# The first-generatrion TDI 
TDItag =  "TDIAET" 


#Masses quoted in the hdf5 files are always redshifted masses.
# m1 =  m1s*(1+z)  ### redshifted masses
# m2 =  m2s*(1+z)
m1=m1s
m2=m2s

chirpmass=mchirp_from_mass1_mass2(m1,m2)
q=q_from_mass1_mass2(m1,m2)
print('chirp mass, q ',chirpmass,q)

apx=['SEOBNRv4HM','IMRPhenomHM']
modes=[[[2,2]],[[2,2],[2,1],[3,2],[3,3],[4,4]]] 
#################################################################

#analytic PSD of noise for TDI A, E and T
del_f= 1e-6
f= np.arange(1e-5,1.0,del_f)
flen = int(1.0/del_f)+1
flow=1e-5
# #freq= logsampling(1e-5,1.0,200)
# PSD_TDIT= noisepsd_T(f)
# PSD_TDIAE= noisepsd_AE(f)
# PSD_TDIT = from_numpy_arrays(f, PSD_TDIT, flen, del_f,flow)
# PSD_TDIAE = from_numpy_arrays(f, PSD_TDIAE, flen, del_f,flow)
# print(PSD_TDIAE.sample_frequencies)
# plt.loglog(PSD_TDIT.sample_frequencies,np.sqrt(PSD_TDIT*PSD_TDIT.sample_frequencies),label='T')
# plt.loglog(PSD_TDIAE.sample_frequencies,np.sqrt(PSD_TDIAE*PSD_TDIAE.sample_frequencies),label='A,E')
# plt.xlabel('freq')
# plt.ylabel('strain')
# plt.xlim(1e-5,1e-1)
# plt.legend()
# plt.show


#PSD functions for a static unequal arm
tmp_year= t0* YRSID_SI
tmp_p0 = funcp0_SSB(tmp_year,p1_SSB=p1_SSB,p2_SSB=p2_SSB,p3_SSB=p3_SSB)
tmp_p1= funcp_SSB(tmp_year,p1_SSB)*R_SI
tmp_p2= funcp_SSB(tmp_year,p2_SSB)*R_SI
tmp_p3= funcp_SSB(tmp_year,p3_SSB)*R_SI

L12= funcL_SSB(tmp_p1,tmp_p2)
L23= funcL_SSB(tmp_p2,tmp_p3)
L13= funcL_SSB(tmp_p1,tmp_p3)
print('armlength',L12,L23,L13)


PSD_A_LISA_uneq = noisepsd_unequal_A(f,L12,L23,L13,'LISA')
PSD_E_LISA_uneq = noisepsd_unequal_E(f,L12,L23,L13,'LISA')
PSD_T_LISA_uneq = noisepsd_unequal_T(f,L12,L23,L13,'LISA')
PSD_TDIA = from_numpy_arrays(f, PSD_A_LISA_uneq , flen, del_f,flow)
PSD_TDIE = from_numpy_arrays(f, PSD_E_LISA_uneq , flen, del_f,flow)
PSD_TDIT = from_numpy_arrays(f, PSD_T_LISA_uneq , flen, del_f,flow)

# PSD_A_LISA_eq = noisepsd_unequal_A(f,L_SI,L_SI,L_SI,'LISA')
# PSD_E_LISA_eq = noisepsd_unequal_E(f,L_SI,L_SI,L_SI,'LISA')
# PSD_T_LISA_eq = noisepsd_unequal_T(f,L_SI,L_SI,L_SI,'LISA')
# PSD_TDIA = from_numpy_arrays(f, PSD_A_LISA_eq , flen, del_f,flow)
# PSD_TDIE = from_numpy_arrays(f, PSD_E_LISA_eq , flen, del_f,flow)
# PSD_TDIT = from_numpy_arrays(f, PSD_T_LISA_eq , flen, del_f,flow)

##############################################################
#generate TDI data
# freqs, A,E,T =get_TDI(chirpmass,q,DL,inc,phi0,chi1,chi2,
#                     lambd,beta,psi,t0,trajdict,TDItag,apx[1],modes[1])
# freqs, A,E,T = get_FR_TDI(chirpmass,q,DL,inc,phi0,chi1,chi2,lambd,beta,psi,t0=t0,apx=apx[1],modes=modes[1],
#                           trajdict=trajdict_numerical_orbits,TDItag='TDIAET',number_orbits=True,TDI_froze_arm=False,frozenLISA=True,df=2e-6,del_t=1.0)


# plt.loglog(freqs,np.abs(A*freqs),label='unequal_arm TDIA')
# plt.loglog(freqs,np.abs(E*freqs),label='unequal_arm TDIE')
# plt.loglog(freqs,np.abs(T*freqs),label='unequal_arm TDIT')
# plt.loglog(PSD_TDIT.sample_frequencies,np.sqrt(PSD_TDIT*PSD_TDIT.sample_frequencies),label='PSD T LISA')
# plt.loglog(PSD_TDIA.sample_frequencies,np.sqrt(PSD_TDIA*PSD_TDIA.sample_frequencies),label='PSD A LISA')
# plt.loglog(PSD_TDIE.sample_frequencies,np.sqrt(PSD_TDIE*PSD_TDIE.sample_frequencies),label='PSD E LISA')
# plt.xlabel('freq')
# plt.ylabel('strain')
# plt.legend()
# plt.xlim(1e-5,1e-0)
# plt.savefig('TDI_AET.png',dpi=300)
# #plt.show()
# plt.clf()

# TDIA= types.FrequencySeries(A,delta_f=freqs[1]-freqs[0])
# TDIE= types.FrequencySeries(E,delta_f=freqs[1]-freqs[0])
# TDIT= types.FrequencySeries(T,delta_f=freqs[1]-freqs[0])
#####################################################################
#read the strain data
num=0
dataA = frame.read_frame('FRTDI/unequal_arm/HM/%d/strainA.gwf'%num,'LISA')
dataE = frame.read_frame('FRTDI/unequal_arm/HM/%d/strainE.gwf'%num,'LISA')
dataT = frame.read_frame('FRTDI/unequal_arm/HM/%d/strainT.gwf'%num,'LISA')
# print('data dt and df',dataA.delta_t,dataA.delta_f,1.0/dataA.duration)

df = 2e-6
tlen =int(1.0 / dataA.delta_t / df)
tstart=int(t0*YRSID_SI *dataA.delta_t)
print(tlen,tstart)
#########Limit to times around the signal
dataA1 = dataA.time_slice(tstart*dataA.delta_t,tstart*dataA.delta_t+tlen*dataA.delta_t)
dataE1 = dataE.time_slice(tstart*dataE.delta_t,tstart*dataE.delta_t+tlen*dataE.delta_t)
dataT1 = dataT.time_slice(tstart*dataT.delta_t,tstart*dataT.delta_t+tlen*dataT.delta_t)
print('limit to times: data dt and df',dataA1.delta_t,dataA1.delta_f,1.0/dataA1.duration) 

TDIA =dataA1.to_frequencyseries()   # Convert to a frequency series by taking the data's FFT
TDIE =dataE1.to_frequencyseries()
TDIT =dataT1.to_frequencyseries()


print('delta_f',TDIA.delta_f)
print(TDIA.sample_frequencies)


data={}
data['LISATDI1'] = TDIA
data['LISATDI2'] = TDIE
# data['LISATDI3'] = TDIT

psds={}
psdA = interpolate(PSD_TDIA,TDIA.delta_f)
psdE = interpolate(PSD_TDIE,TDIE.delta_f)
psdT = interpolate(PSD_TDIT,TDIT.delta_f)
psds['LISATDI1'] = psdA
psds['LISATDI2'] = psdE
# psds['LISATDI3'] = psdT




start=time()
static = {#'chirpmass':chirpmass,
          #'q':q,
          #'distance':DL,
          #'inc':inc,
          'phi0':phi0,
          'chi1':chi1,
          'chi2':chi2,
          'lambd':lambd,
          'beta':beta,
          'psi':psi,
          't0':t0,
          'trajdict':trajdict_numerical_orbits,
          'number_orbits':True,
          'TDI_froze_arm':False,
          'frozenLISA':True,
          'TDItag':TDItag,
          'apx':apx[1],
          'modes':modes[1]
         }
variable = [
            #'m1',
            #'m2',
            'chirpmass',
            'q',
            'distance',
            'inc',
            #'lambd',
            #'beta'
            #'t0'
            ]
inclination_prior = SinAngle(inc=None)
# inclination_prior = Uniform(inc=(0.0,np.pi/2))
distance_prior = Uniform(distance=(DL*0.5, DL*1.5))
lambd_prior= Uniform(lambd=(0.0,2*np.pi))
# lambd_prior= Uniform(lambd=(0.0,3.0))
beta_prior= CosAngle(beta=None)
t0_prior = Uniform(t0=(t0-0.01,t0+0.01))   

truth=[chirpmass,q,DL,inc]
para_range =[(chirpmass-10000,chirpmass+10000),(q-0.01,q+0.01),(DL*0.4,DL*1.2),(0.0,np.pi)]
prior = JointDistribution(variable,UniformLog10(chirpmass=(1e4,1e6)),Uniform(q=(q-0.1,q+0.1)),distance_prior,inclination_prior)
print('prior',prior(chirpmass=chirpmass,q=q,distance=DL,inc=inc))

# truth=[DL,inc,lambd,beta]
# truth=[m1,m2,DL,inc,t0]
# para_range =[(m1-1e3,m1+1e3),(m2-1e2,m2+1e2),(DL*0.9,DL*1.1),(inc-0.01,inc+0.01),(t0-0.01,t0+0.01)]
# para_range =[(chirpmass-10,chirpmass+10),(q-0.01,q+0.01),(DL*0.4,DL*1.2),(0.0,np.pi),(0.0,2*np.pi),(-np.pi/2,-np.pi/2)]

# para_range =[(DL*0.8,DL*1.2),(inc-0.1,inc+0.1),(0.0,2*np.pi),(-np.pi/2,np.pi/2)]
# para_range =[(DL*0.8,DL*1.2),(inc-0.1,inc+0.1),(lambd-0.01,lambd+0.01),(beta-0.01,beta+0.01)]


# prior = JointDistribution(variable,distance_prior,inclination_prior,lambd_prior,beta_prior)
# print('prior',prior(distance=DL,inc=inc,lambd=lambd,beta=beta))
# prior = JointDistribution(variable,Uniform(m1=(m1-1e5,m1+1e5)),Uniform(m2=(m2-1e5,m2+1e5)),distance_prior,inclination_prior)
# prior = JointDistribution(variable,UniformLog10(m1=(1e6,5e6)),UniformLog10(m2=(1e5,5e5)),distance_prior,inclination_prior,t0_prior)
# print('prior',prior(m1=m1,m2=m2,distance=DL,inc=inc,t0=t0))

M1=m1+m2
fhigh = 0.15 /(M1*MTSUN_SI)

# flow = 10 / M1

flow = 5e-4 / (M1*MTSUN_SI)
if(flow<1e-5):
    flow=1e-4
print('flow,fhigh',flow,fhigh)


from templateTDI import TemplateTDI
model_HM =  TemplateTDI(variable,copy.deepcopy(data),
                    low_frequency_cutoff={'LISATDI1':flow,'LISATDI2':flow},
                    high_frequency_cutoff={'LISATDI1':fhigh,'LISATDI2':fhigh},
                    psds = psds,
                    static_params = static,
                    prior = prior,
                    sample_rate = 1,
                    )



smpl = sampler.DynestySampler(model_HM, nlive=1000, nprocesses=150,use_mpi=True) 
#_ = smpl.set_p0() # If we don't set p0, it will use the models prior to draw initial points!
# Note it may take ~1-3 hours for this to run
smpl.run()







#print(res1)
s = smpl.samples
print('samples',s)
res1=smpl._sampler.results
print('result',res1)
np.save('FRTDI/unequal_arm/HM/%d/sampler_results_unEQ_hm.npy'%num,res1)

try:
    weights = np.exp(res1['logwt'] - res1['logz'][-1])
except:
    weights = res1['weights']


sample= res1['samples']
np.save("FRTDI/unequal_arm/HM/%d/sample_unEQ_hm.npy"%num,sample)



###########################################################################

figure = corner.corner(sample,truths=truth,labels=variable,bins=50,weights=weights,
                       quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt='0.4f',
                       range=para_range)
plt.savefig('FRTDI/unequal_arm/HM/%d/corner_unEQ_hm.png'%num,dpi=300)
plt.clf()

from IPython.display import display, Math
#labels=['distance','inclination','tc']
for i in range(len(variable)):
    mcmc = np.percentile(sample[:, i], [16, 50, 84])
    qq = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], qq[0], qq[1], variable[i])
    print(txt)
    display(Math(txt))


end=time()
print('time',(end-start)/3600)




#############################################
from dynesty import plotting as dyplot
fg, ax = dyplot.cornerplot(res1, color='blue', truths=truth,
                           truth_color='black', show_titles=True,title_fmt='0.3f',
                           max_n_ticks=5, quantiles=[0.16,0.5,0.84],labels=variable,
                           span=para_range)

plt.savefig("FRTDI/unequal_arm/HM/%d/coner_plots_unEQ_hm.png"%num,dpi=300)
plt.clf()

fig, axes = dyplot.runplot(res1)  # summary (run) plot
plt.savefig("FRTDI/unequal_arm/HM/%d/summary_plots_unEQ_hm.png"%num,dpi=300)
plt.clf()

fig, axes = dyplot.traceplot(res1, truths=np.zeros(len(variable)),quantiles=[0.16, 0.5, 0.84],
                             truth_color='black', show_titles=True,labels=variable,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))

plt.savefig("FRTDI/unequal_arm/HM/%d/trace_plots_unEQ_hm.png"%num,dpi=300)
plt.clf()

fg, ax = dyplot.cornerpoints(res1, cmap='plasma', truths=truth,labels=variable,
                             kde=False)
plt.savefig("FRTDI/unequal_arm/HM/%d/coner_points_unEQ_hm.png"%num,dpi=300)
plt.clf()


