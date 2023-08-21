# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys, os, math,copy,corner
from time import *
path ='/disk1/home/wrjx'
sys.path.append(path)
# sys.path.append('/disk1/home/wangrj/LDC/MLDC-master/software/LDCpipeline/scripts')
from LitePIG.signal.gensignal import gen_signal,gen_signal_fre,get_fd_htilde_lm,get_fd_LISATDI,func_wfTDI, get_TDI
#from LitePIG.noise.TDInoise import *

from pycbc import types,fft,frame
# from pycbc.filter import highpass,lowpass_fir, matched_filter, matched_filter_core,sigmasq
from pycbc.psd import welch, interpolate
from pycbc.psd.read import from_numpy_arrays
from pycbc.conversions import q_from_mass1_mass2,mchirp_from_mass1_mass2,primary_mass,secondary_mass
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q

from pycbc.distributions import Uniform, JointDistribution, SinAngle,UniformAngle,UniformLog10,CosAngle
from pycbc.inference import  sampler

import Cosmology
import LISAConstants as LC
from pyFDresponse import *
from tdi import noisepsd_T,noisepsd_AE
from dynesty import plotting as dyplot
from dynesty.utils import quantile as _quantile

def FLISA(t,lambd,beta,psi,t0):
    alpha= 2*np.pi*(t-t0)      #t,t0: yr
    #print(alpha)
    beta_L= np.arcsin(np.cos(np.pi/3)*np.sin(beta)-np.sin(np.pi/3)*np.cos(beta)*np.cos(lambd-alpha))
    lambd_L= np.arctan(np.cos(beta)*np.cos(lambd)*(np.cos(np.pi/3)*np.cos(alpha)**2+np.sin(alpha)**2)+\
                    np.cos(beta)*np.sin(lambd)*np.cos(alpha)*np.sin(alpha)*(np.cos(np.pi/3)-1)+\
                    np.sin(np.pi/3)*np.sin(beta)*np.cos(alpha)+\
                    np.cos(beta)*np.sin(lambd)*(np.cos(np.pi/3)*np.sin(alpha)**2+np.cos(alpha)**2)+\
                    np.cos(beta)*np.cos(lambd)*np.cos(alpha)*np.sin(alpha)*(np.cos(np.pi/3)-1)+
                    np.sin(np.pi/3)*np.sin(beta)*np.sin(alpha))
    psi_L= psi +np.arctan(np.cos(np.pi/3)*np.cos(beta)+np.sin(np.pi/3)*np.sin(beta)*np.cos(lambd-alpha)-np.sin(np.pi/3)*np.sin(lambd-alpha))

    F_a_plus = (1+np.sin(beta_L)**2)*np.cos(2*lambd_L-np.pi/3)/2
    F_a_cross = np.sin(beta_L)*np.sin(2*lambd_L-np.pi/3)
    F_e_plus = (1+np.sin(beta_L)**2)*np.cos(2*lambd_L+np.pi/6)/2
    F_e_cross = np.sin(beta_L)*np.sin(2*lambd_L+np.pi/6)
    #include the polarization angle in the pattern function
    Fa_plus = np.cos(2*psi_L)*F_a_plus + np.sin(2*psi_L) *F_a_cross
    Fe_plus = np.cos(2*psi_L)*F_e_plus + np.sin(2*psi_L) *F_e_cross
    Fa_cross = -np.sin(2*psi_L)*F_a_plus + np.cos(2*psi_L) *F_a_cross
    Fe_cross = -np.sin(2*psi_L)*F_e_plus + np.cos(2*psi_L) *F_e_cross
    return Fa_plus,Fa_cross,Fe_plus,Fe_cross

#analytic PSD of noise for TDI A, E and T
del_f= 1e-6
f= np.arange(1e-5,1.0,del_f)
flen = int(1.0/del_f)+1
flow=1e-5
#freq= logsampling(1e-5,1.0,200)
PSD_TDIT= noisepsd_T(f)
PSD_TDIAE= noisepsd_AE(f)
L= 2.5e9/LC.c
PSD_TDIae= PSD_TDIAE/(2*np.sin(2*np.pi*f*L)**2)
PSD_TDIt= PSD_TDIT/(8*np.sin(np.pi*f*L)**2*np.sin(2*np.pi*f*L)**2)
#We can define a strain-like noise PSD
PSD_TDIae= PSD_TDIae/(6*np.pi*f*L)**2
PSD_TDIt= PSD_TDIt/(6*np.pi*f*L)**2
PSD_TDIt = from_numpy_arrays(f, PSD_TDIt, flen, del_f,flow)
PSD_TDIae = from_numpy_arrays(f, PSD_TDIae, flen, del_f,flow)



####################################################################
#generate the template waveform
##MBHBs parameters
z=1
m1s=2e5
m2s=2e4
print('m1,m2',m1s,m2s)

#dimensionless spins: a=cos(theta_s)*s
chi1 = 0.0
chi2 = 0.0

#Tcs = 0.8 * LC.YRSID_SI
# t0= np.random.uniform(0.0,5.0)
t0=0.05
#Ecliptic Longitude, Latitude
theta=1.0
phi=1.5
lambd =  phi 
beta = np.pi/2 -theta
print('lambd,beta',lambd,beta)

inc = 0.5
DL = Cosmology.DL(z, w=0)[0] # in Mpc
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


# m1 =  m1s*(1+z)  ### redshifted masses
# m2 =  m2s*(1+z)
m1=m1s
m2=m2s
# print('m1,m2',m1,m2)
chirpmass=mchirp_from_mass1_mass2(m1,m2)
q=q_from_mass1_mass2(m1,m2)
print('chirp mass, q ',chirpmass,q)

#we use the frequency domain waform model: IMRPhenomHM
apx=['SEOBNRv4HM','IMRPhenomXHM']
modes=[[[2,2]],[[2,2],[2,1],[3,2],[3,3],[4,4]]] 





##################################################################

#generate TDI data
fre,hpf,hcf = gen_signal_fre(chirpmass,q,DL,inc,phi0,chi1,chi2,apx[1],modes[1],df=2e-6)
Fa_plus,Fa_cross,Fe_plus,Fe_cross= FLISA(t0,lambd,beta,psi,0)
af = Fa_plus*hpf + Fa_cross*hcf
ef = Fe_plus*hpf + Fe_cross*hcf

# plt.loglog(htilde_a.sample_frequencies,abs(htilde_a)*htilde_a.sample_frequencies,label='htilde_a(LF)')
# plt.loglog(htilde_e.sample_frequencies,abs(htilde_e)*htilde_e.sample_frequencies,label='htilde_e(LF)')
# plt.loglog(PSD_TDIt.sample_frequencies,np.sqrt(PSD_TDIt*PSD_TDIt.sample_frequencies),label='$S^{t}_{h}$')
# plt.loglog(PSD_TDIae.sample_frequencies,np.sqrt(PSD_TDIae*PSD_TDIae.sample_frequencies),label='$S^{a,e}_{h}$')
# plt.xlabel('freq')
# plt.ylabel('strain')
# plt.legend()
# plt.xlim(1e-4,1e-1)

# #plt.show()
# plt.clf()



######################################################################
num=0
# #read the strain data
# dataA = frame.read_frame('lowF/HM/%d/strainA.gwf'%num,'LISA')
# dataE = frame.read_frame('lowF/HM/%d/strainE.gwf'%num,'LISA')
# # print(dataA.sample_times)
# # print('data dt and df',dataA.delta_t,dataA.delta_f,1.0/dataA.duration)

# tlen =int(1.0 / dataA.delta_t / 2e-6)
# tstart=int(t0*365*24*3600/dataA.delta_t)
# print(tlen,tstart)
# #########Limit to times around the signal
# dataA1 = dataA.time_slice(tstart*dataA.delta_t,tstart*dataA.delta_t+tlen*dataA.delta_t)
# dataE1 = dataE.time_slice(tstart*dataE.delta_t,tstart*dataE.delta_t+tlen*dataE.delta_t)
# print('limit to times: data dt and df',dataA1.delta_t,dataA1.delta_f,1.0/dataA1.duration) 


# af =dataA1.to_frequencyseries()   # Convert to a frequency series by taking the data's FFT
# ef =dataE1.to_frequencyseries()

print('delta_f',af.delta_f)
print(af.sample_frequencies)

# plt.loglog(af.sample_frequencies,abs(af),label='Data(with noise)')
# plt.xlim(1e-4,1)
# plt.ylabel('|a(f)|')
# plt.xlabel('Freq')
# plt.xlim(1e-4,1)
# plt.savefig('fft_data.png',dpi=300)
# plt.clf()





data={}
data['LISATDI1'] = af
data['LISATDI2'] = ef


psds={}
psdAE = interpolate(PSD_TDIae,af.delta_f)
psds['LISATDI1'] = psdAE
psds['LISATDI2'] = psdAE




start=time()
static = {'chirpmass':chirpmass,
          'q':q,
          #'distance':DL,
          #'inc':inc,
          'phi0':phi0,
          'chi1':chi1,
          'chi2':chi2,
          #'lambd':lambd,
          #'beta':beta,
          'psi':psi,
          't0':t0,
          'trajdict':trajdict,
          'number_orbits':False,
          'TDI_froze_arm':True,
          'frozenLISA':True,
          'TDItag':TDItag,
          'apx':apx[1],
          'modes':modes[1]      #modes[1] HM
         }
variable = [
            #'chirpmass',
            #'q',
            'distance',
            'inc',
            'lambd',
            'beta'
            ]
inclination_prior = SinAngle(inc=None)
#inclination_prior = Uniform(inc=(0.0,np.pi))
distance_prior = Uniform(distance=(DL*0.4, DL*2))
#t0_prior = Uniform(tc=(t0-1,t0+1))   #tc=50000+385594
lambd_prior= Uniform(lambd=(0.0,2*np.pi))
#lambd_prior= Uniform(lambd=(lambd-1,2*np.pi))
beta_prior= CosAngle(beta=None)
#beta_prior= Uniform(beta=(beta-1,beta+1))

truth=[DL,inc,lambd,beta]
#para_range =[(chirpmass-10,chirpmass+10),(q-0.01,q+0.01),(DL*0.4,DL*1.2),(0.0,np.pi),(0.0,2*np.pi),(-np.pi/2,-np.pi/2)]
# para_range =[(DL*0.4,DL*1.5),(0.0,np.pi),(0.0,2*np.pi),(-np.pi/2,np.pi/2)]
para_range =[(DL*0.4,DL*2),(inc-0.1,inc+0.1),(lambd-0.01,lambd+0.01),(beta-0.01,beta+0.01)]
prior = JointDistribution(variable,distance_prior,inclination_prior,lambd_prior,beta_prior)
print('prior',prior(distance=DL,inc=inc,lambd=lambd,beta=beta))

# prior = JointDistribution(variable,Uniform(chirpmass=(chirpmass-10,chirpmass+10)),Uniform(q=(q-0.01,q+0.01)),
#                           distance_prior,inclination_prior,lambd_prior,beta_prior)
# print('prior',prior(chirpmass=chirpmass,q=q,distance=DL,inc=inc,lambd=lambd,beta=beta))

# prior = JointDistribution(variable,Uniform(chirpmass=(chirpmass-10,chirpmass+10)),Uniform(q=(q-0.01,q+0.01)),distance_prior,inclination_prior)
# print('prior',prior(chirpmass=chirpmass,q=q,distance=DL,inc=inc))


# if(50/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q))<=1e-4):
#     flow=1e-4
#     fhigh=2e4/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q))
# else:
#     flow=1e-3
#     fhigh=np.minimum(1e-1,2e4/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q)))  
# print('flow,fhigh',flow,fhigh)

M1=m1+m2
fhigh = 0.1 /(M1*MTSUN_SI)

# flow = 10 / M1

flow = 1e-3 / (M1*MTSUN_SI)
if(flow<1e-5):
    flow=1e-5
print('flow,fhigh',flow,fhigh)
from templateTDILF import TemplateTDILF
model_HM =  TemplateTDILF(variable,copy.deepcopy(data),
                    low_frequency_cutoff={'LISATDI1':flow,'LISATDI2':flow},
                    high_frequency_cutoff={'LISATDI1':fhigh,'LISATDI2':fhigh},
                    psds = psds,
                    static_params = static,
                    prior = prior,
                    sample_rate = 2,
                    )



smpl = sampler.DynestySampler(model_HM, nlive=1000, nprocesses=150,use_mpi=True) 
#_ = smpl.set_p0() # If we don't set p0, it will use the models prior to draw initial points!
# Note it may take ~1-3 hours for this to run
smpl.run()


end=time()
print('run time',(end-start)/3600)

s = smpl.samples
print('samples',s)
res1=smpl._sampler.results
print('result',res1)
np.save('lowF/HM/%d/sampler_results_hm.npy'%num,res1)

sample= res1['samples']
np.save("lowF/HM/%d/sample_hm.npy"%num,sample)




###########################################################################
try:
    weights = np.exp(res1['logwt'] - res1['logz'][-1])
except:
    weights = res1['weights']

# figure = corner.corner(sample,truths=truth,labels=variable,bins=50,weights=weights,
#                        quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt='0.3f',
#                        range=para_range)
#plt.savefig('lowF/HM/%d/corner_hm.png'%num,dpi=300)
# plt.clf()

# from IPython.display import display, Math
# labels=['distance','inclination','tc']
# for i in range(len(variable)):
#     mcmc = np.percentile(sample[:, i], [16, 50, 84])
#     qq = np.diff(mcmc)
#     txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
#     txt = txt.format(mcmc[1], qq[0], qq[1], variable[i])
#     print(txt)
#     display(Math(txt))







#############################################

fg, ax = dyplot.cornerplot(res1, color='blue', truths=truth,
                           truth_color='black', show_titles=True,title_fmt='0.3f',
                           max_n_ticks=5, quantiles=[0.16,0.5,0.84],labels=variable,
                           span=para_range)

plt.savefig("lowF/HM/%d/coner_plots_hm.png"%num,dpi=300)
plt.clf()

fig, axes = dyplot.runplot(res1)  # summary (run) plot
plt.savefig("lowF/HM/%d/summary_plots_hm.png"%num,dpi=300)
plt.clf()

fig, axes = dyplot.traceplot(res1, truths=np.zeros(len(variable)),quantiles=[0.16, 0.5, 0.84],
                             truth_color='black', show_titles=True,labels=variable,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))

plt.savefig("lowF/HM/%d/trace_plots_hm.png"%num,dpi=300)
plt.clf()

fg, ax = dyplot.cornerpoints(res1, cmap='plasma', truths=truth,labels=variable,
                             kde=False)
plt.savefig("lowF/HM/%d/coner_points_hm.png"%num,dpi=300)
plt.clf()




for i in range(len(variable)):
    ql, qm, qh = _quantile(sample[:,i], [0.16,0.5,0.84], weights=weights)
    q_minus, q_plus = qm - ql, qh - qm
    print(variable[i],qm,q_minus,q_plus)

DL_min,mean_DL,DL_max= _quantile(sample[:,0], [0.16,0.5,0.84], weights=weights)
del_DL= (DL_max-DL_min)/2
print(del_DL)
print('del_DL/DL',del_DL/mean_DL)

lambd_min,mean_lambd,lambd_max= _quantile(sample[:,2], [0.16,0.5,0.84], weights=weights)
beta_min,mean_beta,beta_max= _quantile(sample[:,2], [0.16,0.5,0.84], weights=weights)

delta_lambd= (lambd_max-lambd_min)/2
delta_beta= (beta_max-beta_min)/2

omega1= 2*np.pi*np.abs(np.sin(mean_beta))*np.sqrt(delta_lambd**2*delta_beta**2)
print(omega1,'rad')
print(omega1/(2*np.pi)**2*360**2,'deg^2')
print(omega1/(2*np.pi)**2*360**2*60**2,'arc min^2')