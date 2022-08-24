# LitePIG :a Lite Parameter Inference system for the Gravitational Waves in the millihertz band
LitePIG is software developed by Renjie Wang and Bin Hu. 
It is open source and publicly available.
The softwavre package include the follwing features: the GW waveform emitted from the massive black hole binaries, the stationary instrumental gaussian noise, the higher-order harmonic modes, the full response function from the time delay interfermetry(TDI) and the gaussian likelihood function with the dynamic nested parameter sampler.
These examples explore how to generate the GW waveform, how we generate LISA instrumental noise, and how to esitmate the GW parameter.

# Requirements
The file requirements.txt which you should install if you plan to use LitePIG for GW data analysis.
The software requeire LISACode_Radler(https://lisa-ldc.lal.in2p3.fr/code).

# Example notebooks
We can made some notebooks available. You can find simple example notebooks.

# Waveform generation
We are going to demonstrate how you might generate the wafeform. The first thing is import the necessary modules:
```
import numpy as np
import copy
import matplot.pyplot as plt
from pycbc import types,fft,noise,frame
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q
from signal.gensignal import gen_signal_fre
import Cosmology
```
Then we"ll set up the parameters of GW
```
#MBHB parameters
z=1
m1s=2e5
m2s=2e4

#dimensionless spins: a=cos(theta_s)*s
chi1 = 0.0
chi2 = 0.0

t0=0.05
#Ecliptic Longitude, Latitude
theta=1.0
phi=1.5
lambd =  phi 
beta = np.pi/2 -theta

inc = 0.5
DL = Cosmology.DL(z, w=0)[0] # in Mpc

psi=0.8
phi0=0.0

chirpmass=mchirp_from_mass1_mass2(m1,m2)
q=q_from_mass1_mass2(m1,m2)
#we use the frequency domain waform model: IMRPhenomXHM
apx=['SEOBNRv4HM','IMRPhenomXHM']
modes=[[[2,2]],[[2,2],[2,1],[3,2],[3,3],[4,4]]] 
```
In the fow-frequency approximation, the response functions are
```
def FLISA(t,lambd,beta,psi,t0):
    alpha= 2*np.pi*(t-t0)      #t,t0: yr
    beta_L= np.arcsin(np.cos(np.pi/3)*np.sin(beta)-np.sin(np.pi/3)*np.cos(beta)*np.cos(lambd-alpha))
    lambd_L= np.arctan(np.cos(beta)*np.cos(lambd)*(np.cos(np.pi/3)*np.cos(alpha)**2+np.sin(alpha)**2)+\
                    np.cos(beta)*np.sin(lambd)*np.cos(alpha)*np.sin(alpha)*(np.cos(np.pi/3)-1)+\
                    np.sin(np.pi/3)*np.sin(beta)*np.cos(alpha)+\
                    np.cos(beta)*np.sin(lambd)*(np.cos(np.pi/3)*np.sin(alpha)**2+np.cos(alpha)**2)+\
                    np.cos(beta)*np.cos(lambd)*np.cos(alpha)*np.sin(alpha)*(np.cos(np.pi/3)-1)+
                    np.sin(np.pi/3)*np.sin(beta)*np.sin(alpha))
    psi_L= psi +np.arctan(np.cos(pi/3)*np.cos(beta)+np.sin(np.pi/3)*np.sin(beta)*np.cos(lambd-alpha)-np.sin(np.pi/3)*np.sin(lambd-alpha))

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
```

the response for two TDI observables can be obtained
```
#get a frequency domain waveform htilde_a, htilde_e
f,hpf,hcf = gen_signal_fre(chirpmass,q,DL,inc,phi0,chi1,chi2,apx[1],modes[1])
Fa_plus,Fa_cross,Fe_plus,Fe_cross= FLISA(t0,lambd,beta,psi,0)
htilde_a = Fa_plus*hpf + Fa_cross*hcf
htilde_e = Fe_plus*hpf + Fe_cross*hcf
```

Next, we can fft it to the time-domain,
```
tmp_ha= copy.deepcopy(htilde_a)
tmp_he= copy.deepcopy(htilde_e)
tlen = int(1.0 / del_t / htilde_a.delta_f)
tmp_ha.resize(tlen//2+1)
tmp_he.resize(tlen//2+1)

ht_a_HM=tmp_ha.to_timeseries()
ht_e_HM=tmp_he.to_timeseries()
```

# LISA Noise Generation
The LISA instrumental noise is imported form the LISA data challenge(LDC) working package.
We also import the necessary modules
```
import numpy as np
from pycbc import noise, frame
from pycbc.psd.read import from_numpy_arrays
from pycbc.filter import highpass,lowpass_fir,
import LISAConstants as LC
from tdi import noisepsd_T,noisepsd_AE
```
The analytic noise power spectral density(PSD) for TDI A,E,T
```
del_f= 1e-6
f= np.arange(1e-5,1.0,del_f)
flen = int(1.0/del_f)+1
flow=1e-5
PSD_TDIT= noisepsd_T(f)
PSD_TDIAE= noisepsd_AE(f)
```
Then, We can factor out the same sine square function from PSD and the reduced PSD reads
```
L= 2.5e9/LC.c
PSD_TDIae= PSD_TDIAE/(2*np.sin(2*np.pi*f*L)**2)
PSD_TDIt= PSD_TDIT/(8*np.sin(np.pi*f*L)**2*np.sin(2*np.pi*f*L)**2)
PSD_TDIae= PSD_TDIae/(6*np.pi*f*L)**2
PSD_TDIt= PSD_TDIt/(6*np.pi*f*L)**2

PSD_TDIt = from_numpy_arrays(f, PSD_TDIt, flen, del_f,flow)
PSD_TDIae = from_numpy_arrays(f, PSD_TDIae, flen, del_f,flow)
```
We can generate 100 days of noise
```
Tobs=3600*24*100
del_t= 0.5
tsamples = int(Tobs / del_t)
noiseT = noise.noise_from_psd(tsamples, del_t, PSD_TDIt)
noiseAE = noise.noise_from_psd(tsamples, del_t, PSD_TDIae)
```
Therefore we can generate the data set by summing the signal and noise in each TDI channels
```
#signal + noise
tstart = int(t0*365*24*3600/del_t)
strainA = types.TimeSeries(noiseAE.data.data[:],delta_t=noiseAE.delta_t)
strainE = types.TimeSeries(noiseAE.data.data[:],delta_t=noiseAE.delta_t)

tmp_htA =types.TimeSeries(ht_a_HM.data.data[:],delta_t=ht_a_HM.delta_t)
tmp_htE =types.TimeSeries(ht_e_HM.data.data[:],delta_t=ht_e_HM.delta_t)

nlen=len(strainA)
tlen=len(tmp_htA)
tmp_htA.start_time = tstart *tmp_htA.delta_t     
tmp_htE.start_time = tstart *tmp_htE.delta_t   

strainA[tstart:tstart+tlen]= strainA[tstart:tstart+tlen] +tmp_htA[0:tlen]
strainE[tstart:tstart+tlen]= strainE[tstart:tstart+tlen] +tmp_htE[0:tlen]
```
Then, we should band-pass the data and select frequecies [1e-4,1e-2]Hz
```
fhigh=1e-2
flow=1e-4
dataA=highpass(strainA,flow)
dataA=lowpass_fir(dataA,fhigh,512)
dataE=highpass(strainE,flow)
dataE=lowpass_fir(dataE,fhigh,512)
frame.write_frame("strainA.gwf", "LISA", dataA)
frame.write_frame("strainE.gwf", "LISA", dataE)

```
# Parameter estimation
We also import the necessary modules
```
from pycbc.psd import welch, interpolate
from pycbc.distributions import Uniform, JointDistribution, SinAngle
from templateTDILF import TemplateTDILF
from pycbc.inference import sampler
```
we read the strain data
```
dataA = frame.read_frame('strainA.gwf','LISA')
dataE = frame.read_frame('strainE.gwf','LISA')
```
we can limit to times series around the signal
```
tlen =int(1.0 / dataA.delta_t / 2e-7)
tstart=int(t0*365*24*3600/dataA.delta_t)
dataA1 = dataA.time_slice(tstart*dataA.delta_t,tstart*dataA.delta_t+tlen*dataA.delta_t)
dataE1 = dataE.time_slice(tstart*dataE.delta_t,tstart*dataE.delta_t+tlen*dataE.delta_t)
```
Then we convert to a frequency series by taking the data's FFT
```
af =dataA1.to_frequencyseries()  
ef =dataE1.to_frequencyseries()
data={}
data['LISATDI1'] = af
data['LISATDI2'] = ef

psds={}
psdAE = interpolate(PSD_TDIae,af.delta_f)
psds['LISATDI1'] = psdAE
psds['LISATDI2'] = psdAE
```

We use the dynesty package for inferring Bayesian posteriors distribution of parameters and evidences.
We can set some parameters and the parameter priors are given by 
```
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
          #'trajdict':trajdict,
          #'TDItag':TDItag,
          'apx':apx[1],
          'modes':modes[1]      
         }
variable = [
            #'chirpmass',
            #'q',
            'distance',
            'inc',
            'lambd',
            'beta'
            ]
distance_prior = Uniform(distance=(DL*0.4, DL*2))    
inclination_prior = SinAngle(inc=None)
lambd_prior= Uniform(lambd=(0.0,2*np.pi))
beta_prior= CosAngle(beta=None)
prior = JointDistribution(variable,distance_prior,inclination_prior,lambd_prior,beta_prior)
```
We can set dynesty and run
```
model_HM =  TemplateTDILF(variable,copy.deepcopy(data),
                    low_frequency_cutoff={'LISATDI1':flow,'LISATDI2':flow},
                    high_frequency_cutoff={'LISATDI1':fhigh,'LISATDI2':fhigh},
                    psds = psds,
                    static_params = static,
                    prior = prior,
                    sample_rate = 2,
                    )

smpl = sampler.DynestySampler(model_HM, nlive=1000, nprocesses=100,use_mpi=True) 
# Note it may take ~1-3 hours for this to run
smpl.run()

s = smpl.samples
res1=smpl._sampler.results
np.save('sampler_results_hm.npy',res1)
```

# Plotting results
```
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot


res2=np.load('sampler_results_hm.npy',allow_pickle=True)
try:
    weights2 = np.exp(res2.item()['logwt'] - res2.item()['logz'][-1])
except:
    weights2 = res2.item()['weights']

sample2= res2.item()['samples']

truth=[DL,inc,lambd,beta]
para_range =[(6500,7000),(0.4,0.6),(lambd-0.01,lambd+0.01),(beta-0.01,beta+0.01)]
label=['$D_{L}$','$\iota$','$\lambda$','$\\beta$']
fg, ax = dyplot.cornerplot(res2.item(), color='blue', truths=truth,
                           truth_color='black', show_titles=True,title_fmt='0.3f',
                           max_n_ticks=5, quantiles=[0.16,0.5,0.84],quantiles_2d=[0.68,0.95,0.99],
                           labels=label,span=para_range)

plt.show()

fig, axes = dyplot.traceplot(res2.item(), truths=np.zeros(len(variable)),quantiles=[0.16, 0.5, 0.84],
                             truth_color='black', show_titles=True,labels=variable,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))
plt.show()

fg, ax = dyplot.cornerpoints(res2.item(), cmap='plasma', truths=truth,labels=variable,kde=False)
plt.show()
```

