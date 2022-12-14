# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
sys.path.append('/disk1/home/wangrj/higherorder/signal')
#sys.path.append('/home/wrj/LDC/lib/python3.9/site-packages')
sys.path.append('/disk1/home/wangrj/LDC/MLDC-master/software/LDCpipeline/scripts')
from pycbc import types,fft,noise,frame
from pycbc.filter import highpass,lowpass_fir, matched_filter, matched_filter_core,sigmasq
from pycbc.psd import welch, interpolate
from pycbc.psd.read import from_numpy_arrays
from pycbc.conversions import q_from_mass1_mass2,mchirp_from_mass1_mass2,primary_mass,secondary_mass
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q
from gensignal import gen_signal,gen_signal_fre,get_fd_htilde_lm,get_fd_LISATDI,func_wfTDI, get_TDI

import Cosmology
import LISAConstants as LC
from pyFDresponse import *
from tdi import noisepsd_T,noisepsd_AE
plt.switch_backend('agg')


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


#analytic PSD of noise for TDI A, E and T
del_f= 1e-6
f= np.arange(1e-5,3.0,del_f)
flen = int(2.0/del_f)+1
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



### Generate 1 year of noise 
Tobs=3600*24*100
del_t= 0.5
tsamples = int(Tobs / del_t)
noiseT = noise.noise_from_psd(tsamples, del_t, PSD_TDIt)
noiseAE = noise.noise_from_psd(tsamples, del_t, PSD_TDIae)
plt.plot(noiseAE.sample_times, noiseAE)
plt.xlabel('time')
plt.ylabel('noise A E')
#plt.savefig('noisehae.png',dpi=300)
#plt.show()
plt.clf()



########################################################################
#generate the template waveform
##MBHBs parameters
z=1.0
m1s=2e5
m2s=2e4

#dimensionless spins: a=cos(theta_s)*s
chi1 = 0.0
chi2 = 0.0

#Tcs = 0.8 * LC.YRSID_SI
#t0= np.random.uniform(0.0,5.0)
t0=0.05
#Ecliptic Longitude, Latitude
theta=1.0
phi=1.5
lambd =  phi 
beta = np.pi/2 -theta

inc = 0.5
DL = Cosmology.DL(z, w=0)[0] # in Mpc
print('Dl',DL)

#phi0 = np.random.uniform(0.0, 2.0*np.pi)
#psi = np.random.uniform(0.0, 2.0*np.pi)   #pols=psi
psi=0.8
phi0=0.0
print('t0,phi0,psi',t0,phi0,psi)
# So far, the only choice implemented: MLDC trajectories and TDI XYZ
[fRef, trajdict, TDItag] = [0., trajdict_MLDC, "TDIAET"]



# m1 =  m1s*(1+z)  ### redshifted masses
# m2 =  m2s*(1+z)
m1=m1s
m2=m2s
print('m1,m2',m1,m2)
chirpmass=mchirp_from_mass1_mass2(m1,m2)
q=q_from_mass1_mass2(m1,m2)

print(mass1_from_mchirp_q(chirpmass,q),mass2_from_mchirp_q(chirpmass,q))

apx=['SEOBNRv4HM','IMRPhenomXHM']
modes=[[[2,2]],[[2,2],[2,1],[3,2],[3,3],[4,4]]] 
#############################################################################




#generate TDI data
f,hpf,hcf = gen_signal_fre(chirpmass,q,DL,inc,phi0,chi1,chi2,apx[1],modes[1])
Fa_plus,Fa_cross,Fe_plus,Fe_cross= FLISA(t0,lambd,beta,psi,0)
print(Fa_plus,Fa_cross,Fe_plus,Fe_cross)
htilde_a = Fa_plus*hpf + Fa_cross*hcf
htilde_e = Fe_plus*hpf + Fe_cross*hcf
tmp_ha= copy.deepcopy(htilde_a)
tmp_he= copy.deepcopy(htilde_e)
#print(htilde_a[100000])
print('delta f',htilde_a.delta_f,htilde_a.sample_frequencies)

plt.loglog(htilde_a.sample_frequencies,abs(htilde_a)*htilde_a.sample_frequencies,label='htilde_a(LF)')
plt.loglog(PSD_TDIae.sample_frequencies,np.sqrt(PSD_TDIae*PSD_TDIae.sample_frequencies),label='$S^{a,e}_{h}$')
plt.legend()
plt.xlim(1e-5,1e-1)
#plt.ylim(1e-26,1e-17)
plt.xlabel('freq')
plt.ylabel('strain')
#plt.show()
#plt.savefig('/disk1/home/wangrj/higherorder/catalogues/TDIdata/Q3nod/detection_a%d_%d.png'%(i,j))
plt.clf()


#compute SNR
psdAE = interpolate(PSD_TDIae,htilde_e.delta_f)
print(htilde_a.delta_f,psdAE.delta_f)
flow=1e-4
fhigh=1e-2
snr = sigmasq(htilde_a,psd=psdAE, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)+\
    sigmasq(htilde_e,psd=psdAE, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh) 

print('snr',np.sqrt(snr))


#FFT it to the time-domain
tlen = int(1.0 / del_t / htilde_a.delta_f)
print(len(htilde_a),tlen//2+1)
tmp_ha.resize(tlen//2+1)
tmp_he.resize(tlen//2+1)
print('h_a frequency',tmp_ha.sample_frequencies)

ht_a_HM=tmp_ha.to_timeseries()
ht_e_HM=tmp_he.to_timeseries()
plt.plot(noiseAE.sample_times, noiseAE,label='noise a')
plt.plot(ht_a_HM.sample_times,ht_a_HM,label='a(HM) ')
plt.xlabel('time')
plt.ylabel('htilde_a')
plt.legend()
#plt.show()
#plt.savefig('detection_a.png')
plt.clf()

plt.plot(noiseAE.sample_times, noiseAE,label='noise e')
plt.plot(ht_e_HM.sample_times,ht_e_HM,label='e(HM)')
plt.xlabel('time')
plt.ylabel('htilde_a')
plt.legend()
#plt.show()
#plt.savefig('detection_e.png')
plt.clf()



#signal + noise

tstart = int(t0*365*24*3600/del_t)
strainA = types.TimeSeries(noiseAE.data.data[:],delta_t=noiseAE.delta_t)
strainE = types.TimeSeries(noiseAE.data.data[:],delta_t=noiseAE.delta_t)

print(ht_a_HM.sample_times,ht_a_HM.duration,ht_a_HM.delta_t)
tmp_htA =types.TimeSeries(ht_a_HM.data.data[:],delta_t=ht_a_HM.delta_t)
tmp_htE =types.TimeSeries(ht_e_HM.data.data[:],delta_t=ht_e_HM.delta_t)


nlen=len(strainA)
tlen=len(tmp_htA)
tmp_htA.start_time = tstart *tmp_htA.delta_t     
tmp_htE.start_time = tstart *tmp_htE.delta_t   

print('dt',tmp_htA.delta_t,strainA.delta_t)
print('df',tmp_htA.delta_f,strainA.delta_f,1.0/strainA.duration)
print(nlen,tlen,tstart)
print(strainA.sample_times[tstart])
print('noise',strainA[tstart:tstart+tlen])
print('signal',tmp_htA[0:tlen])
strainA[tstart:tstart+tlen]= strainA[tstart:tstart+tlen] +tmp_htA[0:tlen]
strainE[tstart:tstart+tlen]= strainE[tstart:tstart+tlen] +tmp_htE[0:tlen]
print('signal+noise',strainA[tstart:tstart+tlen])

if(50/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q))<=1e-4):
    fhigh=2e4/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q))
    flow=1e-4
else:
    fhigh=np.minimum(1e-1,2e4/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q)))
    flow=1e-3

print(flow,fhigh)
dataA=highpass(strainA,flow)
dataA=lowpass_fir(dataA,fhigh,512)
dataE=highpass(strainE,flow)
dataE=lowpass_fir(dataE,fhigh,512)


frame.write_frame("strainA.gwf", "LISA", dataA)
frame.write_frame("strainE.gwf", "LISA", dataE)


