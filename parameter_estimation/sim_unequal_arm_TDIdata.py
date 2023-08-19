# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
path ='/disk1/home/wrjx'
sys.path.append(path)
# sys.path.append('/disk1/home/wrjx/MLDC-master/software/LDCpipeline/scripts')
from LitePIG.signal.gensignal import *
from LitePIG.noise.TDInoise import *

from pycbc import types,fft,noise,frame
from pycbc.filter import highpass,lowpass_fir, matched_filter, matched_filter_core,sigmasq
from pycbc.psd import welch, interpolate
from pycbc.psd.read import from_numpy_arrays
from pycbc.conversions import q_from_mass1_mass2,mchirp_from_mass1_mass2,primary_mass,secondary_mass
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q

import Cosmology
import LISAConstants as LC
from pyFDresponse import *
from tdi import noisepsd_T,noisepsd_AE
plt.switch_backend('agg')



########################################################################
#generate the template waveform
##MBHBs parameters
z=1.0
m1s=2e6
m2s=2e5

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
#################################################################################
#PSD of noise for TDI A, E and T 
del_f= 1e-6
f= np.arange(1e-5,1.0,del_f)
flen = int(1.0/del_f)+1
flow=1e-5
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



### Generate 100days of noise 
Tobs=3600*24*50
del_t= 1.0
tsamples = int(Tobs / del_t)
noiseT = noise.noise_from_psd(tsamples, del_t, PSD_TDIT)
noiseA = noise.noise_from_psd(tsamples, del_t, PSD_TDIA)
noiseE = noise.noise_from_psd(tsamples, del_t, PSD_TDIE)





###########################################################################
#generate TDI data
freqs, A,E,T = get_FR_TDI(chirpmass,q,DL,inc,phi0,chi1,chi2,lambd,beta,psi,t0=t0,apx=apx[1],modes=modes[1],
                          trajdict=trajdict_numerical_orbits,TDItag='TDIAET',
                          number_orbits=True,TDI_froze_arm=False,frozenLISA=True,df=2e-6,del_t=1.0)

TDIA= types.FrequencySeries(A,delta_f=freqs[1]-freqs[0])
TDIE= types.FrequencySeries(E,delta_f=freqs[1]-freqs[0])
TDIT= types.FrequencySeries(T,delta_f=freqs[1]-freqs[0])
tmp_A= copy.deepcopy(TDIA)
tmp_E= copy.deepcopy(TDIE)
tmp_T= copy.deepcopy(TDIT)
print('delta f',TDIA.delta_f,TDIE.delta_f,TDIT.delta_f)
# print(TDIA.sample_frequencies)



M1=m1+m2
fhigh = 0.2 /(M1*MTSUN_SI)
# flow = 10 / M1
flow = 5e-4 / (M1*MTSUN_SI)
print(flow,fhigh)





#compute SNR
psdA = interpolate(PSD_TDIA,TDIA.delta_f)
psdE = interpolate(PSD_TDIE,TDIE.delta_f)
psdT = interpolate(PSD_TDIT,TDIT.delta_f)
# print(TDIA.delta_f,psdA.delta_f)
# print(psdA)
flow1=1e-5
fhigh1=5e-1
snr_AA = sigmasq(TDIA,psd=psdA, low_frequency_cutoff=flow1,high_frequency_cutoff=fhigh1)
snr_EE = sigmasq(TDIE,psd=psdE, low_frequency_cutoff=flow1,high_frequency_cutoff=fhigh1)
snr_TT = sigmasq(TDIT,psd=psdT, low_frequency_cutoff=flow1,high_frequency_cutoff=fhigh1)  
print('snr',np.sqrt(snr_AA+snr_EE+snr_TT))
print('snr A',np.sqrt(snr_AA))
print('snr E',np.sqrt(snr_EE))
print('snr T',np.sqrt(snr_TT))

# df=1e-5
# snr = snr_AA+snr_EE+snr_TT
# while(snr>0):
#     tmp_snr = sigmasq(TDIA,psd=psdA, low_frequency_cutoff=flow1+df,high_frequency_cutoff=fhigh1)+\
#               sigmasq(TDIE,psd=psdE, low_frequency_cutoff=flow1+df,high_frequency_cutoff=fhigh1)+\
#               sigmasq(TDIT,psd=psdT, low_frequency_cutoff=flow1+df,high_frequency_cutoff=fhigh1)
#     if(np.sqrt(abs(snr-tmp_snr))>1e-1):
#         break
#     #print(np.sqrt(tmp_snr))  
#     flow1 += df
#     snr = tmp_snr
# print(flow1)
# while(snr>0):
#     tmp_snr = sigmasq(TDIA,psd=psdA, low_frequency_cutoff=flow1,high_frequency_cutoff=fhigh1-df)+\
#               sigmasq(TDIE,psd=psdE, low_frequency_cutoff=flow1,high_frequency_cutoff=fhigh1-df)+\
#               sigmasq(TDIT,psd=psdT, low_frequency_cutoff=flow1,high_frequency_cutoff=fhigh1-df)
#     if(np.sqrt(abs(snr-tmp_snr))>1e-1):
#         break
#     #print(np.sqrt(tmp_snr))  
#     fhigh1 -= df
#     snr = tmp_snr
# print('flow,fhigh',flow1,fhigh1)


# plt.loglog(freqs,np.abs(A*freqs),label='unequal_arm TDIA')
# plt.loglog(freqs,np.abs(E*freqs),label='unequal_arm TDIE')
# plt.loglog(freqs,np.abs(T*freqs),label='unequal_arm TDIT')
# plt.loglog(PSD_TDIT.sample_frequencies,np.sqrt(PSD_TDIT*PSD_TDIT.sample_frequencies),label='PSD T LISA')
# plt.loglog(PSD_TDIA.sample_frequencies,np.sqrt(PSD_TDIA*PSD_TDIA.sample_frequencies),label='PSD A LISA')
# plt.loglog(PSD_TDIE.sample_frequencies,np.sqrt(PSD_TDIE*PSD_TDIE.sample_frequencies),label='PSD E LISA')
# # plt.axvline(x=flow1,color='red')
# # plt.axvline(x=fhigh1,color='red')
# plt.axvline(x=flow)
# plt.axvline(x=fhigh)
# plt.xlabel('freq')
# plt.ylabel('strain')
# plt.grid()
# plt.legend()
# plt.xlim(1e-5,1e-0)
# plt.savefig('TDI_unequal_AET.png',dpi=300)
# #plt.show()
# plt.clf()





#FFT it to the time-domain
tlen = int(1.0 / del_t / TDIA.delta_f)
print(len(TDIA),tlen//2+1)
tmp_A.resize(tlen//2+1)
tmp_E.resize(tlen//2+1)
tmp_T.resize(tlen//2+1)
print('A frequency',tmp_A.sample_frequencies)

At_HM=tmp_A.to_timeseries()
Et_HM=tmp_E.to_timeseries()
Tt_HM=tmp_T.to_timeseries()



#signal + noise

tstart = int(t0* YRSID_SI / del_t)
strainA = types.TimeSeries(noiseA.data.data[:],delta_t=noiseA.delta_t)
strainE = types.TimeSeries(noiseE.data.data[:],delta_t=noiseE.delta_t)
strainT = types.TimeSeries(noiseT.data.data[:],delta_t=noiseT.delta_t)
# strainA = types.TimeSeries(fir_noiseA.data.data[:],delta_t=noiseA.delta_t)
# strainE = types.TimeSeries(fir_noiseE.data.data[:],delta_t=noiseE.delta_t)
# strainT = types.TimeSeries(fir_noiseT.data.data[:],delta_t=noiseT.delta_t)


print(At_HM.sample_times,At_HM.duration,At_HM.delta_t)
tmp_htA =types.TimeSeries(At_HM.data.data[:],delta_t=At_HM.delta_t)
tmp_htE =types.TimeSeries(Et_HM.data.data[:],delta_t=Et_HM.delta_t)
tmp_htT =types.TimeSeries(Tt_HM.data.data[:],delta_t=Tt_HM.delta_t)


nlen=len(strainA)
tlen=len(tmp_htA)
tmp_htA.start_time = tstart *tmp_htA.delta_t     
tmp_htE.start_time = tstart *tmp_htE.delta_t
tmp_htT.start_time = tstart *tmp_htT.delta_t     

print('dt',tmp_htA.delta_t,strainA.delta_t)
print('df',tmp_htA.delta_f,strainA.delta_f,1.0/strainA.duration)
print(nlen,tlen,tstart)
print(strainA.sample_times[tstart])
print('noise',strainA[tstart:tstart+tlen])
print('signal',tmp_htA[0:tlen])
strainA[tstart:tstart+tlen]= strainA[tstart:tstart+tlen] +tmp_htA[0:tlen]
strainE[tstart:tstart+tlen]= strainE[tstart:tstart+tlen] +tmp_htE[0:tlen]
strainT[tstart:tstart+tlen]= strainT[tstart:tstart+tlen] +tmp_htT[0:tlen]
print('signal+noise',strainA[tstart:tstart+tlen])

# if(50/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q))<=1e-4):
#     fhigh=2e4/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q))
#     flow=1e-4
# else:
#     fhigh=np.minimum(1e-1,2e4/(mass1_from_mchirp_q(chirpmass,q)+mass2_from_mchirp_q(chirpmass,q)))
#     flow=1e-3

flow = 1e-5

dataA=highpass(strainA,flow,filter_order=1)
# dataA=lowpass_fir(dataA,fhigh,512)
dataE=highpass(strainE,flow,filter_order=1)
# dataE=lowpass_fir(dataE,fhigh,512)
dataT=highpass(strainT,flow,filter_order=1)
# dataT=lowpass_fir(dataT,fhigh,512)

print(flow,fhigh)

num=0
frame.write_frame("FRTDI/unequal_arm/HM/%d/strainA.gwf"%num, "LISA", dataA)
frame.write_frame("FRTDI/unequal_arm/HM/%d/strainE.gwf"%num, "LISA", dataE)
frame.write_frame("FRTDI/unequal_arm/HM/%d/strainT.gwf"%num, "LISA", dataT)


