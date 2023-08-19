import numpy as np
import sys
from scipy.special import logsumexp
sys.path.append('/disk1//home/wangrj/higherorder/signal')
sys.path.append('/disk1//home/wangrj/higherorder/noise')
from gensignal import *
from TDInoise import noisepsd_AE_taiji, noisepsd_T_taiji,noisepsd_unequal_A,noisepsd_unequal_E,noisepsd_unequal_T
from tqdm import tqdm
from pycbc import fft,types,frame
from pycbc import filter as pyfilter
from pycbc.waveform import get_fd_waveform,get_td_waveform,utils
from pycbc.detector import Detector
from pycbc.psd import interpolate
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q,q_from_mass1_mass2,mchirp_from_mass1_mass2
from pycbc.inference.models.gaussian_noise import BaseGaussianNoise

# In this model we only calculate terms up to a constant.
# We are primarily interested in the posterior result


class TemplateTDI(BaseGaussianNoise):
    r"""Model that assumes we know all the intrinsic parameters.

    This model assumes we know all the intrinsic parameters, and are only
    maximizing over the extrinsic ones. 

    Parameters
    ----------
    variable_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    data : dict
        A dictionary of data, in which the keys are the detector names and the
        values are the data (assumed to be unwhitened). All data must have the
        same frequency resolution.
    low_frequency_cutoff : dict
        A dictionary of starting frequencies, in which the keys are the
        detector names and the values are the starting frequencies for the
        respective detectors to be used for computing inner products.
    sample_rate : int, optional
        The sample rate to use. Default is 32768.
    polarization_samples: int, optional
        Parameter to specify how finely to marginalize over polarization angle.
        If None, then polarization must be a parameter.
    \**kwargs :
        All other keyword arguments are passed to
        :py:class:`BaseGaussianNoise`; see that class for details.
    """
    name = 'templateTDI'

    def __init__(self, variable_params, data, low_frequency_cutoff,
                 sample_rate=2, polarization_samples=None, **kwargs):

        super(TemplateTDI, self).__init__(
            variable_params, data, low_frequency_cutoff,**kwargs)

        # 
        self.df = data[self.detectors[0]].delta_f
        self.dt = data[self.detectors[0]].delta_t
        p = self.static_params.copy()
        self.sample_rate= sample_rate
        
        print('static parameter',p)
        print('delta_f,delta_t',self.df,self.dt)
        #print('psds',self.psds[self.detectors[0]])
        #print(self.sample_rate)
        


        #polarization array to marginalize over if num_samples given
        self.pflag = 0
        if polarization_samples is not None:
            self.polarization = np.linspace(0, 2*np.pi,
                                               int(polarization_samples))
            self.pflag = 1
        self.time = None






    def _loglr(self):
        r"""Computes the log likelihood ratio

        Returns
        -------
        float
            The value of the log likelihood ratio.
        """
        # calculate <d-h|d-h> = <h|h> - 2<h|d> + <d|d> 
        p = self.current_params.copy()
        p.update(self.static_params)
        df = self.df
        dt = self.dt

        if self.pflag == 0:
            polarization = p['psi']   
        else:
            polarization = self.polarization
        #_ = p.pop('polarization')
        if self.time is None:
            self.time = p['t0']
        #_ = p.pop('tc')
  
       
        ##### Generate TDI
        if "m1" in p :
            p['chirpmass'] = mchirp_from_mass1_mass2(p['m1'],p['m2'])
            p['q'] = q_from_mass1_mass2(p['m1'],p['m2'])
        #m10=m1=mass1_from_mchirp_q(p['mchirp'],p['q'])
        #m20=m2=mass2_from_mchirp_q(p['mchirp'],p['q'])
        #chirpmass=p['chirpmass']
        #q=p['q']
   
        #print(p)
        #mode=[[[2,2]],[[2,2],[2,1],[3,3],[4,4],[5,5]],[[2,2],[2,1],[3,2],[3,3],[4,4]]] 
        freqs, A,E,T =get_FR_TDI(p['chirpmass'],p['q'],p['distance'],p['inc'],p['phi0'],
                                   p['chi1'],p['chi2'],p['lambd'],p['beta'],p['psi'],p['t0'],
                                   trajdict=p['trajdict'],number_orbits=p['number_orbits'],TDItag=p['TDItag'],apx=p['apx'],modes=p['modes'],
                                   TDI_froze_arm=p['TDI_froze_arm'],frozenLISA=p['frozenLISA'],df=df,del_t=dt)
        TDIA= types.FrequencySeries(A,delta_f=freqs[1]-freqs[0])
        TDIE= types.FrequencySeries(E,delta_f=freqs[1]-freqs[0])
        TDIT= types.FrequencySeries(T,delta_f=freqs[1]-freqs[0])
        #print(TDIA.delta_f,self.df)
        #print('generate TDI')


        # Extend template to high sample rate
        flen = int(self.sample_rate/2 / self.df) +1
        #print(flen,len(hf))
        TDIA.resize(flen)
        TDIE.resize(flen)
        TDIT.resize(flen)
        
        TDI ={}  
        TDI[self.detectors[0]]=TDIA
        TDI[self.detectors[1]]=TDIE
        # TDI[self.detectors[2]]=TDIT
        #print('Resample')

    
        # Calculate high sample rate SNR time series
        self.sh = {}
        self.hh = {}
        self.det = {}
        for ifo in self.data:
            flow = self.kmin[ifo] * df
            fhigh = self.kmax[ifo] * df
            #print(flow,fhigh)
            # Extend data to high sample rate
            self.data[ifo].resize(flen)
            snr= pyfilter.matched_filter_core(TDI[ifo],self.data[ifo],psd=self.psds[ifo],low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)[0]
            #snr= pyfilter.matched_filter_core(TDIA,dataA,psd=psdA,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)[0]+\
            #    pyfilter.matched_filter_core(TDIE,dataE,psd=psdE,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)[0]+\
            #    pyfilter.matched_filter_core(TDIT,dataT,psd=psdT,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)[0]
            self.sh[ifo] = 4 * df * snr
            self.hh[ifo]= -0.5 * pyfilter.sigmasq(TDI[ifo],psd=self.psds[ifo], low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
            #self.hh[ifo]= -0.5 * (pyfilter.sigmasq(TDIA,psd=psdA, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)+\
            #                      pyfilter.sigmasq(TDIE,psd=psdE, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh) +\
            #                      pyfilter.sigmasq(TDIT,psd=psdT, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh))

            
            #dt=self.sh[ifo].sample_times
            #print(dt)
            #print(self.sh[ifo].at_time(dt[0]),self.hh[ifo])
    
        shloglr = hhloglr = 0
        for ifo in self.sh:
            dt=self.sh[ifo].sample_times
            sh = self.sh[ifo].at_time(dt[0]) 
            #print('sh',sh)
            shloglr += sh.real
            #shloglr += sh
            hhloglr += self.hh[ifo] 
        #vloglr = np.log(scipy.special.i0e(abs(shloglr)))
        #vloglr += abs(shloglr) + hhloglr
        vloglr=0
        vloglr += shloglr + hhloglr
        #print('loglr',vloglr, shloglr,hhloglr,p['distance'],p['inclination'],self.time)
        #print('mchirp=',chirp_mass,'q=',q,'distance=',p['distance'],'inclination',p['inclination'])
        #print('loglr',vloglr,'<d|h>',shloglr,'<h|h>',hhloglr)
        return vloglr
        # if self.pflag == 0:
        #     return float(vloglr)
        # else:
        #     return float(logsumexp(vloglr)) - np.log(len(vloglr))
