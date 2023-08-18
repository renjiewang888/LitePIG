

import numpy as np
import sys
from scipy.special import logsumexp
path ='/disk1/home/wrjx'
sys.path.append(path)
from tqdm import tqdm
from pycbc import fft,types,frame
from pycbc import filter as pyfilter
from pycbc.waveform import get_fd_waveform,get_td_waveform,utils
from pycbc.detector import Detector
from pycbc.psd import interpolate
from LitePIG.signal.gensignal import *
from pycbc.conversions import q_from_mass1_mass2,mchirp_from_mass1_mass2,mass1_from_mchirp_q,mass2_from_mchirp_q
from pycbc.inference.models.gaussian_noise import BaseGaussianNoise




class TemplateTDILF(BaseGaussianNoise):
    r"""

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
    name = 'templateTDILF'

    def __init__(self, variable_params, data, low_frequency_cutoff,
                 sample_rate=2, polarization_samples=None, **kwargs):

        super(TemplateTDILF, self).__init__(
            variable_params, data, low_frequency_cutoff,**kwargs)

        # 
        self.df = data[self.detectors[0]].delta_f
        self.dt = data[self.detectors[0]].delta_t
        p = self.static_params.copy()
        self.sample_rate= sample_rate
        
        print(p)
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

        if self.pflag == 0:
            polarization = p['psi']   
        elif self.pflag == 1:
            polarization = self.polarization
        #_ = p.pop('polarization')
        if self.time is None:
            self.time = p['t0']
        #_ = p.pop('tc')
  
       
        ##### Generate TDI
        if "m1" in p :
            p['chirpmass'] = mchirp_from_mass1_mass2(p['m1'],p['m2'])
            p['q'] = q_from_mass1_mass2(p['m1'],p['m2'])
        # m1=mass1_from_mchirp_q(p['chirpmass'],p['q'])
        # m2=mass2_from_mchirp_q(p['chirpmass'],p['q'])
        # M1 =m1+m2
        #chirpmass=p['chirpmass']
        #q=p['q']
  
        
        f,hpf,hcf = gen_signal_fre(p['chirpmass'],p['q'],p['distance'],p['inc'],p['phi0'],
                                   p['chi1'],p['chi2'],p['apx'],p['modes'],df=df)
        
        
        Fa_plus,Fa_cross,Fe_plus,Fe_cross= FLISA(p['t0'],p['lambd'],p['beta'],p['psi'],0)
        #print(Fa_plus,Fa_cross,Fe_plus,Fe_cross)
        htilde_a = Fa_plus*hpf + Fa_cross*hcf
        htilde_e = Fe_plus*hpf + Fe_cross*hcf
        #print(TDIA.delta_f,self.df)
        #print('generate TDI')


        # Extend template to high sample rate
        flen = int(self.sample_rate/2 / self.df) +1
        #print(len(htilde_a))
        htilde_a.resize(flen)
        htilde_e.resize(flen)
        #print(htilde_a.delta_f)
        TDI ={}  
        TDI[self.detectors[0]]=htilde_a
        TDI[self.detectors[1]]=htilde_e
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
            #print(flen,len(self.data[ifo]))
            self.data[ifo].resize(flen)
            snr= pyfilter.matched_filter_core(TDI[ifo],self.data[ifo],psd=self.psds[ifo],low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)[0]
            self.sh[ifo] = 4 * df * snr
            self.hh[ifo]= -0.5 * pyfilter.sigmasq(TDI[ifo],psd=self.psds[ifo], low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)


            
            #dt=self.sh[ifo].sample_times
            #print(ifo,'<d|h>',self.sh[ifo][0].real,'<h|h>',self.hh[ifo])
            #print(dt)
            #print(self.sh[ifo].at_time(dt[0]),self.hh[ifo])
    
        shloglr = hhloglr = 0
        for ifo in self.sh:
            dt=self.sh[ifo].sample_times
            sh = self.sh[ifo].at_time(dt[0]) 
            #print(dt[0])
            #print('sh',sh)
            shloglr += sh.real
            #shloglr += sh
            hhloglr += self.hh[ifo] 
        vloglr=0
        vloglr += shloglr + hhloglr
        #print('loglr',vloglr,'<d|h>',shloglr,'<h|h>',hhloglr)
        if self.pflag == 0:
            return float(vloglr)
        else:
            return float(logsumexp(vloglr)) - np.log(len(vloglr))



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

