# coding=utf-8
import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt
import pycbc.noise
import pycbc.psd
import pylab
from tqdm import tqdm
from time import *
from scipy import interpolate,special,integrate
import lalsimulation as lalsim
from pycbc import types
from pycbc.waveform import get_td_waveform,get_fd_waveform
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q
from pycbc.filter import interpolate_complex_frequency, resample_to_delta_t
from pycbc.waveform.utils import phase_from_frequencyseries,amplitude_from_frequencyseries
import lal
from pyFDresponse import *
#plt.switch_backend('agg')





#generate signal 
def gen_signal(chirpmass,q,distance,iota,phi_c,apx,modes):
    #assume a signal
    m1=mass1_from_mchirp_q(chirpmass,q)
    m2=mass2_from_mchirp_q(chirpmass,q)
    print(m1,m2)
    m10=m1
    m20=m2
    Fs=4000
    while(int(m10)>100.0):
        m10=m10/10
    if(int(m10)>50.0):
        m10=m10/4
    m20=m10/q
    print(m10,m20)
    #print(m10,m20)
    dL=distance
    M0=m10+m20
    M1=m1+ m2 
    
    mode=[[[2,2]],[[2,2],[2,1],[3,3],[4,4],[5,5]],[[2,2],[3,3]]] 
    hp, hc = get_td_waveform(approximant=apx,
                         mass1=m10,
                         mass2=m20,
                         spin1z=0.0,
                         spin2z=0.0,
                         distance=dL,
                         inclination=iota,
                         coa_phase=phi_c,
                         #eccentrity=0.4,
                         delta_t=1.0/Fs,
                         f_lower=200/M0,
                         mode_array=modes)
    #hp.start_time = hc.start_time = 0

    #rescale the time/frequency and amplitude
    #dL1=D_l(z)
    t = hp.sample_times *(M1/M0) 
    hp = hp *(dL*M1)/(distance*M0)
    hc = hc *(dL*M1)/(distance*M0)
    return (t,hp,hc)



def gen_signal_fre(chirpmass,q,distance,inc,phi0,chi1,chi2,apx,modes,df=2e-7):
    #assume a signal
    #print(chirpmass,q,distance,inc,phi0)
    m1=mass1_from_mchirp_q(chirpmass,q)
    m2=mass2_from_mchirp_q(chirpmass,q)
    #print(m1,m2)
    m10=m1
    m20=m2
    while(int(m10)>100.0):
        m10=m10/10
    if(int(m10)>50.0):
        m10=m10/4
    m20=m10/q
    #print(m10,m20)

    M0=m10+m20
    M1=m1+ m2 
    Fs=(df)*M1/M0
    rescale= M1/M0
    hp, hc = get_fd_waveform(approximant=apx,
                         mass1=m10,
                         mass2=m20,
                         spin1z=chi1,
                         spin2z=chi2,
                         distance=distance,
                         inclination=inc,
                         coa_phase=phi0,
                         delta_f=Fs,
                         f_lower=50/M0,
                         mode_array=modes)
  
    #rescale the time/frequency and amplitude
    f = hp.sample_frequencies /(M1/M0) 
    hp = types.FrequencySeries(hp.data.data[:],delta_f=hp.delta_f/rescale)*rescale**2
    hc = types.FrequencySeries(hc.data.data[:],delta_f=hc.delta_f/rescale)*rescale**2
    return (f,hp,hc)


#generate the htilde_lm
#h+ -ihx = \sum Y_lm htilde_LM
def get_fd_htilde_lm(chirpmass,q,distance,inc,phi0,chi1,chi2,apx,modes,df=2.0e-7):
    m1=mass1_from_mchirp_q(chirpmass,q)
    m2=mass2_from_mchirp_q(chirpmass,q)
    m10=m1
    m20=m2
    while(int(m10)>100.0):
        m10=m10/10
    if(int(m10)>50.0):
        m10=m10/4
    m20=m10/q

    M0=m10+m20
    M1=m1+ m2 
    flower=10/M0
    Fs=df*M1/M0
    rescale= M1/M0

    hpf, hcf = get_fd_waveform(approximant=apx,
                         mass1=m10,
                         mass2=m20,
                         spin1z=chi1,
                         spin2z=chi2,
                         distance=distance,
                         inclination=inc,
                         coa_phase=phi0,
                         delta_f=Fs,
                         f_lower=flower,
                         mode_array=modes)
    #rescale the time/frequency and amplitude
    f = hpf.sample_frequencies /(M1/M0) 
    hpf=types.FrequencySeries(hpf.data.data[:],delta_f=hpf.delta_f/rescale)*rescale**2
    hcf=types.FrequencySeries(hcf.data.data[:],delta_f=hcf.delta_f/rescale)*rescale**2

    #htilde = h+ / Ylm
    l,m =modes[0]
    #print('l,m=',l,m)
    htilde = htilde_lm(hpf,l,m,inc,phi0)
    phase = phase_from_frequencyseries(htilde,remove_start_phase=False)
    amp = amplitude_from_frequencyseries(htilde)
    wf = [phase.sample_frequencies, amp, -phase+phase[-1]]
    return wf



def htilde_lm(hp,l,m,inc,phi0):
    factorp=1/2.* (lal.SpinWeightedSphericalHarmonic(inc,phi0,-2,l,m)+\
                 (-1)**l *np.conj(lal.SpinWeightedSphericalHarmonic(inc,phi0,-2,l,-m)))
    #print(factorp)
    hlm = hp/factorp
    return hlm



def get_fd_LISATDI(freq, wf, inc, lambd, beta, psi, phi0,modes, t0=0.,trajdict=trajdict_MLDC, TDItag='TDIXYZ', order_fresnel_stencil=0, rescaled=False, frozenLISA=False, responseapprox='full', L=2.5e9):
    """
    @param freq freq array for which the waveform is computed
    @wf list containing wavf freq, ampl, phase
    @inc (iota_0) inclination
    @lam ecliptic Longitude
    @beta ecliptic longitude
    @phi0 (varphi_0) direction to the observer in the source frame
    """
    # Waveform given in the form of downsampled amplitude and phase, ready to be interpolated
    # The frequencies are assumed to have been chosen to ensure sufficient sampling at low and high frequencies
    
    wf_freq = wf[0]
    wf_amp = wf[1]
    wf_phase = wf[2]
    n = len(freq)
    spline_amp = spline(wf_freq, wf_amp)
    spline_phase = spline(wf_freq, wf_phase)
    amp = spline_amp(freq)
    phase = spline_phase(freq)

    # Wave unit vector
    kvec = funck(lambd, beta)

    # Trajectories
    funcp0 = trajdict['funcp0']

    # Compute constant matrices Hplus and Hcross in the SSB frame
    O1 = funcO1(lambd, beta, psi)
    invO1 = funcinverseO1(lambd, beta, psi)
    Hplus = np.dot(O1, np.dot(HSplus, invO1))
    Hcross = np.dot(O1, np.dot(HScross, invO1))

    # Build function tf by interpolating FD phase
    tfspline = spline(freq, 1/(2*pi)*(phase - phase[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
    tfvec = tfspline(freq)

    # Spin-weighted spherical harmonics prefactors for plus and cross FD
    l,m= modes[0]
    #print(l,m)
    Ylm = SpinWeightedSphericalHarmonic(-2, l, m, inc, phi0)
    Ylmstar = np.conjugate(SpinWeightedSphericalHarmonic(-2, l, -m, inc, phi0))
    Yfactorplus = 1./2 * (Ylm + Ylmstar)
    Yfactorcross = 1j/2 * (Ylm - Ylmstar)
    # The matrix H is now complex
    H = Yfactorplus*Hplus + Yfactorcross*Hcross

    # Initialize output
    wfTDI = {}
    wfTDI['TDItag'] = TDItag
    wfTDI['t0'] = t0
    wfTDI['freq'] = np.copy(freq)
    wfTDI['amp'] = np.copy(amp)
    wfTDI['phase'] = np.copy(phase)
    wfTDI['phaseRdelay'] = np.zeros(n, dtype=np.float64)
    wfTDI['transferL1'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL2'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL3'] = np.zeros(n, dtype=np.complex128)

    # For higher-order corrections: Fresnel tranform stencil
    if order_fresnel_stencil>20:
        raise ValueError('Only order_fresnel_stencil<=20 is supported for now.')
    if order_fresnel_stencil>=1:
        coeffs = np.array(coeffs_fresnel_stencil[order_fresnel_stencil])
        coeffs_array = np.concatenate((coeffs[1:][::-1], 2*coeffs[0:1], coeffs[1:])) / 2
        # Values and signs of Tf
        Tfvec, epsTfvec = func_compute_Tf(freq, phase)
    # Main loop over frequencies
    for i in range(n):
        f = freq[i]
        if not frozenLISA:
            t = tfvec[i] + t0*YRSID_SI # t0 is the orbital time at merger - t is the SSB time
        else:
            t = t0*YRSID_SI
        # Phase of transfer at leading order in the delays and Fresnel correction - we keep the phase separate because it is easy to interpolate as a phase but can be less so in Re/Im form
        p0 = funcp0(t)
        kR = np.dot(kvec, p0)
        if responseapprox=='lowf':
            phaseRdelay = 0.
        else:
            phaseRdelay = 2*pi/C_SI*f*kR
        # Transfer function
        if order_fresnel_stencil>=1:
            if frozenLISA:
                raise ValueError('Options frozenLISA and order_fresnel_stencil>=1 are incompatible.')
            Gslr = {}
            Tf = Tfvec[i] # used for higher-order correction of Fresnel type
            epsTf = epsTfvec[i] # keeping track of the sign associated with Tf
            if epsTf==1. or epsTf==0.:
                coeffs_array_signeps = coeffs_array
            else:
                coeffs_array_signeps = np.conj(coeffs_array)
            tvec = t + Tf * np.arange(-order_fresnel_stencil, order_fresnel_stencil+1)
            Gslrvec = EvaluateGslr(tvec, f, H, kvec, trajdict=trajdict, responseapprox=responseapprox, L=L)
            for key in Gslrvec:
                Gslr[key] = np.dot(coeffs_array_signeps, Gslrvec[key])
        else:
            Gslr = EvaluateGslr(t, f, H, kvec, trajdict=trajdict, responseapprox=responseapprox, L=L)
        # Scale out the leading-order correction from the orbital delay term, that we keep separate
        Tslr = {}
        for key in Gslr:
            Tslr[key] = Gslr[key] * exp(-1j*phaseRdelay)
        # Build TDI combinations
        tdi = TDICombinationFD(Tslr, f, TDItag=TDItag, rescaled=rescaled, responseapprox=responseapprox, L=L)
        wfTDI['phaseRdelay'][i] = phaseRdelay
        wfTDI['transferL1'][i] = tdi['transferL1']
        wfTDI['transferL2'][i] = tdi['transferL2']
        wfTDI['transferL3'][i] = tdi['transferL3']

    # Combine everything to produce final TDI
    # NOTE: added here for plotting purposes - in the normal usage, the FD sampling will be insufficient to represent the oscillatory TDI quantities - one should interpolate all the pieces on the wanted frequencies first, before combining them
    ampphasefactor = wfTDI['amp'] * np.exp(1j*(wfTDI['phase'] + wfTDI['phaseRdelay']))
    wfTDI['TDI1'] = ampphasefactor * wfTDI['transferL1']
    wfTDI['TDI2'] = ampphasefactor * wfTDI['transferL2']
    wfTDI['TDI3'] = ampphasefactor * wfTDI['transferL3']

    # Output
    return wfTDI



# Computing given TDI channel by interpolation
def func_wfTDI(freqs, wfTDI):
    res = np.zeros(len(freqs), dtype=complex)
    wftdi = {}
    wftdi['TDItag'] = wfTDI['TDItag']
    wftdi['t0'] = wfTDI['t0']
    #####################
    fbeg = max(freqs[0], wfTDI['freq'][0])
    fend = min(freqs[-1], wfTDI['freq'][-1])
    ibeg = np.where(freqs>=fbeg)[0][0]
    iend = np.where(freqs<=fend)[0][-1]
    fs = freqs[ibeg:iend+1]
    n=len(fs)
    t0 = wfTDI['t0'] # will be used to apply timeshift
    ampspline = spline(wfTDI['freq'], wfTDI['amp'])
    phasespline = spline(wfTDI['freq'], wfTDI['phase'])
    phaseRdelayspline = spline(wfTDI['freq'], wfTDI['phaseRdelay'])
    amp = ampspline(fs)
    phase = phasespline(fs)
    phaseRdelay = phaseRdelayspline(fs)
    wftdi['freq'] = np.copy(fs)
    wftdi['amp'] = np.copy(amp)
    wftdi['phase'] = np.copy(phase)
    wftdi['phaseRdelay'] = np.copy(phaseRdelay)
    #wftdi['transferL1'] = np.zeros(n, dtype=np.complex128)
    #wftdi['transferL2'] = np.zeros(n, dtype=np.complex128)
    #wftdi['transferL3'] = np.zeros(n, dtype=np.complex128)

    for chan in [1,2,3]:
        keytransfer = 'transferL'+str(chan)
        transferLRespline = spline(wfTDI['freq'], np.real(wfTDI[keytransfer]))
        transferLImspline = spline(wfTDI['freq'], np.imag(wfTDI[keytransfer]))
        transferLRe = transferLRespline(fs)
        transferLIm = transferLImspline(fs)
        wftdi[keytransfer]= transferLRe+1j*transferLIm 
        phasetimeshift = 2*pi*t0*fs
        phasetimeshift=0
        vals = (transferLRe+1j*transferLIm) * amp * np.exp(1j*(phase+phaseRdelay+phasetimeshift))
        res[ibeg:iend+1] = vals
        keyTDI = 'TDI' +str(chan)
        # TDIRespline = spline(wfTDI['freq'], np.real(wfTDI[keyTDI]))
        # TDIImspline = spline(wfTDI['freq'], np.imag(wfTDI[keyTDI]))
        # TDIRe = TDIRespline(fs)
        # TDIIm = TDIImspline(fs)
        # vals = TDIRe + 1j*TDIIm
        # res[ibeg:iend+1]= vals
        wftdi[keyTDI]= np.copy(res)
    return wftdi






def get_TDI(chirpmass,q,distance,inc,phi0,chi1,chi2,lambd,beta,psi,t0,trajdict,TDItag,apx,modes,df=2e-7):
    """
    Generates TDIs (X, Y, Z) or(A, E, T) in Fourrier Domain 
    @return freq is array of frequency
    @return complex array of TDI X or A
    @return complex array of TDI Y or E 
    @return complex array of TDI Z or T
    """
    #htilde
    #print(modes)
    hlm = {}
    minindex=0
    for i in range(len(modes)):
        l,m= modes[i]
        #print(l,m)
        hlm[l,m]= get_fd_htilde_lm(chirpmass,q,distance,inc,phi0,chi1,chi2,apx,[modes[i]],df)
        tmp_minindex = np.argmax(hlm[l,m][2])
        minindex = max(tmp_minindex,minindex) 
    
    #AET
    wftdi={}
    wftdi_int={}
    m1=mass1_from_mchirp_q(chirpmass,q)
    m2=mass2_from_mchirp_q(chirpmass,q)
    MfCUT_PhenomD = 0.2
    Ms = (m1 + m2) * MTSUN_SI
    fcut = MfCUT_PhenomD/Ms
    freq = logsampling(hlm[l,m][2].sample_frequencies[minindex],fcut,1000)
    del_t = 5
    fsample_SMBH = 1./del_t
    fnyquist_SMBH = 1./2*fsample_SMBH     #
    df =df
    f0 = np.floor(freq[0]/df)*df
    #print('fmin, fmax, df',f0,fcut,df)
    n_SMBH= int((freq[-1]-f0)/df)
    freqs_SMBH = df*np.arange(n_SMBH) +f0
    freqs_SMBH = freqs_SMBH[(freqs_SMBH>=freq[0]) & (freqs_SMBH<=fnyquist_SMBH)]

    for i in range(len(modes)):
        l,m= modes[i]
        wftdi[l,m]= get_fd_LISATDI(freq,hlm[l,m],inc,lambd,beta,psi,phi0,[[l,m]],t0=t0, trajdict=trajdict, TDItag=TDItag)
        wftdi_int[l,m] = func_wfTDI(freqs_SMBH, wftdi[l,m])

    #sum of all modes
    Nf = int (fnyquist_SMBH/df +1)
    fre = df*np.arange(Nf)
    i_b = int(np.rint(freqs_SMBH[0]/df))
    TDI1 = np.zeros(Nf, dtype='complex128')
    TDI2 = np.zeros(Nf, dtype='complex128')
    TDI3 = np.zeros(Nf, dtype='complex128')

    for i in range(len(modes)):
        l,m= modes[i]
        TDI1[i_b:len(freqs_SMBH)+ i_b] += wftdi_int[l,m]['TDI1']
        TDI2[i_b:len(freqs_SMBH)+i_b] += wftdi_int[l,m]['TDI2']
        TDI3[i_b:len(freqs_SMBH)+i_b] += wftdi_int[l,m]['TDI3']
    return fre, TDI1, TDI2, TDI3 
