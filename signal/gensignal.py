# coding=utf-8
import numpy as np
import sys,os


from pycbc import types
from pycbc.waveform import get_td_waveform,get_fd_waveform,get_fd_waveform_sequence
from pycbc.conversions import mass1_from_mchirp_q,mass2_from_mchirp_q
from pycbc.filter import interpolate_complex_frequency, resample_to_delta_t
from pycbc.waveform.utils import phase_from_frequencyseries,amplitude_from_frequencyseries
from pycbc.types import real_same_precision_as
import lal

from pyFDresponse import *
#from matplotlib.ticker import LinearLocator
from astropy import units as u
from astropy.coordinates import SkyCoord
#plt.switch_backend('agg')



# Cartesian SSB components of spacecraft positions
path =os.path.abspath(os.path.dirname(__file__))
p1 = np.fromfile(path+'/LISA_orbit_data/SCP1_1yr.dat',sep=' ')
p2 = np.fromfile(path+'/LISA_orbit_data/SCP2_1yr.dat',sep=' ')
p3 = np.fromfile(path+'/LISA_orbit_data/SCP3_1yr.dat',sep=' ')
#print(p1)
#print(len(p1))
p1= p1.reshape(int(len(p1)/3),3)
p2= p2.reshape(int(len(p2)/3),3)
p3= p3.reshape(int(len(p3)/3),3)

c1= SkyCoord(x=p1[:,0],y=p1[:,1],z=p1[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
c2= SkyCoord(x=p2[:,0],y=p2[:,1],z=p2[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
c3= SkyCoord(x=p3[:,0],y=p3[:,1],z=p3[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
#print(c1)
#transfrom to Ecliptic system of coordinate
p1_SSB= c1.heliocentrictrueecliptic
p2_SSB= c2.heliocentrictrueecliptic
p3_SSB= c3.heliocentrictrueecliptic
p1_SSB.representation_type='cartesian'
p2_SSB.representation_type='cartesian'
p3_SSB.representation_type='cartesian'
#interpolate 
def funcp_SSB(t,p_SSB):
    year=np.linspace(0,LC.YRSID_SI,366)
    px_SSB= spline(year,p_SSB.x)
    py_SSB= spline(year,p_SSB.y)
    pz_SSB= spline(year,p_SSB.z)
    return np.array([px_SSB(t),py_SSB(t),pz_SSB(t)]) 

#Computing the arm length
def funcL_SSB(p1_SSB,p2_SSB):
    Ln = p2_SSB- p1_SSB
    L = np.sqrt(Ln[0]**2+Ln[1]**2+Ln[2]**2)
    return L

def funcp0_SSB(t,p1_SSB=p1_SSB,p2_SSB=p2_SSB,p3_SSB=p3_SSB):
    p1= funcp_SSB(t,p1_SSB)*R_SI
    p2= funcp_SSB(t,p2_SSB)*R_SI
    p3= funcp_SSB(t,p3_SSB)*R_SI
    p0 = (p1+p2+p3)/3
    return p0

trajdict_numerical_orbits = {
    'funcp0':funcp0_SSB,
    'funcp': funcp_SSB,
    'funcL': funcL_SSB,
    'p1_SSB': p1_SSB,
    'p2_SSB': p2_SSB,
    'p3_SSB': p3_SSB,
    }

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


def get_fd_sequence_htilde_lm(fre,chirpmass,q,distance,inc,phi0,chi1,chi2,apx,modes):
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
    rescale= M1/M0

    hpf, hcf = get_fd_waveform_sequence(approximant=apx,
                         mass1=m10,
                         mass2=m20,
                         spin1z=chi1,
                         spin2z=chi2,
                         distance=distance,
                         inclination=inc,
                         coa_phase=phi0,
                         sample_points=fre*rescale,
                         mode_array=modes)
    #rescale the time/frequency and amplitude
    hpf= hpf*rescale**2
    hcf= hcf*rescale**2

    #htilde = h+ / Ylm
    l,m =modes[0]
    #print('l,m=',l,m)
    htilde = htilde_lm(hpf,l,m,inc,phi0)
    phase = np.unwrap(np.angle(htilde)).astype(real_same_precision_as(htilde))
    amp = abs(htilde).astype(real_same_precision_as(htilde))
    wf = [fre, amp, -phase+phase[-1]]
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
    # For now, we consider single-mode h22 waveforms only
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
    Yfactorplus = 1./2 * (Ylm + (-1)**l*Ylmstar)
    Yfactorcross = 1j/2 * (Ylm - (-1)**l*Ylmstar)
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
        # tdi = CombinationFD(Tslr, f, TDItag=TDItag, rescaled=rescaled, responseapprox=responseapprox, L=L)
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
        #phasetimeshift=0
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






def get_TDI(chirpmass,q,distance,inc,phi0,chi1,chi2,lambd,beta,psi,t0,trajdict,TDItag,apx,modes,df=2e-7,del_t = 1.0):
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

    ####
    del_t = del_t
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





# Fourier-domain LISA response, perturbative formalism
def get_FR_TDI(chirpmass,q,distance,inc,phi0,chi1,chi2,lambd,beta,psi,t0,apx,modes,trajdict, TDItag='TDIXYZ',number_orbits=False,TDI_froze_arm= False ,order_fresnel_stencil=0, rescaled=False, frozenLISA=False, responseapprox='full', L=2.5e9,df=2e-6,del_t=1.0):
    """
    Generates full response TDIs (X, Y, Z) or(A, E, T) in Fourrier Domain 
    @return freq is array of frequency
    @return complex array of TDI X or A
    @return complex array of TDI Y or E 
    @return complex array of TDI Z or T
    """

    m1=mass1_from_mchirp_q(chirpmass,q)
    m2=mass2_from_mchirp_q(chirpmass,q)
    M1=m1+m2
    fmax = 0.6 /(M1*MTSUN_SI)
    fmin = 1e-4 / (M1*MTSUN_SI)

    hlm = {}
    tmp_fmin=1e-5
    for i in range(len(modes)):
        l,m= modes[i]
        #print(l,m)
        # f = logsampling(fmin,fmax,int(5*10**4*m*l/2))
        f1 = logsampling(fmin,fmin*1e2,int(2*10**4*m*l/2))
        f2 = np.linspace(fmin*1e2,fmax,int(1000))
        f2 = np.delete(f2,0)
        f = np.hstack((f1,f2))
        fre = types.array.Array(f)
        hlm[l,m]= get_fd_sequence_htilde_lm(fre,chirpmass,q,distance,inc,phi0,chi1,chi2,apx,[modes[i]])
        tmp_minindex = np.argmax(hlm[l,m][2])
        tmp_fmin = max(fre[tmp_minindex],tmp_fmin)
    
    #AET
    wftdi={}
    wftdi_int={}

    freq = logsampling(tmp_fmin,fmax,1000)

    ####
    del_t = del_t
    fsample_SMBH = 1./del_t
    fnyquist_SMBH = 1./2*fsample_SMBH     #
    #df =df
    f0 = np.floor(freq[0]/df)*df
    #print('fmin, fmax, df',f0,fcut,df)
    n_SMBH= int((freq[-1]-f0)/df)
    freqs_SMBH = df*np.arange(n_SMBH) +f0
    freqs_SMBH = freqs_SMBH[(freqs_SMBH>=freq[0]) & (freqs_SMBH<=fnyquist_SMBH)]

    for i in range(len(modes)):
        l,m= modes[i]
        #wftdi[l,m]= get_fd_LISATDI(freq,hlm[l,m],inc,lambd,beta,psi,phi0,[[l,m]],t0=t0, trajdict=trajdict, TDItag=TDItag)
        wftdi[l,m]=FDresponseTDI(freq,hlm[l,m],inc,lambd,beta,psi,phi0,[[l,m]],t0=t0, trajdict=trajdict, TDItag=TDItag,number_orbits=number_orbits,TDI_froze_arm=TDI_froze_arm,order_fresnel_stencil=order_fresnel_stencil,frozenLISA=frozenLISA,rescaled=rescaled,responseapprox=responseapprox)
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


def FDresponseTDI(freq, wf, inc, lambd, beta, psi, phi0,modes, t0,trajdict, TDItag='TDIXYZ',number_orbits=False, TDI_froze_arm= False ,order_fresnel_stencil=0, rescaled=False, frozenLISA=False, responseapprox='full', L=2.5e9):
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
    # For now, we consider single-mode h22 waveforms only
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
    if number_orbits :
        funcp = trajdict['funcp']   
        funcL = trajdict['funcL']
        p1_SSB = trajdict['p1_SSB']     #AU
        p2_SSB = trajdict['p2_SSB']
        p3_SSB = trajdict['p3_SSB']

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
    Yfactorplus = 1./2 * (Ylm + (-1)**l*Ylmstar)
    Yfactorcross = 1j/2 * (Ylm - (-1)**l*Ylmstar)
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
    if frozenLISA:
        t = t0*YRSID_SI
        if number_orbits :
            # In our approximation, all positions are evaluated at t
            p1 = funcp(t,p1_SSB) *R_SI
            p2 = funcp(t,p2_SSB) *R_SI
            p3 = funcp(t,p3_SSB) *R_SI
            p0 = (p1+p2+p3)/3
            L12= funcL_SSB(p1,p2)
            L23= funcL_SSB(p2,p3)
            L13= funcL_SSB(p1,p3)
            trajdict['p1']= p1
            trajdict['p2']= p2
            trajdict['p3']= p3
            trajdict['L12']= L12
            trajdict['L23']= L23
            trajdict['L13']= L13
        else:
            p0 = funcp0(t)




    for i in range(n):
        f = freq[i]
        if not frozenLISA:
            t = tfvec[i] + t0*YRSID_SI # t0 is the orbital time at merger - t is the SSB time
            if number_orbits:
                # In our approximation, all positions are evaluated at t
                p1 = funcp(t,p1_SSB) *R_SI
                p2 = funcp(t,p2_SSB) *R_SI
                p3 = funcp(t,p3_SSB) *R_SI
                p0 = (p1+p2+p3)/3
                L12= funcL_SSB(p1,p2)
                L23= funcL_SSB(p2,p3)
                L13= funcL_SSB(p1,p3)
                trajdict['p1']= p1
                trajdict['p2']= p2
                trajdict['p3']= p3
                trajdict['L12']= L12
                trajdict['L23']= L23
                trajdict['L13']= L13
            else:
                p0 = funcp0(t)
        # else:
        #     t = t0*YRSID_SI
        # Phase of transfer at leading order in the delays and Fresnel correction - we keep the phase separate because it is easy to interpolate as a phase but can be less so in Re/Im form
    
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
            # Gslrvec = EvaluateGslr(tvec, f, H, kvec, trajdict=trajdict, responseapprox=responseapprox, L=L)
            Gslrvec = get_Gslr(tvec, f, H, kvec, trajdict=trajdict, responseapprox=responseapprox, L=L)
            for key in Gslrvec:
                Gslr[key] = np.dot(coeffs_array_signeps, Gslrvec[key])
        else:
            # Gslr = EvaluateGslr(t, f, H, kvec, trajdict=trajdict, responseapprox=responseapprox, L=L)
            Gslr = get_Gslr(t, f, H, kvec, trajdict=trajdict, responseapprox=responseapprox, L=L)
        # Scale out the leading-order correction from the orbital delay term, that we keep separate
        Tslr = {}
        for key in Gslr:
            #Tslr[key] = Gslr[key] * exp(-1j*phaseRdelay)
            Tslr[key] = Gslr[key]
        # Build TDI combinations
        if TDI_froze_arm :
            tdi = CombinationFD(Tslr, f,trajdict=trajdict_MLDC,TDItag=TDItag, rescaled=rescaled, responseapprox=responseapprox, L=L)
        else:
            tdi = CombinationFD(Tslr, f, trajdict=trajdict,TDItag=TDItag, rescaled=rescaled, responseapprox=responseapprox, L=L)
        wfTDI['phaseRdelay'][i] = phaseRdelay
        wfTDI['transferL1'][i] = tdi['transferL1']
        wfTDI['transferL2'][i] = tdi['transferL2']
        wfTDI['transferL3'][i] = tdi['transferL3']

    #print('armlength',L12,L23,L13)
    # Combine everything to produce final TDI
    # NOTE: added here for plotting purposes - in the normal usage, the FD sampling will be insufficient to represent the oscillatory TDI quantities - one should interpolate all the pieces on the wanted frequencies first, before combining them
    ampphasefactor = wfTDI['amp'] * np.exp(1j*(wfTDI['phase']))
    wfTDI['TDI1'] = ampphasefactor * wfTDI['transferL1']
    wfTDI['TDI2'] = ampphasefactor * wfTDI['transferL2']
    wfTDI['TDI3'] = ampphasefactor * wfTDI['transferL3']

    # Output
    return wfTDI




# Single-link response
# 'full' does include the orbital-delay term, 'constellation' does not
# t can be a scalar or a 1D vector
def get_Gslr(t, f, H, k, trajdict=trajdict_MLDC, responseapprox='full', L=2.5e9):
    # Trajectories, p0 used only for the full response
    if trajdict== trajdict_MLDC:
        funcp0 = trajdict['funcp0']
        funcn1 = trajdict['funcn1']
        funcn2 = trajdict['funcn2']
        funcn3 = trajdict['funcn3']
        funcp1L = trajdict['funcp1L']
        funcp2L = trajdict['funcp2L']
        funcp3L = trajdict['funcp3L']
        funcp1L = trajdict['funcp1L']
        funcp2L = trajdict['funcp2L']
        funcp3L = trajdict['funcp3L']
        # Here t is a scalar or a vector - the trajectory functions are vectorial
        # If t is a N-vector, then these are 3-vectors with each component a N-vector (shape (3,N))
        p0 = funcp0(t)
        p1L = funcp1L(t)
        p2L = funcp2L(t)
        p3L = funcp3L(t)
        n1 = funcn1(t)
        n2 = funcn2(t)
        n3 = funcn3(t)
        p1 = p0+ p1L
        p2 = p0+ p2L
        p3 = p0+ p3L
        L12=L23=L13=L
        #print(L12)
    else:
        # funcp = trajdict['funcp']   
        # funcL = trajdict['funcL']
        # p1_SSB = trajdict['p1_SSB']     #AU
        # p2_SSB = trajdict['p2_SSB']
        # p3_SSB = trajdict['p3_SSB']
        # In our approximation, all positions are evaluated at t
        p1 = trajdict['p1'] 
        p2 = trajdict['p2'] 
        p3 = trajdict['p3'] 
        p0 = (p1+p2+p3)/3
        L12= trajdict['L12'] 
        L23= trajdict['L23']
        L13= trajdict['L13']
        #L12=L23=L13=L
        n1= (p3-p2)/L23   #n23
        n2= (p1-p3)/L13   #n31
        n3= (p2-p1)/L12   #n12


    
    # Compute intermediate scalar products
    # # Distinguishing wether t is a scalar or a vector
    if isinstance(t, np.ndarray): # t vector case
        # Note: np.einsum comes with a startup cost that is excessive here, as we do an external loop on frequencies - revert to ugly for loop with dot
        n = len(t)
        # Transpose vectors to shape (N,3)
        n1T = n1.T
        n2T = n2.T
        n3T = n3.T
        p1LT = p1L.T
        p2LT = p2L.T
        p3LT = p3L.T
        p0T = p0.T
        # H is a fixed 3*3 matrix, not vectorialized
        # n1Hn1 = np.einsum('ik,ij,jk->k', n1, H, n1)
        # n2Hn2 = np.einsum('ik,ij,jk->k', n2, H, n2)
        # n3Hn3 = np.einsum('ik,ij,jk->k', n3, H, n3)
        n1Hn1 = np.zeros(n, dtype='complex')
        n2Hn2 = np.zeros(n, dtype='complex')
        n3Hn3 = np.zeros(n, dtype='complex')
        for i in range(n):
            n1Hn1[i] = np.dot(n1T[i], np.dot(H, n1T[i]))
            n2Hn2[i] = np.dot(n2T[i], np.dot(H, n2T[i]))
            n3Hn3[i] = np.dot(n3T[i], np.dot(H, n3T[i]))
            # if (i<10):
            #     print (i, n1Hn1[i], n2Hn2[i], n3Hn3[i])
        # k is a fixed 3-vector, not vectorialized
        # kn1 = np.einsum('i,ij->j', k, n1)
        # kn2 = np.einsum('i,ij->j', k, n2)
        # kn3 = np.einsum('i,ij->j', k, n3)
        kn1 = np.zeros(n, dtype='float')
        kn2 = np.zeros(n, dtype='float')
        kn3 = np.zeros(n, dtype='float')
        for i in range(n):
            kn1[i] = np.dot(k, n1T[i])
            kn2[i] = np.dot(k, n2T[i])
            kn3[i] = np.dot(k, n3T[i])
            # if (i<10):
            #     print (i, kn1[i], kn2[i], kn3[i])
        # kp1Lp2L = np.einsum('i,ij->j', k, (p1L+p2L))
        # kp2Lp3L = np.einsum('i,ij->j', k, (p2L+p3L))
        # kp3Lp1L = np.einsum('i,ij->j', k, (p3L+p1L))
        # kp0 = np.einsum('i,ij->j', k, p0)
        kp1Lp2L = np.zeros(n, dtype='float')
        kp2Lp3L = np.zeros(n, dtype='float')
        kp3Lp1L = np.zeros(n, dtype='float')
        kp0 = np.zeros(n, dtype='float')
        for i in range(n):
            kp1Lp2L[i] = np.dot(k, p1LT[i] + p2LT[i])
            kp2Lp3L[i] = np.dot(k, p2LT[i] + p3LT[i])
            kp3Lp1L[i] = np.dot(k, p3LT[i] + p1LT[i])
            kp0[i] = np.dot(k, p0T[i])
    else: # t scalar case
        kn1 = np.dot(k, n1)
        kn2 = np.dot(k, n2)
        kn3 = np.dot(k, n3)
        n1Hn1 = np.dot(n1, np.dot(H, n1))
        n2Hn2 = np.dot(n2, np.dot(H, n2))
        n3Hn3 = np.dot(n3, np.dot(H, n3))
        kp1Lp2L = np.dot(k, (p1+p2))
        kp2Lp3L = np.dot(k, (p2+p3))
        kp3Lp1L = np.dot(k, (p3+p1))
        kp0 = np.dot(k, p0)

    # Main orbital delay term -- also called Doppler shift
    if responseapprox=='full' or responseapprox=='lowfL':
        factorcexp0 = exp(1j*2*pi*f/C_SI * kp0)
    elif responseapprox=='lowf':
        factorcexp0 = 1.
    else:
        raise Exception('Keyword responseapprox=%s not recognized.' % (responseapprox))

    # Compute prefactors
    #L12=L23=L13=L
    # L12= L * (1-0.01)
    # L23= L * (1-0.01)
    # L13= L * (1+0.01)
    prefactor12 = pi*f*L12/C_SI
    prefactor23 = pi*f*L23/C_SI
    prefactor31 = pi*f*L13/C_SI
    if responseapprox=='full':
        factorcexp12 = exp(1j*prefactor12 * (1.+kp1Lp2L/L12))
        factorcexp23 = exp(1j*prefactor23 * (1.+kp2Lp3L/L23))
        factorcexp31 = exp(1j*prefactor31 * (1.+kp3Lp1L/L13))
        factorsinc12 = sinc( prefactor12 * (1.-kn3))
        factorsinc21 = sinc( prefactor12 * (1.+kn3))
        factorsinc23 = sinc( prefactor23 * (1.-kn1))
        factorsinc32 = sinc( prefactor23 * (1.+kn1))
        factorsinc31 = sinc( prefactor31 * (1.-kn2))
        factorsinc13 = sinc( prefactor31 * (1.+kn2))
    elif responseapprox=='lowf' or responseapprox=='lowfL':
        factorcexp12 = 1.
        factorcexp23 = 1.
        factorcexp31 = 1.
        factorsinc12 = 1.
        factorsinc21 = 1.
        factorsinc23 = 1.
        factorsinc32 = 1.
        factorsinc31 = 1.
        factorsinc13 = 1.
    else:
        raise Exception('Keyword responseapprox=%s not recognized.' % (responseapprox))


    # Compute the Gslr - either scalars or vectors
    #prefactor = pi*f*L/C_SI
    #commonfac = 1j * prefactor * factorcexp0
    G12 = 1j*prefactor12 * n3Hn3 * factorsinc12 * factorcexp12
    G21 = 1j*prefactor12 * n3Hn3 * factorsinc21 * factorcexp12
    G23 = 1j*prefactor23 * n1Hn1 * factorsinc23 * factorcexp23
    G32 = 1j*prefactor23 * n1Hn1 * factorsinc32 * factorcexp23
    G31 = 1j*prefactor31 * n2Hn2 * factorsinc31 * factorcexp31
    G13 = 1j*prefactor31 * n2Hn2 * factorsinc13 * factorcexp31

    # Output dictionary - either scalars or vectors
    Gslr = {}
    Gslr[(1,2)] = G12
    Gslr[(2,1)] = G21
    Gslr[(2,3)] = G23
    Gslr[(3,2)] = G32
    Gslr[(3,1)] = G31
    Gslr[(1,3)] = G13
    return Gslr



# Fourier-domain TDI combinations
def CombinationFD(Gslr, f,  trajdict=trajdict_MLDC,TDItag='TDIXYZ', rescaled=False, responseapprox='full', L=2.5e9):
    x = pi*f*L/C_SI;
    z = exp(2*1j*x);
    if trajdict == trajdict_MLDC:
        L12=L23=L13=L
        #print(L12)
    else:
        L12= trajdict['L12'] 
        L23= trajdict['L23']
        L13= trajdict['L13']
        

    x12 = pi*f*L12/C_SI
    x13 = pi*f*L13/C_SI
    x23 = pi*f*L23/C_SI
    z12=z21= np.exp(2*1j*x12)
    z13=z31= np.exp(2*1j*x13)
    z23=z32= np.exp(2*1j*x23)
    if responseapprox=='full':
        if rescaled:
            factcaor= 1.
            factor12=factor21=factor13=factor31=factor23=factor32=1.
        else:
            factor = 2*1j*sin(2.*x)
            factor12=factor21 = 2*1j*sin(2.*x12)
            factor13=factor31 = 2*1j*sin(2.*x13)
            factor23=factor32 = 2*1j*sin(2.*x23)

    elif responseapprox=='lowf' or responseapprox=='lowfL':
        if rescaled:
            factor = 1.
            factor12=factor21=factor13=factor31=factor23=factor32=1.
        else:
            factor = 2*1j*2.*x
            factor12=factor21 = 2*1j*2.*x12
            factor13=factor31 = 2*1j*2.*x13
            factor23=factor32 = 2*1j*2.*x23
        z = 1.
        z12=z21=z13=z31=z23=z32=1
    else:
        raise Exception('Keyword responseapprox=%s not recognized.' % (responseapprox))

    X = (-z21 + z21*z13*z31)*Gslr[(1,2)]*factor12 + (-1 + z13*z31)*Gslr[(2,1)]*factor21 + (z31 - z31*z12*z21)*Gslr[(1,3)]*factor13 + (1 - z12*z21)*Gslr[(3,1)]*factor31
    Y = (-z32 + z32*z21*z12)*Gslr[(2,3)]*factor23 + (-1 + z21*z12)*Gslr[(3,2)]*factor32 + (z12 - z12*z23*z32)*Gslr[(2,1)]*factor21 + (1 - z23*z32)*Gslr[(1,2)]*factor12
    Z = (-z13 + z13*z32*z23)*Gslr[(3,1)]*factor31 + (-1 + z32*z23)*Gslr[(1,3)]*factor13 + (z23 - z23*z31*z13)*Gslr[(3,2)]*factor32 + (1 - z31*z13)*Gslr[(2,3)]*factor23
    # X = (Gslr[(3,1)] + z*Gslr[(1,3)] - Gslr[(2,1)] - z*Gslr[(1,2)] + z*z*Gslr[(2,1)] + z*z*z*Gslr[(1,2)] - z*z*Gslr[(3,1)] - z*z*z*Gslr[(1,3)])
    # Y = (Gslr[(1,2)] + z*Gslr[(2,1)] - Gslr[(3,2)] - z*Gslr[(2,3)] + z*z*Gslr[(3,2)] + z*z*z*Gslr[(2,3)] - z*z*Gslr[(1,2)] - z*z*z*Gslr[(2,1)])
    # Z = (Gslr[(2,3)] + z*Gslr[(3,2)] - Gslr[(1,3)] - z*Gslr[(3,1)] + z*z*Gslr[(1,3)] + z*z*z*Gslr[(3,1)] - z*z*Gslr[(2,3)] - z*z*z*Gslr[(3,2)])
    # X = (Gslr[(2,1)] + z*Gslr[(1,2)] - Gslr[(3,1)] - z*Gslr[(1,3)])*z
    # Y = (Gslr[(3,2)] + z*Gslr[(2,3)] - Gslr[(1,2)] - z*Gslr[(2,1)])*z
    # Z = (Gslr[(1,3)] + z*Gslr[(3,1)] - Gslr[(2,3)] - z*Gslr[(3,2)])*z
    TDI = {}
    if TDItag=='TDIXYZ':
        # First-generation TDI XYZ
        # With x=pifL, factor scaled out: 2I*sin2x*e2ix
        # If using lowf approximations, keep leading order scaling of f*L/c
        # if responseapprox=='full':
        #     if rescaled:
        #         factor = 1.
        #     else:
        #         factor = 2*1j*sin(2.*x)
        # elif responseapprox=='lowf' or responseapprox=='lowfL':
        #     if rescaled:
        #         factor = 1.
        #     else:
        #         factor = 2*1j*2.*x
        #     z = 1.
        # else:
        #     raise Exception('Keyword responseapprox=%s not recognized.' % (responseapprox))
        # Xraw = Gslr[(2,1)] + z*Gslr[(1,2)] - Gslr[(3,1)] - z*Gslr[(1,3)]
        # Yraw = Gslr[(3,2)] + z*Gslr[(2,3)] - Gslr[(1,2)] - z*Gslr[(2,1)]
        # Zraw = Gslr[(1,3)] + z*Gslr[(3,1)] - Gslr[(2,3)] - z*Gslr[(3,2)]
        # TDI['transferL1'] = factor * Xraw
        # TDI['transferL2'] = factor * Yraw
        # TDI['transferL3'] = factor * Zraw
        TDI['transferL1'] = X  
        TDI['transferL2'] = Y  
        TDI['transferL3'] = Z  
        return TDI
    elif TDItag=='TDIAET':
        # First-generation TDI AET from X,Y,Z
        # With x=pifL, factors scaled out: A,E:I*sqrt2*sin2x*e2ix T:2*sqrt2*sin2x*sinx*e3ix
        # Here we include a factor 2, because the code was first written using the definitions (2) of McWilliams&al_0911 where A,E,T are 1/2 of their LDC definitions
        factor_convention = 2.
        # If using lowf approximations, keep leading order scaling of f*L/c
        # if responseapprox=='full':
        #     if rescaled:
        #         factorAE = 1.
        #         factorT = 1.
        #     else:
        #         factor = 2*1j*sin(2.*x)
        #         factorAE = 1j*sqrt2*sin(2.*x)*z
        #         factorT = 2.*sqrt2*sin(2.*x)*sin(x)*exp(1j*3.*x)
        # elif responseapprox=='lowf' or responseapprox=='lowfL':
        #     if rescaled:
        #         factorAE = 1.
        #         factorT = 1.
        #     else:
        #         factorAE = 1j*sqrt2*2.*x
        #         factorT = 2.*sqrt2*2.*x*x
        #     z = 1.
        # else:
        #     raise Exception('Keyword responseapprox=%s not recognized.' % (responseapprox))
        # Araw = 0.5 * ( (1.+z)*(Gslr[(3,1)] + Gslr[(1,3)]) - Gslr[(2,3)] - z*Gslr[(3,2)] - Gslr[(2,1)] - z*Gslr[(1,2)] )
        # Eraw = 0.5*invsqrt3 * ( (1.-z)*(Gslr[(1,3)] - Gslr[(3,1)]) + (2.+z)*(Gslr[(1,2)] - Gslr[(3,2)]) + (1.+2*z)*(Gslr[(2,1)] - Gslr[(2,3)]) )
        # Traw = invsqrt6 * ( Gslr[(2,1)] - Gslr[(1,2)] + Gslr[(3,2)] - Gslr[(2,3)] + Gslr[(1,3)] - Gslr[(3,1)])
        # TDI['transferL1'] = factor_convention * factorAE * Araw
        # TDI['transferL2'] = factor_convention * factorAE * Eraw
        # TDI['transferL3'] = factor_convention * factorT * Traw
        TDI['transferL1'] = 1/(sqrt2) * (Z - X)  *factor_convention *0.5
        TDI['transferL2'] = 1/(sqrt6) * (Z + X - 2*Y)  *factor_convention*0.5
        TDI['transferL3'] = 1/(sqrt3) * (Z + X + Y)  *factor_convention*0.5
        return TDI
    else:
        raise ValueError("Error in TDICombinationFD: TDItag not recognized.")