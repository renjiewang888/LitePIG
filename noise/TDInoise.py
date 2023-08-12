# coding=utf-8
import numpy as np
import sys
sys.path.append('/disk1/home/wangrj/LDC/MLDC-master/software/LDCpipeline/scripts')
import matplotlib.pyplot as plt

from pycbc import fft,types,frame,noise,filter
import pycbc.psd
import pylab
from tqdm import tqdm
from scipy import interpolate,special
from pycbc.psd.read import from_numpy_arrays
from pycbc import noise,frame
#from pyIMRPhenomD import IMRPhenomD, MSUN_SI, PC_SI
import Cosmology
import LISAConstants as LC
#import ParseWriteXML as pwXML
from LISAhdf5 import LISAhdf5,ParsUnits
from pyFDresponse import *
from psd import parse_arguments,Series
from tdi import *
from LISACommonFunctions import *
from GenerateFD_SignalTDIs import *
plt.switch_backend('agg')


#equal-arm Taiji
#Taiji TDI A/E
def noisepsd_AE_taiji(frequencydata):
    """
    Compute and return analytic PSD of noise for TDI A and E
     @param frequencydata  either numpy array or FrequencyArray of freqs.
    """
    f = frequencydata.f if isinstance(frequencydata,FrequencyArray) else frequencydata
    x = 2.0 * math.pi * LP.lisaLT * f

    Spm, Sop = taijinoises(f)

    Sa = 8.0 * np.sin(x)**2 * (2.0 * Spm * (3.0 + 2.0*np.cos(x) + np.cos(2*x)) + Sop * (2.0 + np.cos(x)))


    if isinstance(frequencydata,FrequencyArray):
        return FrequencyArray(Sa,kmin=frequencydata.kmin,df=frequencydata.df)
    else:
        return Sa


def noisepsd_T_taiji(frequencydata):
    """
    Compute and return analytic PSD of noise for TDI T
     @param frequencydata  either numpy array or FrequencyArray of freqs.
    """
    f = frequencydata.f if isinstance(frequencydata,FrequencyArray) else frequencydata
    x = 2.0 * math.pi * LP.lisaLT * f

    Spm, Sop = taijinoises(f)

    St = 16.0 * Sop * (1.0 - np.cos(x)) * np.sin(x)**2 + 128.0 * Spm * np.sin(x)**2 * np.sin(0.5*x)**4

    if isinstance(frequencydata,FrequencyArray):
        return FrequencyArray(St,kmin=frequencydata.kmin,df=frequencydata.df)
    else:
        return St



def noisepsd_unequal_A(frequencydata,L12,L23,L31,detector):
    """
    Compute and return  PSD of noise for static unequal arm 
     @param frequencydata  either numpy array or FrequencyArray of freqs.
    """
    f = frequencydata.f if isinstance(frequencydata,FrequencyArray) else frequencydata
    x21=x12 = 2.0 * math.pi * L12/LC.clight * f
    x32=x23 = 2.0 * math.pi * L23/LC.clight * f
    x13=x31 = 2.0 * math.pi * L31/LC.clight * f
    
    if detector == 'Taiji':
        Spm, Sop = taijinoises(f)
    elif detector == 'LISA':
        Spm, Sop = lisanoises(f, model='SciRDv1')


    Sa = 4*Sop*(np.sin(x12)**2 + 2*np.sin(x13)**2 + np.sin(x23)**2 + 2*np.sin(x12)*np.sin(x23)*np.cos(x13)*np.cos(x12-x23))+\
        16*Spm*(2*np.sin(x12)*np.sin(x23)*np.cos(x13)*np.cos(x12-x23)- np.cos(x12)**2*np.cos(x13)**2 - np.cos(x13)**2*np.cos(x23)**2+2)

    if isinstance(frequencydata,FrequencyArray):
        return FrequencyArray(Sa,kmin=frequencydata.kmin,df=frequencydata.df)
    else:
        return Sa


def noisepsd_unequal_E(frequencydata,L12,L23,L31,detector):
    """
    Compute and return  PSD of noise for static unequal arm 
     @param frequencydata  either numpy array or FrequencyArray of freqs.
    """
    f = frequencydata.f if isinstance(frequencydata,FrequencyArray) else frequencydata
    x21=x12 = 2.0 * math.pi * L12/LC.clight * f
    x32=x23 = 2.0 * math.pi * L23/LC.clight * f
    x13=x31 = 2.0 * math.pi * L31/LC.clight * f

    if detector == 'Taiji':
        Spm, Sop = taijinoises(f)
    elif detector == 'LISA':
        Spm, Sop = lisanoises(f, model='SciRDv1')

    Se = 4 *Sop /3*(5*np.sin(x12)**2 + 2*np.sin(x13)**2+ 5*np.sin(x23)**2 + 4*np.sin(x13)*np.sin(x23)*np.cos(x12)*np.cos(x13-x23)-\
        2*np.sin(x12)*np.sin(x23)*np.cos(x13)*np.cos(x12-x23) + 4*np.sin(x12)*np.sin(x13)*np.cos(x23)*np.cos(x12-x13))+\
        16*Spm/3*(6- np.cos(x12)**2*np.cos(x13)**2- 4*np.cos(x12)**2*np.cos(x23)**2- np.cos(x13)**2*np.cos(x23)**2+\
            4*np.sin(x12)*np.sin(x13)*np.cos(x23)*np.cos(x12-x13)- 2*np.sin(x12)*np.sin(x23)*np.cos(x13)*np.cos(x12-x23)+\
                4*np.sin(x13)*np.sin(x23)*np.cos(x12)*np.cos(x13-x23))



    if isinstance(frequencydata,FrequencyArray):
        return FrequencyArray(Se,kmin=frequencydata.kmin,df=frequencydata.df)
    else:
        return Se

def noisepsd_unequal_T(frequencydata,L12,L23,L31,detector):
    """
    Compute and return analytic PSD of noise for TDI T
     @param frequencydata  either numpy array or FrequencyArray of freqs.
    """
    f = frequencydata.f if isinstance(frequencydata,FrequencyArray) else frequencydata
    x21=x12 = 2.0 * math.pi * L12/LC.clight * f
    x32=x23 = 2.0 * math.pi * L23/LC.clight * f
    x13=x31 = 2.0 * math.pi * L31/LC.clight * f

    if detector == 'Taiji':
        Spm, Sop = taijinoises(f)
    elif detector == 'LISA':
        Spm, Sop = lisanoises(f, model='SciRDv1')
    
    if(L12==L23):
        x=2.0 * math.pi * L12/LC.clight * f
        St = 16.0 * Sop * (1.0 - np.cos(x)) * np.sin(x)**2 + 128.0 * Spm * np.sin(x)**2 * np.sin(0.5*x)**4
    else:
        St = 16*Sop/3*(np.sin(x12)**2 + np.sin(x13)**2+ np.sin(x23)**2 - np.sin(x12)*np.sin(x13)*np.cos(x23)*np.cos(x12-x13)-\
                      np.sin(x12)*np.sin(x23)*np.cos(x13)*np.cos(x12-x23)- np.sin(x13)*np.sin(x23)*np.cos(x12)*np.cos(x13-x23))+\
            32*Spm/3*(3- np.cos(x12)**2*np.cos(x13)**2 - np.cos(x12)**2*np.cos(x23)**2 - np.cos(x13)**2*np.cos(x23)**2-\
                     2*np.sin(x12)*np.sin(x13)*np.cos(x23)*np.cos(x12-x13)- 2*np.sin(x12)*np.sin(x23)*np.cos(x13)*np.cos(x12-x23)-\
                     2*np.sin(x13)*np.sin(x23)*np.cos(x12)*np.cos(x13-x23)) 

    if isinstance(frequencydata,FrequencyArray):
        return FrequencyArray(St,kmin=frequencydata.kmin,df=frequencydata.df)
    else:
        return St


def taijinoises(f,unit='relativeFrequency'):
    """
    Return the analytic approximation of the two components of LISA noise,
    i.e. the acceleration and the
    @param f is the frequency array
    @param unit is the unit of the output: 'relativeFrequency' or 'displacement'
    """

    frq = f
    ### Acceleration noise
    ## In acceleration
    Sa_a_taiji = (3.e-15)**2
    Sa_a = Sa_a_taiji *(1.0 +(0.4e-3/frq)**2)*(1.0+(frq/8e-3)**4)
    ## In displacement
    Sa_d = Sa_a*(2.*np.pi*frq)**(-4.)
    ## In relative frequency unit
    Sa_nu = Sa_d*(2.0*np.pi*frq/LC.clight)**2
    Spm =  Sa_nu

    ### Optical Metrology System
    ## In displacement
    Soms_d_taiji =(8.e-12)**2
    Soms_d = Soms_d_taiji* (1. + (2.e-3/f)**4)
    ## In relative frequency unit
    Soms_nu = Soms_d*(2.0*np.pi*frq/LC.clight)**2
    Sop =  Soms_nu

    if unit=='displacement':
        return Sa_d, Soms_d
    elif unit=='relativeFrequency':
        return Spm, Sop
    else:
        raise NotImplementedError(unit)



