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
import matplot.pyplot as plt
from signal.gensignal import gen_signal_fre
import Cosmology
```
The we"ll set up the parameters of GW
```
#assume a signal
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
```
