# import sys,os
# import numpy as np
# from astropy import units as u
# from astropy.coordinates import SkyCoord
# from pyFDresponse import *
# # Cartesian SSB components of spacecraft positions
# path =os.path.abspath(os.path.dirname(__file__))
# # print(path)
# p1 = np.fromfile(path+'/LISA_orbit_data/SCP1_1yr.dat',sep=' ')
# p2 = np.fromfile(path+'/LISA_orbit_data/SCP2_1yr.dat',sep=' ')
# p3 = np.fromfile(path+'/LISA_orbit_data/SCP3_1yr.dat',sep=' ')
# #print(p1)
# #print(len(p1))
# p1= p1.reshape(int(len(p1)/3),3)
# p2= p2.reshape(int(len(p2)/3),3)
# p3= p3.reshape(int(len(p3)/3),3)

# c1= SkyCoord(x=p1[:,0],y=p1[:,1],z=p1[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
# c2= SkyCoord(x=p2[:,0],y=p2[:,1],z=p2[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
# c3= SkyCoord(x=p3[:,0],y=p3[:,1],z=p3[:,2],unit=u.au,frame='icrs',representation_type='cartesian')
# #print(c1)
# #transfrom to Ecliptic system of coordinate
# p1_SSB= c1.heliocentrictrueecliptic
# p2_SSB= c2.heliocentrictrueecliptic
# p3_SSB= c3.heliocentrictrueecliptic
# p1_SSB.representation_type='cartesian'
# p2_SSB.representation_type='cartesian'
# p3_SSB.representation_type='cartesian'
# #interpolate 
# def funcp_SSB(t,p_SSB):
#     year=np.linspace(0,LC.YRSID_SI,366)
#     px_SSB= spline(year,p_SSB.x)
#     py_SSB= spline(year,p_SSB.y)
#     pz_SSB= spline(year,p_SSB.z)
#     return np.array([px_SSB(t),py_SSB(t),pz_SSB(t)]) 

# #Computing the arm length
# def funcL_SSB(p1_SSB,p2_SSB):
#     Ln = p2_SSB- p1_SSB
#     L = np.sqrt(Ln[0]**2+Ln[1]**2+Ln[2]**2)
#     return L

# def funcp0_SSB(t,p1_SSB=p1_SSB,p2_SSB=p2_SSB,p3_SSB=p3_SSB):
#     p1= funcp_SSB(t,p1_SSB)*R_SI
#     p2= funcp_SSB(t,p2_SSB)*R_SI
#     p3= funcp_SSB(t,p3_SSB)*R_SI
#     p0 = (p1+p2+p3)/3
#     return p0

# trajdict_numerical_orbits = {
#     'funcp0':funcp0_SSB,
#     'funcp': funcp_SSB,
#     'funcL': funcL_SSB,
#     'p1_SSB': p1_SSB,
#     'p2_SSB': p2_SSB,
#     'p3_SSB': p3_SSB,
#     }