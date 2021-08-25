import os
import numpy as np
from scipy.interpolate import interp1d
from IC_functions import power_to_gadget

def run(argv):
  
  # Let's just fix these.
  H0 = 0.1 # km/s/kpc
  OmegaM = 0.3089
  G = 4.3022682e-6 # (km/s)^2 kpc/Msun
  h = 0.6774
  
  if len(argv) < 4:
    print('python script.py <box size (kpc/h)> <grid size> <scale factor> ')
    return 1
  
  BoxSize = float(argv[1])
  GridSize = int(argv[2])
  ScaleFactor = float(argv[3])
  
  power_z = '42.2876'
  k, _, _, P = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/power_%s.txt'%power_z).T
  
  k *= 1e-3 # Mpc^-1 -> kpc^-1
  k /= h # kpc^-1 -> h/kpc
  P *= (2*np.pi**2/k**3) # dimensionless -> dimensionful, unit (kpc/h)^3
  P *= ScaleFactor**2*(1+float(power_z))**2 # power spectrum ~ a^2 during MD
  
  pkfun = interp1d(k,P)
  
  power_to_gadget('init.hdf5',pkfun,BoxSize,GridSize,ScaleFactor,H0,OmegaM,G)
  
  # these go in the parameter file
  print('Omega0                   %lg'%OmegaM)
  print('OmegaLambda              %lg'%(1-OmegaM))
  print('HubbleParam              %lg'%h) # h
  print('BoxSize                  %lg'%BoxSize)
  print('TimeBegin                %lg'%ScaleFactor)
  print('UnitLength_in_cm         3.085678e21') # 1 kpc/h
  print('UnitMass_in_g            1.989e33') # 1 Msun/h
  print('UnitVelocity_in_cm_per_s 1e5') # 1 km/s
  print('SofteningComovingClass1  %lg     ; or so'%(BoxSize/GridSize * 0.03))
  
if __name__ == '__main__':
  from sys import argv
  run(argv)
