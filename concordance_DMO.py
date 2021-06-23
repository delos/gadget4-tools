import numpy as np
from scipy.interpolate import interp1d
from IC_functions import power_to_gadget

def run(argv):
  
  # Let's just fix these.
  H0 = 0.06774 # km/s/kpc
  OmegaM = 0.3089
  G = 4.3022682e-6 # (km/s)^2 kpc/Msun
  
  if len(argv) < 4:
    print('python script.py <box size/kpc> <grid size> <scale factor> ')
    return 1
  
  BoxSize = float(argv[1])
  GridSize = int(argv[2])
  ScaleFactor = float(argv[3])
  
  power_scale = '42.2876'
  k, _, _, P = np.loadtxt('power_%s.txt'%power_scale).T
  
  k *= 1e-3 # Mpc^-1 -> kpc^-1
  P *= (2*np.pi**2/k**3) # dimensionless -> dimensionful, unit kpc^3
  P *= (ScaleFactor/float(power_scale))**2 # power spectrum ~ a^2 during MD
  
  pkfun = interp1d(k,P)
  
  power_to_gadget('init.hdf5',pkfun,BoxSize,GridSize,ScaleFactor,H0,OmegaM,G)
  
  # these go in the parameter file
  print('Omega0                   %lg'%OmegaM)
  print('OmegaLambda              %lg'%(1-OmegaM))
  print('HubbleParam              %lg'%(H0/.1)) # h
  print('BoxSize                  %lg'%BoxSize)
  print('UnitLength_in_cm         3.085678e24') # 1 kpc/h
  print('UnitMass_in_g            1.989e33') # 1 Msun/h
  print('UnitVelocity_in_cm_per_s 1e5') # 1 km/s
  
if __name__ == '__main__':
  from sys import argv
  run(argv)