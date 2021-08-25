import numpy as np
from scipy.interpolate import interp1d
from IC_functions import particles_to_gadget_massrange

def run(argv):
  
  # Let's just fix these.
  H0 = 0.1 # km/s/kpc
  OmegaM = 0.3089
  G = 4.3022682e-6 # (km/s)^2 kpc/Msun
  h = 0.6774
  rho = 3*H0**2/(8*np.pi*G) * OmegaM # comoving matter density
  
  if len(argv) < 3:
    print('python script.py <particle count> <lognormal mass center (in Msun/h),spread (e-folds)>')
    return 1
  
  NP = int(argv[1])
  
  mass_info = argv[2].split(',')
  mass_center = float(mass_info[0])
  mass_spread = float(mass_info[1])
  
  # lognormal masses
  mass = np.exp(np.random.normal(np.log(mass_center),mass_spread,NP))
  
  # set box size to fit the mass content
  mass_mean = np.mean(mass)
  BoxSize = (np.sum(mass)/rho)**(1./3)
  
  # uniformly distribute positions
  pos = np.random.uniform(0,BoxSize,(3,NP))
  
  # zero velocities
  # if they were nonzero, we'd have to divide by sqrt(a)
  vel = np.zeros((3,NP))
  
  # write to snapshot
  particles_to_gadget_massrange('init.hdf5',pos,vel,mass)
  
  # these go in the parameter file
  print('Omega0                   %lg'%OmegaM)
  print('OmegaLambda              %lg'%(1-OmegaM))
  print('HubbleParam              %lg'%h) # h
  print('BoxSize                  %lg'%BoxSize)
  print('UnitLength_in_cm         3.085678e21') # 1 kpc/h
  print('UnitMass_in_g            1.989e33') # 1 Msun/h
  print('UnitVelocity_in_cm_per_s 1e5') # 1 km/s
  print('SofteningComovingClass1  %lg     ; or so'%(BoxSize/NP**(1./3) * 0.03))
  
if __name__ == '__main__':
  from sys import argv
  run(argv)
