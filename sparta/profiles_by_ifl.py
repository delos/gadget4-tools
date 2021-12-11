import numpy as np
import h5py
from sparta import sparta
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumtrapz

import matplotlib.pyplot as plt

from sys import argv

def profile(r,mass,bins=None):
  # radial bins
  if bins is None:
    rmin = np.sort(r)[100]/10
    rmax = np.max(r)
    bins = np.geomspace(rmin,rmax,50)
  bin_volume = 4./3 * np.pi * (bins[1:]**3 - bins[:-1]**3)

  hist_mass,_ = np.histogram(r,bins=bins)

  return bins, mass*hist_mass / bin_volume

def tdyn(a,OmegaM,h):
  # 2 R / sqrt(GM/R) = 2/sqrt(GM/R^3) = 2/sqrt(4pi/3 G rho))
  Grho_crit = 0.00124844578 * h**2
  Grho_m = Grho_crit * OmegaM * a**-3
  return 2./np.sqrt(4*np.pi/3 * 200. * Grho_m)

def dtda(a,OmegaM,h):
  OmegaL = 1. - OmegaM
  Grho_crit = 0.00124844578 * h**2
  H = np.sqrt(8*np.pi/3 * Grho_crit * (OmegaL + OmegaM * a**-3))
  return 1./(a*H)

def dNdynda(a,OmegaM,h):
  return dtda(a,OmegaM,h) / tdyn(a,OmegaM,h)

def run(argv):
  filename = argv[1]
  
  try: rscale = int(argv[4])
  except: rscale = 0.

  halos_list = []
  with h5py.File(filename, 'r') as f:
    # cosmology
    OmegaL = f['simulation'].attrs['Omega_L']
    OmegaM = f['simulation'].attrs['Omega_m']
    h =  f['simulation'].attrs['h']

    mass = np.array(f['simulation'].attrs['particle_mass'])/h
    scales = np.array(f['simulation'].attrs['snap_a'])
    times = np.array(f['simulation'].attrs['snap_t'])
    tdyn_list = np.array(f['simulation'].attrs['snap_tdyn'])
    halo_n = np.array(f['tcr_ptl']['res_tjy']['halo_n'])
    idx_halos = np.where(halo_n>0)[0]
    for i in idx_halos:
      halo_id_ = f['halos']['id'][i]
      halos_list += [halo_id_[halo_id_>0][-1]]

  print('halos: ' + ','.join([str(x) for x in halos_list]))

  print('snapshots 0-%d'%(len(scales)-1))

  try: halos = [int(x) for x in argv[2].split(',')]
  except: halos = halos_list
  try: snaps = [int(x) for x in argv[3].split(',')]
  except: snaps = [len(scales)-1]
  if np.max(snaps) > len(scales)-1: return

  # 
  Narr = 500
  a_arr = np.geomspace(scales[0],scales[-1],Narr)
  t_arr = cumtrapz(dtda(a_arr,OmegaM,h),x=a_arr,initial=0)
  t_arr += quad(dtda,0,a_arr[0],args=(OmegaM,h))[0]
  Ndyn_arr = cumtrapz(dNdynda(a_arr,OmegaM,h),x=a_arr,initial=0)
  Ndyn_arr -= Ndyn_arr[-1]

  t_Ndyn_interp = interp1d(Ndyn_arr,t_arr)
  # t0 + avg[tdyn]_t0^t1 = t1

  ax = plt.figure().gca()
  ax.set_xlabel(r'$r$ (kpc)')
  ax.set_ylabel(r'$\rho$ ($\mathrm{M}_\odot$ kpc$^{-3}$)')

  for halo_id in halos:
    d = sparta.load(filename, halo_ids = halo_id, tracers = None, results = ['sbk', 'tjy', 'ifl'], res_match = ['sbk', 'tjy', 'ifl'], res_pad_unmatched = True, analyses = ['rsp'])

    for s in snaps:
      r = d['tcr_ptl']['res_tjy']['r'][:,s]/h
      idx = r>0
      print('a=%2f: %d particles'%(scales[s],np.sum(idx)))
      
      rad_bins,dens = profile(r[idx],mass)
      rad = .5*(rad_bins[1:]+rad_bins[:-1])

      ax.loglog(rad,dens*rad**rscale)

      Ndyn = 0.
      while True:
        Ndyn_next = Ndyn - 1.
        if Ndyn_next < Ndyn_arr[0]:
          break
        idx = d['tcr_ptl']['res_ifl']['t_infall'] <= t_Ndyn_interp(Ndyn)
        idx = idx & ( d['tcr_ptl']['res_ifl']['t_infall'] > t_Ndyn_interp(Ndyn_next) )
        idx = idx & (r>0)
        npar = np.sum(idx)
        print('(%.0f,%.0f): %d particles'%(Ndyn_next,Ndyn,npar))
        if npar == 0:
          continue
        _,dens = profile(r[idx],mass,bins=rad_bins)
        ax.loglog(rad,dens*rad**rscale,lw=.5)
        Ndyn = Ndyn_next

  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv) 
