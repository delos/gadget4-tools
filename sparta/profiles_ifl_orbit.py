import numpy as np
import h5py
from sparta import sparta
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumtrapz
from numba import njit

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

@njit
def tdyn(a,OmegaM,h):
  # 2 R / sqrt(GM/R) = 2/sqrt(GM/R^3) = 2/sqrt(4pi/3 G rho))
  Grho_crit = 0.00124844578 * h**2
  Grho_m = Grho_crit * OmegaM * a**-3
  return 2./np.sqrt(4*np.pi/3 * 200. * Grho_m)

@njit
def dtda(a,OmegaM,h):
  OmegaL = 1. - OmegaM
  Grho_crit = 0.00124844578 * h**2
  H = np.sqrt(8*np.pi/3 * Grho_crit * (OmegaL + OmegaM * a**-3))
  return 1./(a*H)

@njit
def dNdynda(a,OmegaM,h):
  return dtda(a,OmegaM,h) / tdyn(a,OmegaM,h)

def run(argv):
  filename = argv[1]
  
  try: rscale = float(argv[5])
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

  try: Ndyn_next, Ndyn = [float(x) for x in argv[4].split(',')]
  except: Ndyn_next, Ndyn = -1.,0.

  # 
  Narr = 500
  a_arr = np.geomspace(scales[0],scales[-1],Narr)
  t_arr = cumtrapz(dtda(a_arr,OmegaM,h),x=a_arr,initial=0)
  t_arr += quad(dtda,0,a_arr[0],args=(OmegaM,h))[0]
  Ndyn_arr = cumtrapz(dNdynda(a_arr,OmegaM,h),x=a_arr,initial=0)
  Ndyn_arr -= Ndyn_arr[-1]

  t_Ndyn_interp = interp1d(Ndyn_arr,t_arr)
  a_Ndyn_interp = interp1d(Ndyn_arr,a_arr)

  ax = plt.figure().gca()
  ax.set_xlabel(r'$r$ (kpc)')
  if rscale == 0:
    ax.set_ylabel(r'$\rho$ ($\mathrm{M}_\odot$ kpc$^{-3}$)')
  elif rscale.is_integer():
    ax.set_ylabel(r'$\rho r^{%.0f}$ ($\mathrm{M}_\odot$ kpc$^{%.0f}$)'%(rscale,rscale-3))
  else:
    ax.set_ylabel(r'$\rho r^{%.1f}$ ($\mathrm{M}_\odot$ kpc$^{%.1f}$)'%(rscale,rscale-3))

  for halo_id in halos:
    d = sparta.load(filename, halo_ids = halo_id, tracers = None, results = ['sbk', 'tjy', 'ifl'], res_match = ['sbk', 'tjy', 'ifl'], res_pad_unmatched = True, analyses = ['rsp'])
    idx = d['tcr_ptl']['res_ifl']['t_infall'] <= t_Ndyn_interp(Ndyn)
    idx = idx & ( d['tcr_ptl']['res_ifl']['t_infall'] > t_Ndyn_interp(Ndyn_next) )

    for s in snaps:
      r = d['tcr_ptl']['res_tjy']['r'][:,s]/h
      
      idx_ = idx & (r>0)
      npar_ = np.sum(idx_)
      print('a=%.2f, (%.0f,%.0f): %d particles'%(scales[s],Ndyn_next,Ndyn,npar_))

      rad_bins,dens = profile(r[idx_],1)
      rad = .5*(rad_bins[1:]+rad_bins[:-1])
      dens /= npar_

      ax.loglog(rad,dens*rad**rscale,lw=2)

    for i in range(10):
      pidx = np.random.choice(np.where(idx)[0])
      print('particle %d'%pidx)
      r = d['tcr_ptl']['res_tjy']['r'][pidx]/h
      vr = d['tcr_ptl']['res_tjy']['vr'][pidx]/h
      dtdV = np.abs(1./vr) / (4./3*np.pi*r**2)

      odens = 2*dtdV/tdyn_list

      orb, = ax.loglog(r[r>0],odens[r>0]*r[r>0]**rscale,lw=.5)
      for s in snaps:
        ax.loglog(r[s],odens[s]*r[s]**rscale,marker='o',color=orb.get_color(),ms=3)

    ax.text(.02,.97,r'$a=%.1f$'%scales[s] + '\n' + r'$%.1fa_\mathrm{ifl}<%.1f$'%(a_Ndyn_interp(Ndyn_next),a_Ndyn_interp(Ndyn)),transform=ax.transAxes,ha='left',va='top')

  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv) 
