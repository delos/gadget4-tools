import numpy as np
import h5py
from sparta import sparta

import matplotlib.pyplot as plt

from sys import argv

def profile(r,mass,bins=None,):
  # radial bins
  if bins is None:
    rmin = np.sort(r)[100]/10
    rmax = np.max(r)
    bins = np.geomspace(rmin,rmax,50)
  bin_volume = 4./3 * np.pi * (bins[1:]**3 - bins[:-1]**3)

  hist_mass,_ = np.histogram(r,bins=bins)

  return 0.5*(bins[1:]+bins[:-1]), mass*hist_mass / bin_volume

def run(argv):
  filename = argv[1]

  halos_list = []
  with h5py.File(filename, 'r') as f:
    mass = np.array(f['simulation'].attrs['particle_mass'])
    scales = np.array(f['simulation'].attrs['snap_a'])
    times = np.array(f['simulation'].attrs['snap_t'])
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

  ax = plt.figure().gca()
  ax.set_xlabel(r'$r$')
  ax.set_ylabel(r'$\rho$')

  for h in halos:
    d = sparta.load(filename, halo_ids = h, tracers = None, results = ['sbk', 'tjy', 'ifl'], res_match = ['sbk', 'tjy', 'ifl'], res_pad_unmatched = True, analyses = ['rsp'])

    for s in snaps:
      r = d['tcr_ptl']['res_tjy']['r'][:,s]
      print('%d particles'%len(r))
      
      rad,dens = profile(r[r>0],mass)

      ax.loglog(rad,dens)

  plt.show()

if __name__ == '__main__':
  from sys import argv
  run(argv) 
