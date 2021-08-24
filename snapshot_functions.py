import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import h5py
from pathlib import Path

fileprefix_snapshot = 'snapdir_%03d/snapshot_%03d'
fileprefix_subhalo = 'groups_%03d/fof_subhalo_tab_%03d'
fileprefix_subhalo_desc = 'groups_%03d/subhalo_desc_%03d'
fileprefix_subhalo_prog = 'groups_%03d/subhalo_prog_%03d'

def gadget_to_particles(fileprefix, opts={'pos':True,'vel':True,'ID':False,'mass':True}):

  '''
  
  Read particles from GADGET HDF5 snapshot.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

    opts: which fields to read and return
    
  Returns:
    
    pos: position array, shape (3,NP), comoving
    
    vel: velocity array, shape (3,NP), peculiar

    ID: ID array, shape (NP,)
    
    mass: mass array, shape (NP,)
    
    header: a dict with header info, use list(header) to see the fields
  
  '''

  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  fileinst = 0
  pinst = 0
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      MassTable = header['MassTable']
      ScaleFactor = 1./(1+header['Redshift'])
      NP = header['NumPart_ThisFile']
      NPtot = header['NumPart_Total']
      numfiles = header['NumFilesPerSnapshot']

      if fileinst == 0:
        # allocate full-sized memory blocks in advance, for efficiency
        if opts.get('pos'): pos = np.zeros((3,np.sum(NPtot)),dtype=np.float32)
        if opts.get('vel'): vel = np.zeros((3,np.sum(NPtot)),dtype=np.float32)
        if opts.get('mass'): mass = np.zeros(np.sum(NPtot),dtype=np.float32)
        if opts.get('ID'): ID = np.zeros(np.sum(NPtot),dtype=np.uint32)
 
      for typ in range(len(NPtot)):
        NPtyp = int(NP[typ])
        if NPtyp == 0:
          continue

        if opts.get('pos'): pos[:,pinst:pinst+NPtyp] = np.array(f['PartType%d/Coordinates'%typ]).T
        if opts.get('vel'): vel[:,pinst:pinst+NPtyp] = np.array(f['PartType%d/Velocities'%typ]).T * np.sqrt(ScaleFactor)

        if opts.get('mass'):
          if MassTable[typ] == 0.:
            mass[pinst:pinst+NPtyp] = np.array(f['PartType%d/Masses'%typ])
          else:
            mass[pinst:pinst+NPtyp] = np.full(NPtyp,MassTable[typ])

        if opts.get('ID'): ID[pinst:pinst+NPtyp] = np.array(f['PartType%d/ParticleIDs'%typ])

        pinst += NPtyp

    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [pos]
  if opts.get('vel'): ret += [vel]
  if opts.get('mass'): ret += [mass]
  if opts.get('ID'): ret += [ID]
  ret += [header]

  return tuple(ret)

def fof_to_halos(fileprefix,opts={'pos':True,'vel':True,'mass':True}):

  '''
  
  Read halos from GADGET HDF5 FOF file.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_tab_000, not fof_tab_000.0.hdf5)

    opts: which fields to read and return
    
  Returns:
    
    pos: position array, shape (3,NH), comoving
    
    vel: velocity array, shape (3,NH), peculiar
    
    mass: mass array, shape (NH,)

    header: a dict with header info, use list(header) to see the fields
    
  '''

  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  fileinst = 0
  if opts.get('pos'): pos = []
  if opts.get('vel'): vel = []
  if opts.get('mass'): mass = []
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      ScaleFactor = 1./(1+header['Redshift'])
      numfiles = header['NumFiles']

      if header['Ngroups_Total'] == 0:
        if opts.get('pos'): pos = [[]]
        if opts.get('vel'): vel = [[]]
        if opts.get('mass'): mass = [[]]
        break
 
      if header['Ngroups_ThisFile'] > 0:
        if opts.get('pos'): pos += [np.array(f['Group/GroupPos']).T]
        if opts.get('vel'): vel += [np.array(f['Group/GroupVel']).T * np.sqrt(ScaleFactor)]
        if opts.get('mass'): mass += [np.array(f['Group/GroupMass'])]

    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [np.concatenate(pos,axis=1)]
  if opts.get('vel'): ret += [np.concatenate(vel,axis=1)]
  if opts.get('mass'): ret += [np.concatenate(mass)]
  ret += [header]

  return tuple(ret)

def cic_bin(x,BoxSize,GridSize,weights=1,density=True):
  
  '''
  
  Bin particles into a density field using cloud-in-cell method
  
  Parameters:
    
    x: 3D positions, shape (3,NP) where NP is the number of particles
    
    BoxSize: size of periodic region
    
    GridSize: resolution of output density field, per dimension
    
    weights: weight (e.g. mass) to assign to each particle, either a number or
    an array of length NP
    
    density: If False, output the total mass within each cell. If True, output
    mass/volume.
    
  Returns:
    
    field of shape (GridSize,GridSize,GridSize)
    
    bin edges
  
  '''
  
  NP = x.shape[1]
  
  N = GridSize
  dx = BoxSize / GridSize
  bins = dx * np.arange(N+1)
  
  # idea:
  # i and i1 are indices of the two adjacent cells (in each dimension)
  # f is the fraction from i to i1 where particle lies
  # shapes are (3,NP)
  
  f = x / dx
  
  f[f < 0.5] += N
  f[f >= N+0.5] -= N
  
  i = (f-0.5).astype(np.int32)
  f -= i + 0.5
  
  i1 = i+1
  
  i[i<0] += N
  i[i>=N] -= N
  i1[i1<0] += N
  i1[i1>=N] -= N
  
  # now appropriately add each particle into the 8 adjacent cells
  
  hist = np.zeros((N,N,N))
  np.add.at(hist,(i[0],i[1],i[2]),(1-f[0])*(1-f[1])*(1-f[2])*weights)
  np.add.at(hist,(i1[0],i[1],i[2]),f[0]*(1-f[1])*(1-f[2])*weights)
  np.add.at(hist,(i[0],i1[1],i[2]),(1-f[0])*f[1]*(1-f[2])*weights)
  np.add.at(hist,(i[0],i[1],i1[2]),(1-f[0])*(1-f[1])*f[2]*weights)
  np.add.at(hist,(i1[0],i1[1],i[2]),f[0]*f[1]*(1-f[2])*weights)
  np.add.at(hist,(i[0],i1[1],i1[2]),(1-f[0])*f[1]*f[2]*weights)
  np.add.at(hist,(i1[0],i[1],i1[2]),f[0]*(1-f[1])*f[2]*weights)
  np.add.at(hist,(i1[0],i1[1],i1[2]),f[0]*f[1]*f[2]*weights)
  
  if density:
    hist /= dx**3
    
  return hist,bins

def power_spectrum(delta,BoxSize,bins=None):
  
  '''
  
  Find spherically averaged power spectrum of density field
  
  Parameters:
    
    delta: input density field
    
    BoxSize: width of periodic box
    
    bins: desired k bin edges
    
  Returns:
    
    k: array of wavenumbers
    
    P(k): array comprising the power spectrum as a function of k
  
  '''
  
  GridSize = delta.shape[0]
  dk = 2*np.pi/BoxSize
  
  # radial bins for k
  if bins is None:
    # corner of cube is at distance np.sqrt(3)/2*length from center
    bins = np.arange(1,int((GridSize+1) * np.sqrt(3)/2)) * dk
  
  # get wavenumbers associated with k-space grid
  k = ((np.indices(delta.shape)+GridSize//2)%GridSize-GridSize//2) * dk
  k_mag = np.sqrt(np.sum(k**2,axis=0))
  
  # Fourier transform and get power spectrum
  pk = np.abs(fftn(delta,overwrite_x=True))**2*BoxSize**3/GridSize**6
  
  hist_pk,_ = np.histogram(k_mag,bins=bins,weights=pk)
  hist_ct,_ = np.histogram(k_mag,bins=bins)
  hist_k,_ = np.histogram(k_mag,bins=bins,weights=k_mag)
  
  return hist_k/hist_ct, hist_pk/hist_ct

def density_profile(pos,mass,bins=None,BoxSize=None):
  
  '''
  
  Spherically averaged density profile centered at position (0,0,0)
  
  Parameters:
    
    pos: 3D positions relative to center, shape (3,NP) where NP is the number of particles

    mass: masses of particles, shape (NP)

    bins: radial bin edges
    
    BoxSize: size of periodic region (None if not periodic)
    
  Returns:
    
    radius, density

  
  '''
  
  NP = pos.shape[1]

  # shift periodic box
  if BoxSize is not None:
    pos[pos >= 0.5*BoxSize] -= BoxSize
    pos[pos < -0.5*BoxSize] -= BoxSize

  # radii
  r = np.sqrt(np.sum(pos**2,axis=0))

  # radial bins
  if bins is None:
    rmin = np.sort(r)[100]/10
    rmax = np.max(r)
    bins = np.geomspace(rmin,rmax,50)
  bin_volume = 4./3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
  
  hist_mass,_ = np.histogram(r,bins=bins,weights=mass)

  return 0.5*(bins[1:]+bins[:-1]), hist_mass / bin_volume

def subhalo_tracing_data(snapshot_number,subhalo_number):

  '''
  
  Get a subhalo's mass, position, velocity, progenitor, descendant,
  and other tracking information.
  
  Parameters:
    
    snapshot_number

    subhalo_number
    
  Returns:
    
    prog: best-scoring progenitor

    desc: best-scoring descendant

    pos: subhalo (comoving) position, shape (3,)

    vel: subhalo (peculiar) velocity, shape (3,)

    mass: subhalo mass
 
    ID: dict with group ID, subhalo ID, and most bound particle ID
  
    header: a dict with header info, use list(header) to see the fields
    
  '''

  prefix_sub = fileprefix_subhalo%(snapshot_number,snapshot_number)
  prefix_desc = fileprefix_subhalo_desc%(snapshot_number,snapshot_number)
  prefix_prog = fileprefix_subhalo_prog%(snapshot_number,snapshot_number)

  filepath = [
    Path(prefix_sub + '.hdf5'),
    Path(prefix_sub + '.0.hdf5'),
    ]

  if filepath[0].is_file():
    filebase_sub = prefix_sub + '.hdf5'
    filebase_desc = prefix_desc + '.hdf5'
    filebase_prog = prefix_prog + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase_sub = prefix_sub + '.%d.hdf5'
    filebase_desc = prefix_desc + '.%d.hdf5'
    filebase_prog = prefix_prog + '.%d.hdf5'
    numfiles = 2
  
  prog, desc, pos, vel, mass, ID, header = -1, -1, np.zeros(3), np.zeros(3), 0., {}, {}

  fileinst = 0
  hinst = 0
  ginst = 0
  while fileinst < numfiles:

    if numfiles == 1:
      filename_sub = filebase_sub
      filename_desc = filebase_desc
      filename_prog = filebase_prog
    else:
      filename_sub = filebase_sub%fileinst
      filename_desc = filebase_desc%fileinst
      filename_prog = filebase_prog%fileinst

    with h5py.File(filename_sub, 'r') as f:
      print('reading %s'%filename_sub)

      header = dict(f['Header'].attrs)

      ScaleFactor = 1./(1+header['Redshift'])
      numfiles = header['NumFiles']

      if hinst + header['Nsubhalos_ThisFile'] > subhalo_number:
        index = subhalo_number - hinst

        pos = np.array(f['Subhalo/SubhaloPos'])[index]
        vel = np.array(f['Subhalo/SubhaloVel'])[index] * np.sqrt(ScaleFactor)
        ID = {'group':np.array(f['Subhalo/SubhaloGroupNr'])[index],
          'subhalo':subhalo_number,
          'particle':np.array(f['Subhalo/SubhaloIDMostbound'])[index]}

        if ID['group']>=ginst:
          mass = {'group':np.array(f['Group/GroupMass'])[ID['group']-ginst],
            'subhalo':np.array(f['Subhalo/SubhaloMass'])[index]}
        else:
          ginst2 = ginst
          fileinst2 = fileinst
          while ID['group'] < ginst2:
            fileinst2 -= 1
            filename_sub2 = filebase_sub%fileinst2
            with h5py.File(filename_sub2, 'r') as f2:
              print('reading %s'%filename_sub2)
              header2 = dict(f2['Header'].attrs)
              ginst2 -= int(header2['Ngroups_ThisFile'])
              if ID['group'] >= ginst2:
                mass = {'group':np.array(f2['Group/GroupMass'])[ID['group']-ginst2],
                  'subhalo':np.array(f['Subhalo/SubhaloMass'])[index]}

        try:
          with h5py.File(filename_desc, 'r') as fd:
            print('reading %s'%filename_desc)
            if np.array(fd['Subhalo/SubhaloNr'])[index] != subhalo_number:
              raise Exception('halo number mismatch, %d != %d'%(np.array(fd['Subhalo/SubhaloNr'])[index],subhalo_number))
            desc = np.array(fd['Subhalo/DescSubhaloNr'])[index]
        except Exception as e:
          print(str(e))
          desc = -1

        try:
          with h5py.File(filename_prog, 'r') as fp:
            print('reading %s'%filename_prog)
            if np.array(fp['Subhalo/SubhaloNr'])[index] != subhalo_number:
              raise Exception('halo number mismatch, %d != %d'%(np.array(fp['Subhalo/SubhaloNr'])[index],subhalo_number))
            prog = np.array(fp['Subhalo/ProgSubhaloNr'])[index]
        except Exception as e:
          print(str(e))
          prog = -1

        break

      hinst += int(header['Nsubhalos_ThisFile'])
      ginst += int(header['Ngroups_ThisFile'])

    fileinst += 1
  else:
    print('Warning: halo %d not found'%subhalo_number)

  return prog, desc, pos, vel, mass, ID, header

def trace_subhalo(snapshot_number,subhalo_number):

  '''
  
  Trace a subhalo's position, mass, and other tracking information across snapshots.
  
  Parameters:
    
    snapshot_number

    subhalo_number
    
  Returns:
    
    num: snapshot number

    time: scale factor array, shape (NT,)

    pos: position array, shape (NT,3), comoving
    
    vel: velocity array, shape (NT,3), peculiar
    
    mass: mass array, shape (NT,)

    group: host group, shape (NT,)

    ID: list of dicts with group ID, subhalo ID, and most bound particle ID; shape (NT,)
    
  '''

  prog, desc, pos_, vel_, mass_, ID_, header_ = subhalo_tracing_data(snapshot_number,subhalo_number)

  print('halo: %d in snapshot %d'%(subhalo_number,snapshot_number))

  pos = [pos_]
  vel = [vel_]
  mass = [mass_]
  ID = [ID_]
  time = [header_['Time']]
  num = [snapshot_number]

  shift = 0
  while prog >= 0:
    shift += 1
    print('progenitor: %d in snapshot %d'%(prog,snapshot_number-shift))

    prog, _, pos_, vel_, mass_, ID_, header_ = subhalo_tracing_data(snapshot_number-shift,prog)

    pos += [pos_]
    vel += [vel_]
    mass += [mass_]
    ID += [ID_]
    time += [header_['Time']]
    num += [snapshot_number-shift]

  pos = pos[::-1]
  vel = vel[::-1]
  mass = mass[::-1]
  ID = ID[::-1]
  time = time[::-1]
  num = num[::-1]

  shift = 0
  while desc >= 0:
    shift += 1
    print('descendant: %d in snapshot %d'%(desc,snapshot_number+shift))

    _, desc, pos_, vel_, mass_, ID_, header_ = subhalo_tracing_data(snapshot_number+shift,desc)

    pos += [pos_]
    vel += [vel_]
    mass += [mass_]
    ID += [ID_]
    time += [header_['Time']]
    num += [snapshot_number+shift]

  return np.array(num), np.array(time), np.array(pos), np.array(vel), mass, ID

def subhalo_group_data(fileprefix,opts={'mass':True,'len':False,'pos':False},parttype=None):

  '''
  
  Read halos from GADGET HDF5 FOF+subhalo file and return data relevant to group membership.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_subhalo_tab_000, not fof_subhalo_tab_000.0.hdf5)

    opts: which fields to read and return

    parttype: if not None, consider only particles of the given type for certain outputs

  Returns:
    
    group: host group number

    rank: rank of subhalo within host group

    parentrank: rank of parent subhalo within host group

    mass: subhalo mass

    groupmass: mass of host group

    length: subhalo particle count

    grouplength: host group particle count

    pos: subhalo position (NH,3)

    grouppos: host group position (NH,3)

    header: a dict with header info, use list(header) to see the fields
    
  '''

  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  fileinst = 0
  group = []
  rank = []
  parentrank = []
  if opts.get('mass'):
    mass = []
    _groupmass = []
  if opts.get('len'):
    length = []
    _grouplength = []
  if opts.get('pos'):
    pos = []
    _grouppos = []
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      ScaleFactor = 1./(1+header['Redshift'])
      numfiles = header['NumFiles']

      group += [np.array(f['Subhalo/SubhaloGroupNr'])]
      rank += [np.array(f['Subhalo/SubhaloRankInGr'])]
      parentrank += [np.array(f['Subhalo/SubhaloParentRank'])]
      if parttype is None:
        if opts.get('mass'):
          mass += [np.array(f['Subhalo/SubhaloMass'])]
          _groupmass += [np.array(f['Group/GroupMass'])]
        if opts.get('len'):
          length += [np.array(f['Subhalo/SubhaloLen'])]
          _grouplength += [np.array(f['Group/GroupLen'])]
      else:
        if opts.get('mass'):
          mass += [np.array(f['Subhalo/SubhaloMassType'][:,parttype])]
          _groupmass += [np.array(f['Group/GroupMassType'][:,parttype])]
        if opts.get('len'):
          length += [np.array(f['Subhalo/SubhaloLenType'][:,parttype])]
          _grouplength += [np.array(f['Group/GroupLenType'][:,parttype])]
      if opts.get('pos'):
        pos += [np.array(f['Subhalo/SubhaloPos'])]
        _grouppos += [np.array(f['Group/GroupPos'])]

    fileinst += 1

  group = np.concatenate(group)

  ret = [group, np.concatenate(rank), np.concatenate(parentrank)]

  if opts.get('mass'):
    groupmass = np.concatenate(_groupmass)[group]
    ret += [np.concatenate(mass), groupmass]
  if opts.get('len'):
    grouplength = np.concatenate(_grouplength)[group]
    ret += [np.concatenate(length), grouplength]
  if opts.get('pos'):
    grouppos = np.concatenate(_grouppos,axis=0)[group]
    ret += [np.concatenate(pos), grouppos]

  return tuple(ret + [header])

def group_extent(fileprefix,group,size_definition='TopHat200'):

  '''
  
  Return position and extent of a group from a GADGET HDF5 FOF file (SUBFIND required).
  
  Parameters:
    
    fileprefix: input file prefix (e.g., fof_tab_000, not fof_tab_000.0.hdf5)

    group: group number

    size_definition: 'Crit200', 'Crit500', 'Mean200', or 'TopHat200' (default)
    
  Returns:
    
    pos: shape (3,), comoving
    
    radius
    
    header: a dict with header info, use list(header) to see the fields
    
  '''

  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  fileinst = 0
  hinst = 0
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      numfiles = header['NumFiles']

      if hinst + header['Ngroups_ThisFile'] > group:
        index = group - hinst

        pos = np.array(f['Group/GroupPos'])[index]
        radius = np.array(f['Group/Group_R_'+size_definition])[index]

        break

      hinst += int(header['Ngroups_ThisFile'])

    fileinst += 1
  else:
    print('Warning: halo %d not found'%group)
    pos = np.zeros(3)
    radius = 0

  return pos, radius, header

def count_halos():

  '''
  
  Trace a subhalo's position, mass, and other tracking information across snapshots.
  
  Parameters:
    
  Returns:
    
    num: snapshot number

    times: scale factor array, shape (NT,)

    groups: group count array, shape (NT,)

    subhalos: subhalo count array, shape (NT,)
    
    mass: total mass in halos, shape (NT,)

  '''

  times = []
  groups = []
  subhalos = []
  mass = []

  snapshot_number = 0
  while True:
    fileprefix = fileprefix_subhalo%(snapshot_number,snapshot_number)
    try:
      mass_, header = fof_to_halos(fileprefix,opts={'pos':False,'vel':False,'mass':True})
    except Exception as e:
      print(e)
      break

    times += [header['Time']]
    groups += [header['Ngroups_Total']]
    subhalos += [header['Nsubhalos_Total']]
    mass += [np.sum(mass_)]

    snapshot_number += 1

  return np.arange(snapshot_number), np.array(times), np.array(groups), np.array(subhalos), np.array(mass)

def list_snapshots():

  '''
  
  Get all snapshot names and headers.
  
  Parameters:
    
  Returns:
    
    names: list of snapshot prefixes

    headers: list of header dicts
    
  '''

  snapshot_number = 0

  names = []
  headers = []

  while True:
    prefix = fileprefix_snapshot%(snapshot_number,snapshot_number)

    filepath = [
      Path(prefix + '.hdf5'),
      Path(prefix + '.0.hdf5'),
      ]

    if filepath[0].is_file():
      filename = prefix + '.hdf5'
      numfiles = 1
    elif filepath[1].is_file():
      filename = prefix + '.0.hdf5'
      numfiles = 2
    else:
      break

    names += [prefix]

    with h5py.File(filename, 'r') as f:
      headers += [dict(f['Header'].attrs)]

    snapshot_number += 1

  return names, headers

def read_particles_filter(fileprefix, center=None, rotation=None, radius=None, halfwidth=None, ID_list=None, opts={'pos':True,'vel':True,'ID':False,'mass':True}):

  '''
  
  Read particles from GADGET HDF5 snapshot. Return only particles matching filter.
  
  Parameters:
    
    fileprefix: input file prefix (e.g., snapshot_000, not snapshot_000.0.hdf5)

    center: center particles about this position

    rotation: 3x3 rotation matrix to apply. Requires center.

    radius: filter particles within radius. Requires center.

    halfwidth: filter particles within box of width 2*halfwidth. Requires center.

    ID_list: only include particles whose IDs are listed

    opts: which fields to read and return
    
  Returns:
    
    pos: position array, shape (NP,3), comoving
    
    vel: velocity array, shape (NP,3), peculiar

    ID: ID array, shape (NP,)
    
    mass: mass array, shape (NP,)
    
    header: a dict with header info, use list(header) to see the fields
  
  '''
  filepath = [
    Path(fileprefix + '.hdf5'),
    Path(fileprefix + '.0.hdf5'),
    Path(fileprefix),
    ]

  if filepath[0].is_file():
    filebase = fileprefix + '.hdf5'
    numfiles = 1
  elif filepath[1].is_file():
    filebase = fileprefix + '.%d.hdf5'
    numfiles = 2
  elif filepath[2].is_file():
    # exact filename was passed - will cause error if >1 files, otherwise fine
    filebase = fileprefix
    numfiles = 1

  if center is not None:
    if len(np.shape(center)) < 2:
      center = np.array(center)[None,:]
    else:
      center = np.array(center)

  fileinst = 0
  if opts.get('pos'): pos = []
  if opts.get('vel'): vel = []
  if opts.get('mass'): mass = []
  if opts.get('ID'): ID = []
  while fileinst < numfiles:

    if numfiles == 1:
      filename = filebase
    else:
      filename = filebase%fileinst

    with h5py.File(filename, 'r') as f:
      print('reading %s'%filename)

      header = dict(f['Header'].attrs)

      MassTable = header['MassTable']
      ScaleFactor = 1./(1+header['Redshift'])
      NP = header['NumPart_ThisFile']
      NPtot = header['NumPart_Total']
      numfiles = header['NumFilesPerSnapshot']
      BoxSize = header['BoxSize']

      for typ in range(len(NPtot)):
        NPtyp = int(NP[typ])
        if NPtyp == 0:
          continue

        chunksize = 1048576
        Nleft = NPtyp
        i0 = 0

        while Nleft > 0:
          Nc = min(chunksize,Nleft)
          iread = slice(i0,i0+Nc)

          idx = np.full(Nc,True)

          if center is not None: # filter by position
            pos_ = np.array(f['PartType%d/Coordinates'%typ][iread])
            pos_ -= center
            if BoxSize > 0.:
              pos_[pos_ > .5*BoxSize] -= BoxSize
              pos_[pos_ < -.5*BoxSize] += BoxSize
            if rotation is not None: pos_ = (rotation@(pos_.T)).T
            if radius is not None:
              idx = idx & (np.sum(pos_**2,axis=1) < radius**2)
            if halfwidth is not None:
              idx = idx&(-halfwidth<=pos_[:,0])&(pos_[:,0]<=halfwidth)
              idx = idx&(-halfwidth<=pos_[:,1])&(pos_[:,1]<=halfwidth)
              idx = idx&(-halfwidth<=pos_[:,2])&(pos_[:,2]<=halfwidth)

          if ID_list is not None: # filter by ID
            ID_ = np.array(f['PartType%d/ParticleIDs'%typ][iread])
            idx = idx & np.isin(ID_,ID_list,assume_unique=True)

          if np.sum(idx) > 0:
            if opts.get('pos'):
              if center is None:
                pos += [np.array(f['PartType%d/Coordinates'%typ][iread])[idx]]
              else: pos += [pos_[idx]]
            if opts.get('vel'):
              vel += [np.array(f['PartType%d/Velocities'%typ][iread])[idx] * np.sqrt(ScaleFactor)]

            if opts.get('mass'):
              if MassTable[typ] == 0.:
                mass_ += np.array(f['PartType%d/Masses'%typ][iread])
              else:
                mass_ = np.full(Nc,MassTable[typ])
              mass += [mass_[idx]]
            if opts.get('ID'):
              if ID_list is None:
                ID += [np.array(f['PartType%d/ParticleIDs'%typ][iread])[idx]]
              else: ID += [ID_[idx]]

          Nleft -= Nc
          i0 += Nc

    fileinst += 1

  ret = []
  if opts.get('pos'): ret += [np.concatenate(pos,axis=0)]
  if opts.get('vel'): ret += [np.concatenate(vel,axis=0)]
  if opts.get('mass'): ret += [np.concatenate(mass)]
  if opts.get('ID'): ret += [np.concatenate(ID)]
  ret += [header]

  return tuple(ret)

