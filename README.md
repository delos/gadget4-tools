# gadget4-tools
Simple examples of IC creation and snapshot reading/interpretation for the [GADGET-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/) simulation code. These scripts are intended only as a starting point for further development.

Requires: numpy, scipy, matplotlib, h5py

## Helper functions

1. **IC_functions.py**: Helper functions related to preparing initial conditions for simulations.
2. **snapshot_functions.py**: Helper functions related to reading and analyzing simulation snapshots.

See code for further documentation.

## Example initial conditions

1. **concordance_DMO.py**: generate dark matter initial conditions from a standard power spectrum.
2. **poisson.py**: generate primordial black hole initial conditions. Particles are Poisson-distributed with zero velocity and a lognormal mass spectrum.

## Example snapshot analysis

1. **plot_field.py**: read a simulation snapshot and plot the cloud-in-cell binned density field.
2. **plot_power.py**: read a simulation snapshot and plot the density power spectrum.
3. **plot_profiles.py**: read a simulation snapshot and the associated FOF file and plot the density profiles of the largest few halos. 
