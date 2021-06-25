#!/bin/bash -l
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --job-name sim
#SBATCH --partition=p.test


echo
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo

#srun -N $SLURM_NNODES -n $SLURM_NPROCS -c 1 --cpu_bind=cores ~/gadget4/Gadget4 param.txt
mpiexec -np $SLURM_NPROCS ~/gadget4/Gadget4 param.txt
