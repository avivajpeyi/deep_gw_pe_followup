#!/bin/bash
#SBATCH --job-name=0_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=build_cache.log

module --force purge && module load git/2.18.0 git-lfs/2.4.0 gcc/9.2.0 openmpi/4.0.2 numpy/1.19.2-python-3.8.5 mpi4py/3.0.3-python-3.8.5 && module unload zlib && source /fred/oz980/avajpeyi/envs/sstar_venv/bin/activate

export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export MPI_PER_NODE=1

deep_followup_setup ptChia.ini
deep_followup_setup ptMateu.ini
deep_followup_setup ptNitz.ini