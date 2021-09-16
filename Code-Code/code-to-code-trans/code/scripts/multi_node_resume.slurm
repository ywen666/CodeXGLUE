#!/bin/bash

# Sample Slurm job script
#   for TACC Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J GPTNEO_resume           # Job name
#SBATCH -o slurm_logs/gpt125M_2nodeacc4_p1.o%j            # Name of stdout output file
#SBATCH -e slurm_logs/gpt125M_2nodeacc4_p1.e%j            # Name of stdout output file
#SBATCH -N 2                            # Total # of nodes 
#SBATCH -n 8                           # Total # of mpi tasks
#SBATCH -t 48:00:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=XXX                 # Email address
#SBATCH --mail-type=end                 # Send email at begin and end of job
#SBATCH -p rtx                          # Queue
#SBATCH -A ASC21003                     # Allocation

# Note mpiexec calls may need to be changed depending on MPI type

HOSTFILE=/tmp/hostfile

scontrol show hostnames $SLURM_NODELIST > $HOSTFILE
cat $HOSTFILE

GPU_PER_NODE=4
NODES=$(wc -l < $HOSTFILE)
MASTER_NODE=$(head -n 1 $HOSTFILE)
PARTITION=1

mpiexec.hydra -f $HOSTFILE -np $NODES -ppn 1 \
  ./scripts/multi_node_resume.sh \
    --ngpus $GPU_PER_NODE --nnodes $NODES --master $MASTER_NODE --partition $PARTITION
