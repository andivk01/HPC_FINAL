#!/bin/bash

#SBATCH --job-name=weak_scalability
#SBATCH --nodes=4
#SBATCH --time=60
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --exclusive
#SBATCH --output=weak_scalability.out
module load openMPI/4.1.5/gnu

x_min=-2.0
x_max=2.0
y_min=-2.0
y_max=2.0
width=3840
base_height=250
max_iterations=1000

OUTPUT_DIR="./output/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

output_file="./output/MPI_weak.csv"
echo "np,height,time" > $output_file

# MPI weak scalability
for np in $(seq 2 2 96)
do
    output=$(mpirun -np ${np} --map-by socket --bind-to socket -x OMP_NUM_THREADS=1 ./compute $x_min $x_max $y_min $y_max $width $((base_height * np)) $max_iterations)
    echo "${np},$((base_height * np)),${output}" >> $output_file
done


output_file="./output/OpenMP_weak.csv"
echo "nt,height,time" > $output_file

# OpenMP weak scalability
for nt in $(seq 2 2 24)
do
    output=$(mpirun -np 2 --map-by socket --bind-to socket -x OMP_NUM_THREADS=${nt} ./compute $x_min $x_max $y_min $y_max $width $((base_height * nt)) $max_iterations)
    echo "${nt},$((base_height * nt)),${output}" >> $output_file
done