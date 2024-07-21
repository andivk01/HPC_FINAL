#!/bin/bash

#SBATCH --job-name=strong_scalability
#SBATCH --nodes=4
#SBATCH --time=60
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --exclusive
#SBATCH --output=strong_scalability.out
module load openMPI/4.1.5/gnu

x_min=-2.0
x_max=2.0
y_min=-2.0
y_max=2.0
width=3840
height=3840
max_iterations=1000

OUTPUT_DIR="./output/"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

output_file="./output/MPI_strong.csv"
echo "np,time" > $output_file

# MPI Strong scalability
for np in $(seq 2 2 96)
do
    output=$(mpirun -np ${np} --map-by socket --bind-to socket -x OMP_NUM_THREADS=1 ./compute $x_min $x_max $y_min $y_max $width $height $max_iterations)
    echo "${np},${output}" >> $output_file
done


output_file="./output/OpenMP_strong.csv"
echo "nt,time" > $output_file

# OpenMP Strong scalability
for nt in $(seq 2 2 24)
do
    output=$(mpirun -np 2 --map-by socket --bind-to none -x OMP_NUM_THREADS=${nt} ./compute $x_min $x_max $y_min $y_max $width $height $max_iterations)
    echo "${nt},${output}" >> $output_file
done