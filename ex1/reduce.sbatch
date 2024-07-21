#!/bin/bash

#SBATCH --job-name=aivkovic_reduce_benchmark
#SBATCH --nodes=2
#SBATCH --time=90
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --exclusive
#SBATCH --output=reduce_benchmark.out

# Some of the algorithms available:
# 0:    ignore (default),
# 1:    linear,
# 2:    chain,
# 3:    pipeline,
# 4:    binary
module load openMPI/4.1.5/gnu

OUTPUT_DIR="./output/reduce"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

for np in $(seq 2 2 48)
do
    for alg in 0 1 2 4 # default, linear, chain, binary
    do
        if [ -f $OUTPUT_DIR/np$np-a$alg.csv ]; then
            echo "Output file $OUTPUT_DIR/np$np-a$alg.csv already exists. Skipping..."
            continue
        fi
        mpirun -np $np --map-by core --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_reduce_algorithm $alg ./osu/osu_reduce --full --iterations 10000 --warmup 1000 --message-size 1:1048576 -f csv >> $OUTPUT_DIR/np$np-a$alg.csv
    done
done
