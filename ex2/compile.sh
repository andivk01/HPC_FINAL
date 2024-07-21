module load openMPI/4.1.5/gnu
mpic++ -o compute compute.cpp -Wall -Wextra -O3 -march=native -fopenmp -lm