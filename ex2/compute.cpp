#include <iostream>
#include <fstream>
#include <complex>
#include <mpi.h>
#include <omp.h>
#include <vector>

#define TAG_ASK_FOR_WORK 0
#define TAG_BEGIN_IDX 1
#define MAX_NUM_PXS_PER_WORKER 8192

struct Pixel {
    int x;
    int y;
};

struct ApproxConfig {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    int width;
    int height;
    int max_iterations;
};

void save_pgm(const std::vector<short int>& image, int width, int height, const std::string& filename, short int max_value) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "P2" << std::endl;
    file << width << " " << height << std::endl;
    file << max_value << std::endl;

    for (long unsigned int i = 0; i < image.size(); i++) {
        file << image[i] << " ";
        if ((i + 1) % width == 0) {
            file << std::endl;
        }
    }

    file.close();
}

std::complex<double> px_to_pt(Pixel& p, ApproxConfig& config) {
    double x = config.x_min + p.x * (config.x_max - config.x_min) / config.width;
    double y = config.y_min + p.y * (config.y_max - config.y_min) / config.height;
    return std::complex<double>(x, y);
}

int compute_point(std::complex<double> c, int max_iterations) {
    std::complex<double> z(0, 0);

    int iter = 0;
    while (iter < max_iterations && std::abs(z) < 2.0) {
        z = z * z + c;
        iter++;
    }
    if (iter == max_iterations) {
        return 0;
    }
    return iter;
}

bool all_workers_done(const std::vector<int>& worker_begin_idx, int num_pixels) {
    for (std::vector<int>::size_type i = 0; i < worker_begin_idx.size(); i++) {
        if (worker_begin_idx[i] < num_pixels) {
            return false;
        }
    }
    return true;
}

void recv_image_part(std::vector<short int>& image, int begin_idx, int computed_pxs, int worker_rank) {
    std::vector<short int> pixel_iterations(computed_pxs, 0);
    MPI_Recv(pixel_iterations.data(), computed_pxs, MPI_SHORT, worker_rank, TAG_ASK_FOR_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int idx = begin_idx;
    for (int i = 0; i < computed_pxs; i++) {
        image[idx] = pixel_iterations[i];
        idx++;
    }
}

void master_process(ApproxConfig& config, int num_workers) {
    int num_pixels = config.width * config.height;
    int begin_idx = 0;
    std::vector<short int> image(num_pixels, 0);
    std::vector<int> worker_begin_idx(num_workers, 0);
    while (!all_workers_done(worker_begin_idx, num_pixels)) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, TAG_ASK_FOR_WORK, MPI_COMM_WORLD, &status);
        int worker_rank = status.MPI_SOURCE;
        int computed_pxs;
        MPI_Get_count(&status, MPI_SHORT, &computed_pxs);
        if (computed_pxs != 0) { // something to receive
            recv_image_part(image, worker_begin_idx[worker_rank - 1], computed_pxs, worker_rank);
        } else { // at the beginning no data is received
            MPI_Recv(NULL, 0, MPI_SHORT, worker_rank, TAG_ASK_FOR_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Send(&begin_idx, 1, MPI_INT, worker_rank, TAG_BEGIN_IDX, MPI_COMM_WORLD);
        worker_begin_idx[worker_rank - 1] = begin_idx;
        begin_idx += MAX_NUM_PXS_PER_WORKER;
    }
    save_pgm(image, config.width, config.height, "output.pgm", config.max_iterations);
}

void worker_process(ApproxConfig& config) {
    int begin_idx;
    int num_pixels = config.width * config.height;
    MPI_Send(NULL, 0, MPI_SHORT, 0, TAG_ASK_FOR_WORK, MPI_COMM_WORLD); // no data at the beginning

    while (true) {
        MPI_Recv(&begin_idx, 1, MPI_INT, 0, TAG_BEGIN_IDX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int computed_pxs = num_pixels - begin_idx;
        if (computed_pxs <= 0) { // no more work
            break;
        }
        computed_pxs = std::min(computed_pxs, MAX_NUM_PXS_PER_WORKER);

        std::vector<short int> pixel_iterations(computed_pxs, 0);

        #pragma omp parallel for schedule(dynamic, 250)
        for (int i = 0; i < computed_pxs; i++) {
            int idx = i + begin_idx;
            Pixel p = {idx % config.width, idx / config.width};
            std::complex<double> pt = px_to_pt(p, config);
            pixel_iterations[i] = compute_point(pt, config.max_iterations);
        }
        MPI_Send(pixel_iterations.data(), computed_pxs, MPI_SHORT, 0, TAG_ASK_FOR_WORK, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <x_min> <x_max> <y_min> <y_max> <width> <height> <max_iterations>" << std::endl;
        return 1;
    }

    ApproxConfig config;
    config.x_min = std::stod(argv[1]);
    config.x_max = std::stod(argv[2]);
    config.y_min = std::stod(argv[3]);
    config.y_max = std::stod(argv[4]);
    config.width = std::stoi(argv[5]);
    config.height = std::stoi(argv[6]);
    config.max_iterations = std::stoi(argv[7]);
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    int num_workers = num_processes - 1;

    if (num_workers == 0) {
        std::cerr << "At least 2 processes are required" << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        double start_time = MPI_Wtime();
        master_process(config, num_workers);
        std::cout << MPI_Wtime() - start_time << std::endl;
    } else {
        worker_process(config);
    }
    MPI_Finalize();
    return 0;
}