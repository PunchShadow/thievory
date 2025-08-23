#include <numa.h>

#include <iostream>

#include "../src/bfs/bfs.cuh"
#include "../src/cc/cc.cuh"
#include "../src/pr/pr.cuh"
#include "../src/sssp/sssp.cuh"
#include "utils.hpp"

enum BYTES {
  _4BYTE = 4,
  _8BYTE = 8,
};

void usage(const char *program_name) {
  std::cout << "Usage: " << program_name << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --input : Specify the path to graph input " << std::endl;
  std::cout << "  --algo : Specify the algorithm {'bfs', 'cc', 'pr', 'sssp'} "
               "(default = 'bfs')"
            << std::endl;
  std::cout << "  --edgeSize : Specify the edge size {4, 8} (default = 4)"
            << std::endl;

  std::cout
      << "  --type : Specify PageRank type {'push', 'pull'} (default = 'push')"
      << std::endl;

  std::cout
      << "  --source : Specify the source vertex {0, 1, 2, ...} (default = 1)"
      << std::endl;
  std::cout
      << "  --runs : Specify the number of runs {0, 1, 2, ...} (default = 1) "
      << std::endl;

  std::cout << "  --gpus : Specify the number of NEIGHBOR GPUs {0, 1, 2, ...} "
               "(default = 0)"
            << std::endl;

  exit(0);
}

int main(int argc, char **argv) {
  cudaFree(0);

  cudaDeviceProp deviceProperties;
  memset(&deviceProperties, 0, sizeof(cudaDeviceProp));

  deviceProperties.major = 6;
  deviceProperties.minor = 0;

  int device;  // Selected device
  cudaChooseDevice(
      &device, &deviceProperties);  // Select the device that fits criteria best
  cudaSetDevice(device);
  cudaSetDevice(0);

  std::cout << "Selected device " << device << std::endl;

  // size_t totalMemory;
  // size_t availMemory;
  // cudaMemGetInfo(&availMemory, &totalMemory);
  //
  // printf("Free memory: %lu \n", availMemory);
  //
  // size_t newFreeMemory = 4ULL * 1024 * 1024 * 1024;

  // size_t allocSize = availMemory - newFreeMemory;

  // void *d_ptr;
  // cudaMalloc(&d_ptr, allocSize);

  // cudaMemGetInfo(&availMemory, &totalMemory);
  //
  // printf("New free memory: %lu \n", availMemory);
  //
  // cudaMemGetInfo(&availMemory, &totalMemory);

  // Default parameters
  std::string filePath;
  bool hasInput = false;
  std::string algorithm = "bfs";
  std::string type = "push";
  uint32 edgeSize = _4BYTE;
  uint32 srcVertex = 1;
  uint32 nRuns = 1;
  uint32 nNGPUs = 0;
  uint32 defaultDevice = 0;

  try {
    for (unsigned int i = 1; i < argc - 1; i = i + 2) {
      if (strcmp(argv[i], "--input") == 0) {
        filePath = std::string(argv[i + 1]);
        hasInput = true;
      } else if (strcmp(argv[i], "--algo") == 0)
        algorithm = std::string(argv[i + 1]);
      else if (strcmp(argv[i], "--edgeSize") == 0)
        edgeSize = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "--type") == 0)
        type = std::string(argv[i + 1]);
      else if (strcmp(argv[i], "--source") == 0)
        srcVertex = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "--runs") == 0)
        nRuns = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "--gpus") == 0)
        nNGPUs = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "--device") == 0)
        defaultDevice = atoi(argv[i + 1]);
    }
  } catch (...) {
    std::cerr << "An exception has occurred.\n";
    exit(0);
  }

  if (!hasInput) exit(0);

  std::string topoOutput = runNvidiaSmiTopo();

  if (topoOutput.empty()) {
    std::cout
        << "Failed to run nvidia-smi topo -m | awk '/^GPU/ {print $1, $(NF-1)}'"
        << std::endl;
    exit(0);
  }

  // Parse the output to extract NUMA affinities
  auto numaAffinities = parseNumaAffinities(topoOutput);

  if (numaAffinities.empty()) {
    std::cout << "No NUMA affinities found. Please check the output format.\n";
    return 1;
  }

  // Print all the GPU-NUMA mappings
  std::cout << "GPU to NUMA Affinity Mapping:\n";
  for (const auto &entry : numaAffinities) {
    std::cout << "GPU" << entry.first << " -> NUMA Node " << entry.second
              << std::endl;
  }

  for (const auto &entry : numaAffinities) {
    if (entry.first == 0) {
      numa_run_on_node(entry.second);
      std::cout << "Primary Numa Node: " << entry.second << std::endl;
    }
  }

  std::cout << "Running " << algorithm << " with edge size of " << edgeSize
            << "B and using " << nNGPUs << " neighbor GPUs" << std::endl;

  if (algorithm == "bfs") {
    std::cout << "Source vertex is: " << srcVertex << std::endl;

    if (edgeSize == _4BYTE)
      BFS32(filePath, srcVertex, nRuns, nNGPUs, numaAffinities);

    else if (edgeSize == _8BYTE)
      BFS64(filePath, srcVertex, nRuns, nNGPUs, numaAffinities);

    else {
      std::cout << "Error: wrong --edgeSize" << std::endl;
      usage(argv[0]);
    }

  } else if (algorithm == "cc") {
    if (edgeSize == _4BYTE)
      CC32(filePath, nRuns, nNGPUs, numaAffinities);

    else if (edgeSize == _8BYTE)
      CC64(filePath, nRuns, nNGPUs, numaAffinities);

    else {
      std::cout << "Error: wrong --edgeSize" << std::endl;
      usage(argv[0]);
    }

  } else if (algorithm == "sssp") {
    std::cout << "Source vertex is: " << srcVertex << std::endl;

    if (edgeSize == _4BYTE)
      SSSP32(filePath, srcVertex, nRuns, nNGPUs, numaAffinities);

    else if (edgeSize == _8BYTE)
      SSSP64(filePath, srcVertex, nRuns, nNGPUs, numaAffinities);

    else {
      std::cout << "Error: wrong --edgeSize" << std::endl;
      usage(argv[0]);
    }

  } else if (algorithm == "pr") {
    if (edgeSize == _4BYTE) {
      if (type == "push") {
        std::cout << "Running PageRank Push Implementation" << std::endl;
        PR32_PUSH(filePath, nRuns, nNGPUs, numaAffinities);
      }

      else if (type == "pull") {
        std::cout << "Running PageRank Pull Implementation" << std::endl;
        PR32(filePath, nRuns, nNGPUs, numaAffinities);
      }

      else {
        std::cout << "Error: wrong --type" << std::endl;
        usage(argv[0]);
      }
    } else if (edgeSize == _8BYTE) {
      if (type == "push") {
        std::cout << "Running PageRank Push Implementation" << std::endl;
        PR64_PUSH(filePath, nRuns, nNGPUs, numaAffinities);
      }

      else if (type == "pull") {
        std::cout << "Running PageRank Pull Implementation" << std::endl;
        PR64(filePath, nRuns, nNGPUs, numaAffinities);
      } else {
        std::cout << "Error: wrong --type" << std::endl;
        usage(argv[0]);
      }
    } else {
      std::cout << "Error: wrong --edgeSize" << std::endl;
      usage(argv[0]);
    }

  } else
    usage(argv[0]);

  // cudaFree(d_ptr);

  return 0;
}
