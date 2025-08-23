#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

std::string runNvidiaSmiTopo() {
  std::string command = "nvidia-smi topo -m | awk '/^GPU/ {print $1, $(NF-1)}'";
  std::string output;
  char buffer[128];

  // Open a pipe to run the command
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"),
                                                pclose);

  if (!pipe)
    throw std::runtime_error("popen() failed!");

  // Read the output from the command
  while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr)
    output += buffer;

  return output;
}

// Function to parse the output and store the NUMA affinities in a map
std::unordered_map<int, int>
parseNumaAffinities(const std::string &topoOutput) {
  std::unordered_map<int, int> numaAffinities;
  std::istringstream stream(topoOutput);
  std::string line;

  while (std::getline(stream, line)) {
    std::istringstream lineStream(line);
    std::string gpu;
    int numaNode;

    // Extract GPU ID and NUMA node from each line
    lineStream >> gpu >> numaNode;

    // Remove "GPU" prefix and convert to integer
    int gpuId = std::stoi(gpu.substr(3));
    numaAffinities[gpuId] = numaNode;
  }

  return numaAffinities;
}

// Function to get the NUMA affinity for a given GPU ID from the map
int getNumaAffinity(int gpuId,
                    const std::unordered_map<int, int> &numaAffinities) {
  auto it = numaAffinities.find(gpuId);
  if (it != numaAffinities.end()) {
    return it->second;
  }
  return -1; // Indicates that the GPU ID was not found
}

#endif // UTILS_HPP
