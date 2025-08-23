// #ifndef UTILS_HPP
// #define UTILS_HPP

#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

typedef unsigned int uint32; // 4 byte data type
std::string runNvidiaSmiTopo() {
  std::string command = "nvidia-smi topo -m | awk '/^GPU/ {print $1, $(NF-1)}'";
  std::string output;
  char buffer[128];

  // Open a pipe to run the command
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"),
                                                pclose);

  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  // Read the output from the command
  while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
    output += buffer;
  }

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

int main() {

  std::string topoOutput = runNvidiaSmiTopo();

  if (topoOutput.empty()) {
    std::cout
        << "Failed to run nvidia-smi topo -m | awk '/^GPU/ {print $1, $(NF-1)}'"
        << std::endl;
    exit(0);
  }

  // Parse the output to extract NUMA affinities
  // auto numaAffinities = parseNumaAffinities(topoOutput);

  std::unordered_map<uint32, uint32> numaAffinities = {
      {0, 3}, // GPU0 -> NUMA Node 3
      {1, 1}, // GPU1 -> NUMA Node 1
      {2, 7}, // GPU2 -> NUMA Node 7
      {3, 3}  // GPU3 -> NUMA Node 3
  };

  if (numaAffinities.empty()) {
    std::cout << "No NUMA affinities found. Please check the output format.\n";
    return 1;
  }

  // Print all the GPU-NUMA mappings
  std::cout << "GPU to NUMA Affinity Mapping:\n";
  for (const auto &entry : numaAffinities) {
    std::cout << "GPU" << entry.first << " -> NUMA Node " << entry.second
              << "\n";
  }

  std::set<uint32> uniqueNumaNodes;
  for (const auto &entry : numaAffinities) {
    uniqueNumaNodes.insert(entry.second);
  }

  // Convert set to vector
  std::vector<uint32> numaNodesIndexing(uniqueNumaNodes.begin(),
                                        uniqueNumaNodes.end());

  // Create a map from NUMA nodes to their indices
  std::unordered_map<uint32, uint32> numaNodeToIndex;
  for (uint32 i = 0; i < numaNodesIndexing.size(); ++i) {
    numaNodeToIndex[numaNodesIndexing[i]] = i;
  }

  // Print the unique NUMA nodes and their indices for verification
  std::cout << "NUMA Node -> Index Mapping:" << std::endl;
  for (const auto &pair : numaNodeToIndex) {
    std::cout << "NUMA Node " << pair.first << " -> Index " << pair.second
              << std::endl;
  }

  // Step 2: Create a map from GPU IDs to indices in h_edges
  std::unordered_map<uint32, uint32> gpuToIndex;
  for (const auto &entry : numaAffinities) {
    uint32 gpuId = entry.first;
    uint32 numaNode = entry.second;
    gpuToIndex[gpuId] = numaNodeToIndex[numaNode];
  }

  // Print the GPU ID to h_edges index mapping
  std::cout << "GPU ID -> h_edges Index Mapping:" << std::endl;
  for (const auto &pair : gpuToIndex) {
    std::cout << "GPU" << pair.first << " -> Index " << pair.second
              << std::endl;
  }

  return 0;
}

// #endif // UTILS_HPP
