#ifndef CC_H
#define CC_H

#include "../common.cuh"
#include "../graph.cuh"
#include "cc_kernels.cuh"

void CC32(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs,
          std::unordered_map<int, int> affinityMap);

void CC64(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs,
          std::unordered_map<int, int> affinityMap);

#endif
