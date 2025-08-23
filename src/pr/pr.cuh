#ifndef PR_H
#define PR_H

#include "../common.cuh"
#include "../graph.cuh"
#include "pr_kernels.cuh"

void PR32(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs,
          std::unordered_map<int, int> affinityMap);

void PR64(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs,
          std::unordered_map<int, int> affinityMap);

void PR32_PUSH(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs,
               std::unordered_map<int, int> affinityMap);

void PR64_PUSH(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs,
               std::unordered_map<int, int> affinityMap);
#endif
