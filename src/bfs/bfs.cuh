#ifndef BFS_H
#define BFS_H

#include "../common.cuh"
#include "../graph.cuh"
#include "bfs_kernels.cuh"

void BFS32(std::string filePath, uint32 srcVertex, uint32 nRuns,
           uint32 nNeighborGPUs, std::unordered_map<int, int> affinityMap);

void BFS64(std::string filePath, uint32 srcVertex, uint32 nRuns,
           uint32 nNeighborGPUs, std::unordered_map<int, int> affinityMap);
#endif
