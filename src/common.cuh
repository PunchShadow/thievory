#ifndef COMMON_CUH
#define COMMON_CUH

// Cuda runtime API
#include <cuda.h>
#include <cuda_runtime.h>

#include "timer.hpp"
// Thrust utilities
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <set>
#include <unordered_map>

typedef unsigned int uint32;       // 4 byte data type
typedef unsigned long long uint64; // 8 byte data type

#define THREADS_PER_BLOCK 1024 // Threads per block set

#define WARP_SHIFT 5 // Warp shift to get laneID
#define WARP_SIZE 32 // Warp Size
#define WARP_SIZE_INVERSE 0.03125f

#define WARPS_PER_BLOCK                                                        \
  (THREADS_PER_BLOCK / WARP_SIZE) // Coincidentally it is Warp Size (32)

#define MEM_ALIGN_32 (~(0x1fULL)) // Align mask for 4 byte data type
#define MEM_ALIGN_64 (~(0xfULL))  // Align mask for 8 byte data type

#define FRAGMENT 4096 // 16kB for 4Byte number (Static Edges Chunk Size)

#define PARTITION_SIZE_32 8388608 * 2 // Partition size for 4 byte data (32 MB)
#define PARTITION_SIZE_64 4194304     // Partition size for 8 byte data (32 MB)

// #define PARTITION_SIZE_MB 33554432
#define PARTITION_SIZE_MB 67108864
// #define PARTITION_SIZE_MB 16777216
// #define EDGES_IN_PARTITION 8388608 // 32 MB with 4B edge

#define EDGES_IN_PARTITION 134217728 + 67108864 // 768MB with 4B edge

#define N_FILTER_STREAMS 64 / 4
#define N_TARGET_FILTER_STREAMS 24 / 4

#define TOLERANCE 0.001f // Page Rank Specific
#define ALPHA 0.85f      // Page Rank Specific

#define GPUAssert(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
enum ALGORITHM_TYPE { BFS, SSSP, CC, PR };

__constant__ float d_filterThreshold;

// GPU Kernels
template <typename EdgeType>
__global__ void
setStaticNDemandFrontiers(EdgeType *h_numVertices, bool *d_frontier,
                          bool *d_staticFrontier, bool *d_demandFrontier,
                          bool *d_inStatic) {
  for (EdgeType vertexId = blockIdx.x * blockDim.x + threadIdx.x;
       vertexId < *h_numVertices; vertexId += blockDim.x * gridDim.x) {
    if (d_frontier[vertexId]) {
      if (d_inStatic[vertexId])
        d_staticFrontier[vertexId] = 1;
      else
        d_demandFrontier[vertexId] = 1;
    }
  }
}

template <typename EdgeType>
__global__ void setStaticList(EdgeType *h_numVertices, EdgeType *d_staticList,
                              bool *d_staticFrontier, EdgeType *d_prefixSum) {
  for (EdgeType vertexId = blockIdx.x * blockDim.x + threadIdx.x;
       vertexId < *h_numVertices; vertexId += blockDim.x * gridDim.x) {
    if (d_staticFrontier[vertexId]) {
      d_staticList[d_prefixSum[vertexId]] = vertexId;
      d_staticFrontier[vertexId] = 0;
    }
  }
}

// Maybe unify these two
template <typename EdgeType>
__global__ void setDemandList(EdgeType *h_numVertices, EdgeType *d_demandList,
                              bool *d_demandFrontier, EdgeType *d_prefixSum) {
  for (EdgeType vertexId = blockIdx.x * blockDim.x + threadIdx.x;
       vertexId < *h_numVertices; vertexId += blockDim.x * gridDim.x) {
    if (d_demandFrontier[vertexId]) {
      d_demandList[d_prefixSum[vertexId]] = vertexId;
      d_demandFrontier[vertexId] = 0;
    }
  }
}

// PR PUSH
template <typename EdgeType>
__global__ void setFrontier(EdgeType *activeNum, EdgeType *activeNodes,
                            bool *d_frontier) {
  for (EdgeType vertexId = blockIdx.x * blockDim.x + threadIdx.x;
       vertexId < *activeNum; vertexId += blockDim.x * gridDim.x) {
    if (d_frontier[activeNodes[vertexId]])
      d_frontier[activeNodes[vertexId]] = 0;
  }
}

template <typename EdgeType>
__global__ void CalculateActiveEdgesPerPartition(
    uint32 *numPartitions, uint32 *d_partitionsOffsets, uint64 *d_offsets,
    float *d_partitionCost, bool *d_demandFrontier, bool *d_filterFrontier) {

  uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / 256;
  uint32 warpIdx = tid >> 8;
  const uint32 laneIdx = tid & ((1 << 8) - 1);

  for (; warpIdx < *numPartitions; warpIdx += numWarps) {
    const uint64 start = d_partitionsOffsets[warpIdx];
    const uint64 end = d_partitionsOffsets[warpIdx + 1];

    uint32 activeEdgesInCurrentPartition = 0;

    for (uint64 vertexId = start + laneIdx; vertexId < end; vertexId += 256) {
      if (d_demandFrontier[vertexId])
        activeEdgesInCurrentPartition +=
            d_offsets[vertexId + 1] - d_offsets[vertexId];
    }

    atomicAdd(&d_partitionCost[warpIdx], activeEdgesInCurrentPartition);
  }
}

template <typename EdgeType>
__global__ void
CalculateActiveEdgesRatio(uint32 *numPartitions, uint32 *d_partitionsOffsets,
                          uint64 *d_offsets, float *d_partitionCost,
                          bool *d_demandFrontier, bool *d_filterFrontier) {
  uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (; tid < *numPartitions; tid += blockDim.x * gridDim.x) {
    const uint32 edgesInCurrentPartition =
        d_offsets[d_partitionsOffsets[tid + 1]] -
        d_offsets[d_partitionsOffsets[tid]];

    d_partitionCost[tid] /= edgesInCurrentPartition;
  }
}

template <typename EdgeType>
__global__ void SplitZeroCopyNFilterFrontiers(
    uint32 *numPartitions, uint32 *d_partitionsOffsets, uint64 *d_offsets,
    float *d_partitionCost, bool *d_demandFrontier, bool *d_filterFrontier) {
  for (uint32 partition = 0; partition < *numPartitions; partition++) {
    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    const uint64 start = d_partitionsOffsets[partition];
    const uint64 end = d_partitionsOffsets[partition + 1];

    // Grid-Stride loop across offsets
    for (tid += start; tid < end; tid += blockDim.x * gridDim.x) {
      // If the vertex is active
      if (d_demandFrontier[tid] &&
          d_partitionCost[partition] > d_filterThreshold) {
        d_filterFrontier[tid] = 1; // Swap these two
        d_demandFrontier[tid] = 0;
      }
    }
    // d_zerocopyPartitionCost[partition] = 0;
  }
}

#endif // COMMON_CUH
