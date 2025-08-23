#include "pr_kernels.cuh"

// Atomic addition for double values
__device__ double atomicAddDouble(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Pull

// Testar o mesmo metodo com warps que utilizamos pro demand
__global__ void PR32_Static_Kernel(const uint32 *staticSize,
                                   const uint32 *d_staticList,
                                   const uint64 *d_offsets,
                                   const uint32 *d_staticEdges,
                                   double *d_valuesPR, uint32 *d_outDegree,
                                   double *d_sum, const bool *d_inStatic) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *staticSize; warpIdx += numWarps) {
    uint32 vertexId = d_staticList[warpIdx];

    // Neighbors to access
    uint64 start = d_offsets[vertexId];
    uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_staticEdges[i];

      if (d_outDegree[neighborId] != 0) {
        double tempValue = d_valuesPR[neighborId] / d_outDegree[neighborId];
        atomicAdd(&d_sum[vertexId], tempValue);
      }
    }
  }
}

__global__ void PR64_Static_Kernel(const uint64 *staticSize,
                                   const uint64 *d_staticList,
                                   const uint64 *d_offsets,
                                   const uint64 *d_staticEdges,
                                   double *d_valuesPR, uint32 *d_outDegree,
                                   double *d_sum, const bool *d_inStatic) {
  for (uint64 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *staticSize; index += blockDim.x * gridDim.x) {
    uint64 vertexId = d_staticList[index];

    // Pretty sure we can remove this but lets review it first
    if (d_inStatic[vertexId]) {
      // Neighbors to access
      uint64 startNeighbor = d_offsets[vertexId];
      uint64 endNeighbor = d_offsets[vertexId + 1];

      for (uint64 i = startNeighbor; i < endNeighbor; i++) {
        uint64 neighborId = d_staticEdges[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (d_outDegree[neighborId] != 0) {
          double tempValue =
              d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
          atomicAdd(&d_sum[vertexId], tempValue);
        }
      }
    }
  }
}

__global__ void PR32_Demand_Kernel(const uint32 *demandSize,
                                   const uint32 *d_demandList,
                                   const uint32 *h_edges,
                                   const uint64 *d_offsets, double *d_valuesPR,
                                   uint32 *d_outDegree, double *d_sum,
                                   const bool *d_inStatic) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *demandSize; warpIdx += numWarps) {
    // const uint32 traverseIndex = warpIdx;
    uint32 vertexId = d_demandList[warpIdx];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {
        uint32 neighborId = h_edges[i];

        if (d_outDegree[neighborId] != 0) {
          double tempValue = d_valuesPR[neighborId] / d_outDegree[neighborId];
          atomicAdd(&d_sum[vertexId], tempValue);
        }
      }
    }
  }
}

__global__ void PR32_Filter_Kernel(const uint32 *partitionList,
                                   uint32 *d_partitionsOffsets,
                                   uint32 *d_values, bool *d_frontier,
                                   const uint32 *d_filterEdges,
                                   const uint64 *d_offsets,
                                   bool *d_filterFrontier, double *d_valuesPR,
                                   uint32 *d_outDegree, double *d_sum) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      if (d_outDegree[neighborId] != 0) {
        double tempValue =
            d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
        atomicAdd(&d_sum[warpIdx], tempValue);
      }
    }
  }
}

__global__ void PR64_Filter_Kernel(const uint32 *partitionList,
                                   uint32 *d_partitionsOffsets,
                                   uint64 *d_values, bool *d_frontier,
                                   const uint64 *d_filterEdges,
                                   const uint64 *d_offsets,
                                   bool *d_filterFrontier, double *d_valuesPR,
                                   uint32 *d_outDegree, double *d_sum) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      if (d_outDegree[neighborId] != 0) {
        double tempValue =
            d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
        atomicAdd(&d_sum[warpIdx], tempValue);
      }
    }
  }
}

__global__ void PR32_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, uint32 *d_outDegree,
    double *d_sum) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      if (d_outDegree[neighborId] != 0) {
        double tempValue =
            d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
        atomicAdd(&d_sum[warpIdx], tempValue);
      }
    }
  }
}

__global__ void PR64_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, uint32 *d_outDegree,
    double *d_sum) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      if (d_outDegree[neighborId] != 0) {
        double tempValue =
            d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
        atomicAdd(&d_sum[warpIdx], tempValue);
      }
    }
  }
}
__global__ void PR32_Static_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, uint32 *d_outDegree,
    double *d_sum) {
  // {
  //   const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  //   uint32 warpIdx = tid >> WARP_SHIFT;
  //   const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  //   const uint32 numWarps =
  //       gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;
  //
  //   uint32 partition = partitionList[0];
  //   //  Start Edge
  //
  //   // Grid-Stride loop using Warp ID makes it easier to calculate with the
  //   .y
  //   // dimension
  //   for (warpIdx += d_partitionsOffsets[partition];
  //        warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
  //     if (!d_filterFrontier[warpIdx]) continue;
  //
  //     d_filterFrontier[warpIdx] = 0;
  //
  //     const uint64 start = d_offsets[warpIdx];
  //     const uint64 end = d_offsets[warpIdx + 1];
  //
  //     for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
  //       uint32 neighborId = d_filterEdges[i];
  //
  //       if (d_outDegree[neighborId] != 0) {
  //         double tempValue =
  //             d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
  //         atomicAdd(&d_sum[warpIdx], tempValue);
  //       }
  //     }
  //   }
  // }

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx];
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      if (d_outDegree[neighborId] != 0) {
        double tempValue =
            d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
        atomicAdd(&d_sum[warpIdx], tempValue);
      }
    }
  }
}

__global__ void PR64_Static_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, uint32 *d_outDegree,
    double *d_sum) {
  // {
  //   const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  //   uint32 warpIdx = tid >> WARP_SHIFT;
  //   const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  //   const uint32 numWarps =
  //       gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;
  //
  //   uint32 partition = partitionList[0];
  //   //  Start Edge
  //
  //   // Grid-Stride loop using Warp ID makes it easier to calculate with the
  //   .y
  //   // dimension
  //   for (warpIdx += d_partitionsOffsets[partition];
  //        warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
  //     if (!d_filterFrontier[warpIdx]) continue;
  //
  //     d_filterFrontier[warpIdx] = 0;
  //
  //     const uint64 start = d_offsets[warpIdx];
  //     const uint64 end = d_offsets[warpIdx + 1];
  //
  //     for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
  //       uint32 neighborId = d_filterEdges[i];
  //
  //       if (d_outDegree[neighborId] != 0) {
  //         double tempValue =
  //             d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
  //         atomicAdd(&d_sum[warpIdx], tempValue);
  //       }
  //     }
  //   }
  // }

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx];
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      if (d_outDegree[neighborId] != 0) {
        double tempValue =
            d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
        atomicAdd(&d_sum[warpIdx], tempValue);
      }
    }
  }
}

__global__ void PR64_Demand_Kernel(const uint64 *demandSize,
                                   const uint64 *d_demandList,
                                   const uint64 *h_edges,
                                   const uint64 *d_offsets, double *d_valuesPR,
                                   uint32 *d_outDegree, double *d_sum,
                                   const bool *d_inStatic) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *demandSize; warpIdx += numWarps) {
    const uint64 traverseIndex = warpIdx;
    uint64 vertexId = d_demandList[traverseIndex];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {
        uint64 neighborId = h_edges[i];

        if (d_outDegree[neighborId] != 0) {
          double tempValue =
              d_valuesPR[neighborId] / (double)d_outDegree[neighborId];
          atomicAdd(&d_sum[vertexId], tempValue);
        }
      }
    }
  }
}

__global__ void PR32_Update_Values(const uint32 *numVertices,
                                   double *d_valuesPR, double *d_sum,
                                   bool *d_frontier) {
  for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *numVertices; index += blockDim.x * gridDim.x) {
    if (d_frontier[index]) {
      double tempValue = (1 - ALPHA) + ALPHA * d_sum[index];
      // K tempValue = 0.15*valueD[index] + 0.85*sumD[index];

      double diff = tempValue > d_valuesPR[index]
                        ? (tempValue - d_valuesPR[index])
                        : (d_valuesPR[index] - tempValue);
      d_valuesPR[index] = tempValue;

      // Substituir este if else
      if (diff > TOLERANCE)
        d_frontier[index] = 1;

      else
        d_frontier[index] = 0;

      // if (diff <= TOLERANCE)
      //   d_frontier[index] = 0;

      d_sum[index] = 0;
    }
  }
}

__global__ void PR64_Update_Values(const uint64 *numVertices,
                                   double *d_valuesPR, double *d_sum,
                                   bool *d_frontier) {
  for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *numVertices; index += blockDim.x * gridDim.x) {
    if (d_frontier[index]) {
      double tempValue = (1 - ALPHA) + ALPHA * d_sum[index];
      // K tempValue = 0.15*valueD[index] + 0.85*sumD[index];

      double diff = tempValue > d_valuesPR[index]
                        ? (tempValue - d_valuesPR[index])
                        : (d_valuesPR[index] - tempValue);
      d_valuesPR[index] = tempValue;

      if (diff > TOLERANCE)
        d_frontier[index] = 1;

      else
        d_frontier[index] = 0;

      d_sum[index] = 0;
    }
  }
}

// Push
__global__ void PR32_Static_Kernel_PUSH(
    const uint32 *staticSize, const uint32 *d_staticList,
    const uint64 *d_offsets, const uint32 *d_staticEdges, bool *d_frontier,
    const bool *d_inStatic, double *d_delta, double *d_residual) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *staticSize; warpIdx += numWarps) {
    //
    // for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    //      index < *staticSize; index += blockDim.x * gridDim.x) {

    // if (!d_inStatic[warpIdx]) {
    //   continue;
    // }
    //
    // if (!d_frontier[warpIdx]) {
    //   continue;
    // }
    // uint32 vertexId = warpIdx;
    // d_frontier[warpIdx] = 0;

    uint32 vertexId = d_staticList[warpIdx];

    // Neighbors to access
    uint64 startNeighbor = d_offsets[vertexId];
    uint64 endNeighbor = d_offsets[vertexId + 1];

    for (uint64 i = startNeighbor + laneIdx; i < endNeighbor; i += WARP_SIZE) {
      uint32 neighborId = d_staticEdges[i];
      atomicAdd(&d_residual[neighborId], d_delta[vertexId]);
    }
  }
}

__global__ void PR64_Static_Kernel_PUSH(
    const uint64 *staticSize, const uint64 *d_staticList,
    const uint64 *d_offsets, const uint64 *d_staticEdges, bool *d_frontier,
    const bool *d_inStatic, double *d_delta, double *d_residual) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *staticSize; warpIdx += numWarps) {
    //
    // for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    //      index < *staticSize; index += blockDim.x * gridDim.x) {

    // if (!d_inStatic[warpIdx]) {
    //   continue;
    // }
    //
    // if (!d_frontier[warpIdx]) {
    //   continue;
    // }
    // uint32 vertexId = warpIdx;
    // d_frontier[warpIdx] = 0;

    uint32 vertexId = d_staticList[warpIdx];

    // Neighbors to access
    uint64 startNeighbor = d_offsets[vertexId];
    uint64 endNeighbor = d_offsets[vertexId + 1];

    for (uint64 i = startNeighbor + laneIdx; i < endNeighbor; i += WARP_SIZE) {
      uint32 neighborId = d_staticEdges[i];
      atomicAdd(&d_residual[neighborId], d_delta[vertexId]);
    }
  }
}

__global__ void PR32_Demand_Kernel_PUSH(const uint32 *demandSize,
                                        const uint32 *d_demandList,
                                        bool *d_frontier, const uint32 *h_edges,
                                        const uint64 *d_offsets,
                                        double *d_delta, double *d_residual) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *demandSize; warpIdx += numWarps) {
    const uint32 traverseIndex = warpIdx;
    uint32 vertexId = d_demandList[traverseIndex];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = h_edges[i];

      if (i >= start) atomicAdd(&d_residual[neighborId], d_delta[vertexId]);
    }
    // d_frontier[vertexId] = 0;
  }
}

__global__ void PR64_Demand_Kernel_PUSH(const uint64 *demandSize,
                                        const uint64 *d_demandList,
                                        bool *d_frontier, const uint64 *h_edges,
                                        const uint64 *d_offsets,
                                        double *d_delta, double *d_residual) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *demandSize; warpIdx += numWarps) {
    const uint32 traverseIndex = warpIdx;
    uint32 vertexId = d_demandList[traverseIndex];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = h_edges[i];

      if (i >= start) atomicAdd(&d_residual[neighborId], d_delta[vertexId]);
    }
    // d_frontier[vertexId] = 0;
  }
}

__global__ void PR32_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, double *d_residual,
    double *d_delta) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      atomicAdd(&d_residual[neighborId], d_delta[warpIdx]);
    }
  }
}

__global__ void PR64_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, double *d_residual,
    double *d_delta) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      atomicAdd(&d_residual[neighborId], d_delta[warpIdx]);
    }
  }
}

__global__ void PR32_Static_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, double *d_residual,
    double *d_delta) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  // uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx];
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      atomicAdd(&d_residual[neighborId], d_delta[warpIdx]);
    }
  }
}

__global__ void PR64_Static_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, double *d_residual,
    double *d_delta) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  // uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx];
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      atomicAdd(&d_residual[neighborId], d_delta[warpIdx]);
    }
  }
}
__global__ void PR32_NeighborFilter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, double *d_residual,
    double *d_delta) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      atomicAdd(&d_residual[neighborId], d_delta[warpIdx]);
    }
  }
}

__global__ void PR64_NeighborFilter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, double *d_valuesPR, double *d_residual,
    double *d_delta) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      atomicAdd(&d_residual[neighborId], d_delta[warpIdx]);
    }
  }
}
__global__ void PR32_Update_Values_PUSH(const uint32 *numVertices,
                                        double *d_valuesPR, bool *d_frontier,
                                        const uint64 *d_offsets,
                                        double *d_delta, double *d_residual) {
  for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *numVertices; index += blockDim.x * gridDim.x) {
    if (d_residual[index] > TOLERANCE) {
      d_valuesPR[index] += d_residual[index];
      d_delta[index] =
          d_residual[index] * ALPHA / (d_offsets[index + 1] - d_offsets[index]);
      d_residual[index] = 0.0f;
      d_frontier[index] = 1;
    }
  }
}

__global__ void PR64_Update_Values_PUSH(const uint64 *numVertices,
                                        double *d_valuesPR, bool *d_frontier,
                                        const uint64 *d_offsets,
                                        double *d_delta, double *d_residual) {
  for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *numVertices; index += blockDim.x * gridDim.x) {
    if (d_residual[index] > TOLERANCE) {
      d_valuesPR[index] += d_residual[index];
      d_delta[index] =
          d_residual[index] * ALPHA / (d_offsets[index + 1] - d_offsets[index]);
      d_residual[index] = 0.0f;
      d_frontier[index] = 1;
    }
  }
}
