#include "sssp_kernels.cuh"

// Testar o mesmo metodo com warps que utilizamos pro demand
__global__ void SSSP32_Static_Kernel(const uint32 *staticSize,
                                     const uint32 *d_staticList,
                                     const uint64 *d_offsets,
                                     const uint32 *d_staticEdges,
                                     const uint32 *d_staticWeights,
                                     uint32 *d_values, bool *d_frontier,
                                     bool *d_staticFrontier) {
  // for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
  //      index < *staticSize; index += blockDim.x * gridDim.x) {
  //   uint32 vertexId = d_staticList[index];
  //
  //   // if (!d_staticFrontier[index])
  //   //     continue;
  //
  //   // SSSP specific
  //   uint32 sourceValue = d_values[vertexId];
  //
  //   // Neighbors to access
  //   uint64 startNeighbor = d_offsets[vertexId];
  //   uint64 endNeighbor = d_offsets[vertexId + 1];
  //
  //   for (uint32 i = startNeighbor; i < endNeighbor; i++) {
  //     uint32 neighborId = d_staticEdges[i];
  //
  //     uint32 newValue = sourceValue + d_staticWeights[i];
  //     // If this new path has lower cost than the previous then change and
  //     add
  //     // the neighbor to the frontier
  //     if (newValue < d_values[neighborId]) {
  //       atomicMin(&d_values[neighborId], newValue);
  //       d_frontier[neighborId] = 1;
  //     }
  //   }
  //   // d_staticFrontier[index] = 0;
  // }

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *staticSize; warpIdx += numWarps) {
    // if (!d_staticFrontier[warpIdx])
    //     continue;

    // const uint32 traverseIndex = warpIdx;
    uint32 vertexId = d_staticList[warpIdx];

    uint32 newValue = d_values[vertexId] + 20;

    const uint64 start = d_offsets[vertexId];
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_staticEdges[i];

      // uint32 newValue = sourceValue + d_staticWeights[i];
      //  If this new path has lower cost than the previous then change and add
      //  the neighbor to the frontier
      if (newValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], newValue);
        d_frontier[neighborId] = 1;
      }
    }
    // d_staticFrontier[warpIdx] = 0;
  }
}

__global__ void SSSP64_Static_Kernel(const uint64 *staticSize,
                                     const uint64 *d_staticList,
                                     const uint64 *d_offsets,
                                     const uint64 *d_staticEdges,
                                     const uint64 *d_staticWeights,
                                     uint64 *d_values, bool *d_frontier,
                                     const bool *d_inStatic) {
  for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *staticSize; index += blockDim.x * gridDim.x) {
    uint64 vertexId = d_staticList[index];

    // Pretty sure we can remove this but lets review it first
    if (d_inStatic[vertexId]) {
      // SSSP specific
      uint64 sourceValue = d_values[vertexId];

      // Neighbors to access
      uint64 startNeighbor = d_offsets[vertexId];
      uint64 endNeighbor = d_offsets[vertexId + 1];

      for (uint64 i = startNeighbor; i < endNeighbor; i++) {
        uint64 neighborId = d_staticEdges[i];
        uint64 newValue = sourceValue + d_staticWeights[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (newValue < d_values[neighborId]) {
          atomicMin(&d_values[neighborId], newValue);
          d_frontier[neighborId] = 1;
        }
      }
    }
  }
}

__global__ void SSSP32_Demand_Kernel(const uint32 *demandSize,
                                     const uint32 *d_demandList,
                                     uint32 *d_values, bool *d_frontier,
                                     const uint32 *h_edges,
                                     const uint32 *h_weights,
                                     const uint64 *d_offsets) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK /
                          WARP_SIZE;  // Try to avoid this

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *demandSize; warpIdx += numWarps) {
    // const uint32 traverseIndex = warpIdx;
    const uint32 vertexId = d_demandList[warpIdx];

    const uint32 sourceValue = d_values[vertexId];
    uint32 newValue;
    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {
        uint32 neighborId = h_edges[i];  // Both of these accesses are aligned
        newValue = sourceValue + h_weights[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (newValue < d_values[neighborId]) {
          atomicMin(&d_values[neighborId], newValue);
          d_frontier[neighborId] = 1;
        }
      }
    }
  }
}

__global__ void SSSP64_Demand_Kernel(const uint64 *demandSize,
                                     const uint64 *d_demandList,
                                     uint64 *d_values, bool *d_frontier,
                                     const uint64 *h_edges,
                                     const uint64 *h_weights,
                                     const uint64 *d_offsets) {
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
    uint64 vertexId = d_demandList[traverseIndex];

    // uint64 srcValue = d_values[id];
    uint64 sourceValue = d_values[vertexId];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_64;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {
        uint64 neighborId = h_edges[i];
        uint64 newValue = sourceValue + h_weights[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (newValue < d_values[neighborId]) {
          atomicMin(&d_values[neighborId], newValue);
          d_frontier[neighborId] = 1;
        }
      }
    }
  }
}

__global__ void SSSP64_Filter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, const uint64 *d_filterWeights) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint64 startEdge = d_offsets[d_partitionsOffsets[partition]];

  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    uint64 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint64 neighborId = d_filterEdges[i];

      uint64 newValue = sourceValue + d_filterWeights[i];
      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (newValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], newValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}

__global__ void SSSP32_Filter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, const uint32 *d_filterWeights) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      uint64 newValue = sourceValue + d_filterWeights[i];
      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (newValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], newValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}

__global__ void SSSP64_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, const uint64 *d_filterWeights) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  uint64 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    uint64 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint64 neighborId = d_filterEdges[i];

      uint64 newValue = sourceValue + d_filterWeights[i];
      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (newValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], newValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}

__global__ void SSSP32_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, const uint32 *d_filterWeights) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      uint64 newValue = sourceValue + d_filterWeights[i];
      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (newValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], newValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}

__global__ void SSSP32_Static_Filter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, const uint32 *d_filterWeights) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  // uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx];
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      uint64 newValue = sourceValue + d_filterWeights[i];
      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (newValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], newValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}

__global__ void SSSP64_Static_Filter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, const uint64 *d_filterWeights) {
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  // uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx]) continue;

    d_filterFrontier[warpIdx] = 0;

    uint64 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx];
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint64 neighborId = d_filterEdges[i];

      uint64 newValue = sourceValue + d_filterWeights[i];
      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (newValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], newValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}
