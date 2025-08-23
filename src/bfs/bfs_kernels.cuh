#ifndef BFS_KERNELS_H
#define BFS_KERNELS_H

#include "../common.cuh"

__global__ void BFS32_Static_Kernel(const uint32 *h_numVertices,
                                    const uint32 *d_staticList,
                                    const uint64 *d_offsets,
                                    const uint32 *d_staticEdges,
                                    uint32 *d_values, bool *d_frontier,
                                    bool *d_staticFrontier);

__global__ void BFS64_Static_Kernel(const uint64 *h_numVertices,
                                    const uint64 *d_staticList,
                                    const uint64 *d_offsets,
                                    const uint64 *d_staticEdges,
                                    uint64 *d_values, bool *d_frontier,
                                    const bool *d_inStatic);

__global__ void BFS32_Demand_Kernel(const uint32 *demandSize,
                                    const uint32 *d_demandList,
                                    uint32 *d_values, bool *d_frontier,
                                    const uint32 *h_edges,
                                    const uint64 *d_offsets);

__global__ void BFS64_Demand_Kernel(const uint64 *demandSize,
                                    const uint64 *d_demandList,
                                    uint64 *d_values, bool *d_frontier,
                                    const uint64 *h_edges,
                                    const uint64 *d_offsets);

__global__ void BFS32_Filter_Kernel(const uint32 *partitionList,
                                    uint32 *d_partitionsOffsets,
                                    uint32 *d_values, bool *d_frontier,
                                    const uint32 *d_filterEdges,
                                    const uint64 *d_offsets,
                                    bool *d_filterFrontier);

__global__ void BFS64_Filter_Kernel(const uint32 *partitionList,
                                    uint32 *d_partitionsOffsets,
                                    uint64 *d_values, bool *d_frontier,
                                    const uint64 *d_filterEdges,
                                    const uint64 *d_offsets,
                                    bool *d_filterFrontier);

__global__ void BFS64_NeighborFilter_Kernel(const uint32 *partitionList,
                                            uint32 *d_partitionsOffsets,
                                            uint64 *d_values, bool *d_frontier,
                                            const uint64 *d_filterEdges,
                                            const uint64 *d_offsets,
                                            bool *d_filterFrontier);
// Purely for profiling purposes
__global__ void BFS32_NeighborFilter_Kernel(const uint32 *partitionList,
                                            uint32 *d_partitionsOffsets,
                                            uint32 *d_values, bool *d_frontier,
                                            const uint32 *d_filterEdges,
                                            const uint64 *d_offsets,
                                            bool *d_filterFrontier);

__global__ void BFS32_Static_Filter_Kernel(const uint32 *partitionList,
                                           uint32 *d_partitionsOffsets,
                                           uint32 *d_values, bool *d_frontier,
                                           const uint32 *d_filterEdges,
                                           const uint64 *d_offsets,
                                           bool *d_filterFrontier);

#endif
