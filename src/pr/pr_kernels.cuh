#ifndef PR_KERNELS_H
#define PR_KERNELS_H

#include "../common.cuh"

// Pull PR
__global__ void PR32_Static_Kernel(const uint32 *staticSize,
                                   const uint32 *d_staticList,
                                   const uint64 *d_offsets,
                                   const uint32 *d_staticEdges,
                                   float *d_valuesPR, uint32 *d_outDegree,
                                   float *d_sum, const bool *d_inStatic);

__global__ void PR64_Static_Kernel(const uint64 *staticSize,
                                   const uint64 *d_staticList,
                                   const uint64 *d_offsets,
                                   const uint64 *d_staticEdges,
                                   float *d_valuesPR, uint32 *d_outDegree,
                                   float *d_sum, const bool *d_inStatic);

__global__ void PR32_Demand_Kernel(const uint32 *demandSize,
                                   const uint32 *d_demandList,
                                   const uint32 *h_edges,
                                   const uint64 *d_offsets, float *d_valuesPR,
                                   uint32 *d_outDegree, float *d_sum,
                                   const bool *d_inStatic);

__global__ void PR64_Demand_Kernel(const uint64 *demandSize,
                                   const uint64 *d_demandList,
                                   const uint64 *h_edges,
                                   const uint64 *d_offsets, float *d_valuesPR,
                                   uint32 *d_outDegree, float *d_sum,
                                   const bool *d_inStatic);

__global__ void PR32_Filter_Kernel(const uint32 *partitionList,
                                   uint32 *d_partitionsOffsets,
                                   uint32 *d_values, bool *d_frontier,
                                   const uint32 *d_filterEdges,
                                   const uint64 *d_offsets,
                                   bool *d_filterFrontier, float *d_valuesPR,
                                   uint32 *d_outDegree, float *d_sum);

__global__ void PR64_Filter_Kernel(const uint32 *partitionList,
                                   uint32 *d_partitionsOffsets,
                                   uint64 *d_values, bool *d_frontier,
                                   const uint64 *d_filterEdges,
                                   const uint64 *d_offsets,
                                   bool *d_filterFrontier, float *d_valuesPR,
                                   uint32 *d_outDegree, float *d_sum);

__global__ void PR32_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, uint32 *d_outDegree,
    float *d_sum);

__global__ void PR32_Update_Values(const uint32 *numVertices,
                                   float *d_valuesPR, float *d_sum,
                                   bool *d_frontier);

__global__ void PR64_Update_Values(const uint64 *numVertices,
                                   float *d_valuesPR, float *d_sum,
                                   bool *d_frontier);

// Push PR
__global__ void PR32_Static_Kernel_PUSH(
    const uint32 *staticSize, const uint32 *d_staticList,
    const uint64 *d_offsets, const uint32 *d_staticEdges, bool *d_frontier,
    const bool *d_inStatic, float *d_delta, float *d_residual);

__global__ void PR64_Static_Kernel_PUSH(
    const uint64 *staticSize, const uint64 *d_staticList,
    const uint64 *d_offsets, const uint64 *d_staticEdges, bool *d_frontier,
    const bool *d_inStatic, float *d_delta, float *d_residual);

__global__ void PR32_Demand_Kernel_PUSH(const uint32 *demandSize,
                                        const uint32 *d_demandList,
                                        bool *d_frontier, const uint32 *h_edges,
                                        const uint64 *d_offsets,
                                        float *d_delta, float *d_residual);

__global__ void PR64_Demand_Kernel_PUSH(const uint64 *demandSize,
                                        const uint64 *d_demandList,
                                        bool *d_frontier, const uint64 *h_edges,
                                        const uint64 *d_offsets,
                                        float *d_delta, float *d_residual);

__global__ void PR32_Update_Values_PUSH(const uint32 *numVertices,
                                        float *d_valuesPR, bool *d_frontier,
                                        const uint64 *d_offsets,
                                        float *d_delta, float *d_residual);

__global__ void PR32_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, float *d_residual,
    float *d_delta);

__global__ void PR64_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, float *d_residual,
    float *d_delta);

__global__ void PR64_Static_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, float *d_residual,
    float *d_delta);

__global__ void PR64_NeighborFilter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, float *d_residual,
    float *d_delta);

__global__ void PR32_NeighborFilter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, float *d_residual,
    float *d_delta);

__global__ void PR32_Static_Filter_Kernel_PUSH(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, float *d_residual,
    float *d_delta);

__global__ void PR32_Static_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, uint32 *d_outDegree,
    float *d_sum);

__global__ void PR64_Static_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, uint32 *d_outDegree,
    float *d_sum);

__global__ void PR64_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint64 *d_values,
    bool *d_frontier, const uint64 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, float *d_valuesPR, uint32 *d_outDegree,
    float *d_sum);

__global__ void PR64_Update_Values_PUSH(const uint64 *numVertices,
                                        float *d_valuesPR, bool *d_frontier,
                                        const uint64 *d_offsets,
                                        float *d_delta, float *d_residual);
#endif
