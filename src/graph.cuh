#ifndef GRAPH_CUH
#define GRAPH_CUH

#include "common.cuh"
#include "numa.h"

#include <fstream>
#include <string>
#include <vector>

template <class EdgeType> class CSR {
public:
  ALGORITHM_TYPE algorithm; // Chosen algorithm
  std::string type;
  uint32 nNGPUs;
  EdgeType *numVertices; // Total number of vertices transferred via Zero-Copy
  uint64 numEdges;       // Total number of edges
  uint64 numStaticEdges; // Number of edges in device memory

  EdgeType *frontierSize; // Redo this as a EdgeType
  EdgeType *staticSize;   // Static Size transferred via Zero-Copy
  EdgeType *demandSize;   // Demand Size transferred via Zero-Copy

  std::vector<uint32> h_partitionsOffsets;
  // uint32 *h_partitionsOffsets; // Offsets array for partitions
  uint32 *d_partitionsOffsets; // Offsets array for partitions

  uint64 *h_offsets; // Vertices array
  uint64 *d_offsets; // Device vertices array

  EdgeType *h_edges;       // Edges in host memory
  EdgeType *d_staticEdges; // Edges in device memory

  EdgeType *d_edges;
  EdgeType *d_weights;
  std::unordered_map<uint32, uint32> GPUAffinityMap;
  EdgeType **h_edges2; // Edges in host memory
  EdgeType *h_weights; // Host weights

  EdgeType **h_weights2;     // Host weights
  EdgeType *d_staticWeights; // Device weights

  EdgeType *h_values; // Host values array
  EdgeType *d_values; // Device values array

  // Partition stuff
  uint32 *numPartitions;
  uint64 maxEdgesInPartition = std::floor(PARTITION_SIZE_MB / sizeof(EdgeType));

  // Filtering partitions data
  bool *h_filterFrontier;
  bool *d_filterFrontier; // Device Filter Frontier

  float *h_filterPartitionCost;
  float *d_filterPartitionCost; // Device transfer cost for partitions given
                                // filtering approach

  float *h_partitionCost;
  float *d_partitionCost;

  EdgeType *d_filterEdges[N_TARGET_FILTER_STREAMS];   // Filter partition to be
                                                      // transferred
  EdgeType *d_filterWeights[N_TARGET_FILTER_STREAMS]; // Device weights

  EdgeType ***d_nFilterEdges;
  EdgeType ***d_nFilterWeights;

  float h_filterThreshold;

  uint32 avgVertPerPart;

  // Partition list
  uint32 *h_partitionList;
  uint32 *d_partitionList;

  // Neighbors partition list
  uint32 **d_nPartList;
  uint32 **h_nPartList;

  bool *h_finished;
  bool *d_finished;

  bool *d_inStatic; // Data in shared memory

  bool *h_frontier;                        // Host Frontier
  bool *d_frontier;                        // Device Frontier
  thrust::device_ptr<bool> thrustFrontier; // Wrapper for Frontier

  bool *d_staticFrontier; // Device frontier in static memory
  EdgeType *d_staticList; // Device list for frontier vertices in static region
  thrust::device_ptr<bool> thurstStaticFrontier; // Wrapper for Static Frontier

  bool *d_demandFrontier; // Frontier inside host memory
  EdgeType *d_demandList; //  Device list for frontier vertices in host memory
  thrust::device_ptr<bool>
      thurstDemandFrontier; // Wrapper for On-Demand Frontier

  EdgeType *d_prefixSum; // Auxiliary array for Prefix Sum
  thrust::device_ptr<EdgeType> thurstPrefixSum; // Wrapper for Prefix Sum

  EdgeType srcVertex; // Source vertex for BFS

  uint32 *h_degree; // Pull PR specific
  uint32 *d_degree; // Pull PR specific

  double *h_valuesPR; // Host values array for each edge (PR only)
  double *d_valuesPR; // Device values array for each edge (PR only)

  double *d_sum;                          // PR only PUSH
  thrust::device_ptr<double> d_thurstSum; // PR only PUSH

  double *d_delta;    // PR only PUSH
  double *d_residual; // PR only PUSH

  double *h_delta;
  double *h_residual;

  // Streams
  cudaStream_t staticStream, demandStream, frontierStream;

  std::vector<std::array<cudaStream_t, N_FILTER_STREAMS>> neighborMemCpyStreams;

  std::vector<std::array<cudaStream_t, N_FILTER_STREAMS>>
      neighborComputeStreams;

  cudaStream_t streams[N_TARGET_FILTER_STREAMS];

  // Reset frontier and values arrays
  void ResetFrontierNValues();

  // Initializes frontier and values arrays
  void InitFrontierNValues();

  // Sets the maximum partition size for edges in static memory
  void SetPartitionsConfig();

  // Sets the maximum partition size for edges in static memory
  void SetNumStaticEdges();

  // Initializes graph
  void InitData();

  // Reads input file
  void ReadInputFile(const std::string &filePath, ALGORITHM_TYPE algo,
                     uint64 sourceVertex, uint32 numberNGPUs,
                     std::unordered_map<int, int> affinityMap,
                     std::string algoType = "push");

  void SetFrontierToRatio(float ratio);

  // Dumps results to file
  void DumpValues();

  // Frees all allocated arrays in host and device
  void Free();
};

template <class EdgeType> void CSR<EdgeType>::SetFrontierToRatio(float ratio) {

  for (uint32 i = 0; i < *numVertices; i++) {
    h_frontier[i] = 0;
  }

  double totalRatio = 0;
  for (uint32 i = 0; i < *numPartitions; i++) {
    const uint64 edgesInCurrentPartition =
        h_offsets[h_partitionsOffsets[i + 1]] -
        h_offsets[h_partitionsOffsets[i]];

    std::vector<uint32> frontierList;

    const uint64 start = h_partitionsOffsets[i];
    const uint64 end = h_partitionsOffsets[i + 1];

    const uint64 minDesiredEdges = ratio * edgesInCurrentPartition;
    const uint64 maxDesiredEdges = (ratio + 0.05f) * edgesInCurrentPartition;
    //
    // std::cout << "Min edges: " << minDesiredEdges
    //           << " Max Edges: " << maxDesiredEdges << std::endl;

    double partRatio = 0;
    for (uint32 vertexId = start; vertexId < end - 14000; vertexId++) {
      uint64 newEdges = h_offsets[vertexId + 1] - h_offsets[vertexId];
      partRatio += newEdges;
    }

    partRatio /= edgesInCurrentPartition;
    totalRatio += partRatio;
  }
  totalRatio /= *numPartitions;

  std::cout << "Avg partition ratio: " << totalRatio * 100 << std::endl;

  return;
}

template <class EdgeType> void CSR<EdgeType>::ResetFrontierNValues() {
  if (algorithm == PR) {
    cudaMemcpy(d_valuesPR, h_valuesPR, *numVertices * sizeof(*h_valuesPR),
               cudaMemcpyHostToDevice);

    if (type == "push") {
      cudaMemcpy(d_delta, h_delta, *numVertices * sizeof(*h_delta),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_residual, h_residual, *numVertices * sizeof(*h_residual),
                 cudaMemcpyHostToDevice);
    }

  } else
    cudaMemcpy(d_values, h_values, *numVertices * sizeof(*d_values),
               cudaMemcpyHostToDevice);

  cudaMemcpy(d_frontier, h_frontier, *numVertices * sizeof(*d_frontier),
             cudaMemcpyHostToDevice);

  cudaMemset(d_staticFrontier, 0, *numVertices * sizeof(*d_staticFrontier));

  cudaMemset(d_demandFrontier, 0, *numVertices * sizeof(*d_demandFrontier));

  cudaMemset(d_filterFrontier, 0, *numVertices * sizeof(*d_filterFrontier));

  cudaMemset(d_inStatic, 0, *numVertices * sizeof(*d_inStatic));

  cudaDeviceSynchronize();

  return;
}

template <class EdgeType> void CSR<EdgeType>::InitFrontierNValues() {
  switch (algorithm) {
  case PR:
    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 1;

      if (type == "push") {
        h_valuesPR[i] = (double)(1.0f - ALPHA);
        h_delta[i] = (double)((1.0f - ALPHA) * ALPHA /
                              (h_offsets[i + 1] - h_offsets[i]));
        h_residual[i] = (double)0.0f;
      } else
        h_valuesPR[i] = 1.0 / (*numVertices);
    }
    break;
  case BFS:
    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 0;
      h_values[i] = *numVertices + 1;
    }
    h_frontier[srcVertex] = 1;
    h_values[srcVertex] = 1;
    break;
  case SSSP:
    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 0;
      h_values[i] = *numVertices + 1;
    }
    h_frontier[srcVertex] = 1;
    h_values[srcVertex] = 1;
    break;
  case CC:
    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 1;
      h_values[i] = i;
    }
    break;
  }
  return;
}

template <class EdgeType> void CSR<EdgeType>::SetPartitionsConfig() {
  h_partitionsOffsets.push_back(0);

  uint64 edgesInCurrentPartition = 0;

  for (uint32 i = 0; i < *numVertices; i++) {
    if ((h_offsets[i + 1] - h_offsets[i]) + edgesInCurrentPartition <=
        maxEdgesInPartition)

      edgesInCurrentPartition =
          edgesInCurrentPartition + (h_offsets[i + 1] - h_offsets[i]);

    else {
      h_partitionsOffsets.push_back(i);
      edgesInCurrentPartition =
          h_offsets[i + 1] - h_offsets[i]; // This vertex should already be
                                           // included in the new partition

      if (edgesInCurrentPartition > maxEdgesInPartition)
        std::cout << "One partition is too small for 1 vertex: "
                  << h_partitionsOffsets.size() << std::endl;
    }
  }

  h_partitionsOffsets.push_back(*numVertices);

  *numPartitions = h_partitionsOffsets.size() - 1;
  h_partitionCost = new float[*numPartitions];

  std::cout << "Maximum number of edges per partition: " << maxEdgesInPartition
            << std::endl;
  std::cout << "Number of partitions: " << *numPartitions << std::endl;

  return;
}

template <class EdgeType> void CSR<EdgeType>::SetNumStaticEdges() {
  size_t freeMemory;  // Available GPU memory
  size_t totalMemory; // Total GPU memory
  cudaMemGetInfo(&freeMemory,
                 &totalMemory); // Available and Total memory in bytes

  if (algorithm == SSSP)
    freeMemory *= 0.45; // SSSP requires two more allocs of equal size
  else
    freeMemory *= 0.9;

  unsigned long remainder = freeMemory % FRAGMENT;
  freeMemory = freeMemory - remainder; // Free memory in 16kB chunks

  numStaticEdges =
      freeMemory / sizeof(EdgeType); // Available memory in Edges count

  if (numStaticEdges > numEdges) // Potential overflow
    numStaticEdges = numEdges;

  uint64 edgesInStatic = 0;

  // numStaticEdges = EDGES_IN_PARTITION;
  for (uint32 i = 0; i < *numVertices; i++) {
    // Build static inStatic array
    if (h_offsets[i + 1] <= numStaticEdges) {
      edgesInStatic = edgesInStatic + (h_offsets[i + 1] - h_offsets[i]);
    }
  }

  numStaticEdges = edgesInStatic;

  std::cout << "Number of static edges: " << numStaticEdges << std::endl;

  return;
}

// TODO: verificar se damos allocs desnecessarios
template <class EdgeType> void CSR<EdgeType>::InitData() {
  Timer timer(std::string("InitData execution time: "));

  GPUAssert(cudaStreamCreate(&frontierStream));
  GPUAssert(cudaStreamCreate(&staticStream));
  GPUAssert(cudaStreamCreate(&demandStream));

  neighborMemCpyStreams.resize(nNGPUs);
  neighborComputeStreams.resize(nNGPUs);

  for (int i = 0; i < nNGPUs; ++i) {
    cudaSetDevice(i + 1);
    for (int j = 0; j < N_FILTER_STREAMS; ++j)
      GPUAssert(cudaStreamCreate(&neighborMemCpyStreams[i][j]));
  }

  cudaSetDevice(0);

  for (int i = 0; i < nNGPUs; ++i) {
    for (int j = 0; j < N_FILTER_STREAMS; ++j)
      GPUAssert(cudaStreamCreate(&neighborComputeStreams[i][j]));
  }

  for (uint32 i = 0; i < N_TARGET_FILTER_STREAMS; i++)
    GPUAssert(cudaStreamCreate(&streams[i]));

  if (nNGPUs == 0)
    h_filterThreshold = 0.85;
  else if (nNGPUs == 1) {
    GPUAssert(cudaDeviceEnablePeerAccess(1, 0));
    h_filterThreshold = 0.45;
  } else if (nNGPUs == 2) {
    GPUAssert(cudaDeviceEnablePeerAccess(1, 0));
    GPUAssert(cudaDeviceEnablePeerAccess(2, 0));
    h_filterThreshold = 0.30;
  } else if (nNGPUs == 3) {
    GPUAssert(cudaDeviceEnablePeerAccess(1, 0));
    GPUAssert(cudaDeviceEnablePeerAccess(2, 0));
    GPUAssert(cudaDeviceEnablePeerAccess(3, 0));
    h_filterThreshold = 0.20;
  }

  // Copy the filter threshold to constant memory
  cudaMemcpyToSymbol(d_filterThreshold, &h_filterThreshold,
                     sizeof(h_filterThreshold));

  // Allocate data in the second Numa Node
  // if (nNGPUs > 1) {
  //
  //   h_edges2 = (EdgeType *)numa_alloc_onnode(numEdges * sizeof(*h_edges2),
  //   1);
  //
  //   cudaHostRegister(h_edges2, numEdges * sizeof(*h_edges2),
  //                    cudaHostRegisterDefault);
  //
  //   cudaMemcpy(h_edges2, h_edges, numEdges * sizeof(*h_edges2),
  //              cudaMemcpyHostToHost);
  //
  //   if (algorithm == SSSP) {
  //     h_weights2 =
  //         (uint32 *)numa_alloc_onnode(numEdges * sizeof(*h_weights2), 1);
  //
  //     cudaHostRegister(h_weights2, numEdges * sizeof(*h_weights2),
  //                      cudaHostRegisterDefault);
  //
  //     cudaMemcpy(h_weights2, h_weights, numEdges * sizeof(*h_weights2),
  //                cudaMemcpyHostToHost);
  //   }
  //
  //   cudaDeviceSynchronize();
  // }

  if (algorithm == PR) {
    if (type == "push") {
      h_delta = new double[*numVertices];
      h_residual = new double[*numVertices];
    }

    h_valuesPR = new double[*numVertices];
  } else
    h_values = new EdgeType[*numVertices];

  h_frontier = new bool[*numVertices];

  // Sizes that might get transferred to the GPU (Change to CudaDefault if we
  // end up transferring via cudaMemcpy())
  GPUAssert(cudaHostAlloc((void **)&frontierSize, sizeof(*frontierSize),
                          cudaHostAllocMapped));
  GPUAssert(cudaHostAlloc((void **)&staticSize, sizeof(*staticSize),
                          cudaHostAllocMapped));
  GPUAssert(cudaHostAlloc((void **)&demandSize, sizeof(*demandSize),
                          cudaHostAllocMapped));
  GPUAssert(cudaHostAlloc((void **)&numPartitions, sizeof(*numPartitions),
                          cudaHostAllocMapped));

  // GPUAssert(cudaHostAlloc((void **)&h_finished, sizeof(*h_finished),
  //                         cudaHostAllocDefault));
  //
  // GPUAssert(cudaMalloc(&d_finished, sizeof(*d_finished)));

  // *h_finished = 0;
  *frontierSize = 0;
  *staticSize = 0;
  *demandSize = 0;
  *numPartitions = 0;

  InitFrontierNValues();
  // Offsets Array
  GPUAssert(cudaMalloc(&d_offsets, (*numVertices + 1) * sizeof(*d_offsets)));
  GPUAssert(cudaMemcpy(d_offsets, h_offsets,
                       (*numVertices + 1) * sizeof(*d_offsets),
                       cudaMemcpyHostToDevice));

  // Demand List Array
  GPUAssert(cudaMalloc(&d_demandList, *numVertices * sizeof(*d_demandList)));
  GPUAssert(cudaMemset(d_demandList, 0, *numVertices * sizeof(*d_demandList)));

  // Static List Array
  GPUAssert(cudaMalloc(&d_staticList, *numVertices * sizeof(*d_staticList)));
  GPUAssert(cudaMemset(
      d_staticList, 0,
      *numVertices * sizeof(*d_staticList))); // This can only be as big as the

  // Prefix Sum Array
  GPUAssert(cudaMalloc(&d_prefixSum, *numVertices * sizeof(*d_prefixSum)));
  cudaMemset(d_prefixSum, 0,
             *numVertices * sizeof(*d_prefixSum)); // Not reallyh needed I think

  // Frontier Array
  GPUAssert(cudaMalloc(&d_frontier, *numVertices * sizeof(*d_frontier)));
  GPUAssert(cudaMemcpy(d_frontier, h_frontier,
                       *numVertices * sizeof(*d_frontier),
                       cudaMemcpyHostToDevice));

  // Static Frontier Array
  GPUAssert(
      cudaMalloc(&d_staticFrontier, *numVertices * sizeof(*d_staticFrontier)));
  GPUAssert(cudaMemset(d_staticFrontier, 0,
                       *numVertices * sizeof(*d_staticFrontier)));

  // Demand Frontier Array
  GPUAssert(
      cudaMalloc(&d_demandFrontier, *numVertices * sizeof(*d_demandFrontier)));
  GPUAssert(cudaMemset(d_demandFrontier, 0,
                       *numVertices * sizeof(*d_demandFrontier)));

  GPUAssert(
      cudaMalloc(&d_filterFrontier, *numVertices * sizeof(*d_filterFrontier)));
  GPUAssert(cudaMemset(d_filterFrontier, 0,
                       *numVertices * sizeof(*d_filterFrontier)));

  // Mostly Values Array
  if (algorithm == PR) {

    // Comment this for push
    if (type == "push") {
      GPUAssert(cudaMalloc(&d_delta, (*numVertices + 1) * sizeof(*d_delta)));
      GPUAssert(
          cudaMalloc(&d_residual, (*numVertices + 1) * sizeof(*d_residual)));
    } else {
      GPUAssert(cudaMalloc(&d_degree, *numVertices * sizeof(*d_degree)));
      GPUAssert(cudaMemcpy(d_degree, h_degree, *numVertices * sizeof(*d_degree),
                           cudaMemcpyHostToDevice));
      GPUAssert(cudaMalloc(&d_sum, *numVertices * sizeof(*d_sum)));
      GPUAssert(cudaMemset(d_sum, 0, *numVertices * sizeof(*d_sum)));

      d_thurstSum = thrust::device_ptr<double>(d_sum);
    }

    GPUAssert(cudaMalloc(&d_valuesPR, *numVertices * sizeof(*d_valuesPR)));
    GPUAssert(cudaMemcpy(d_valuesPR, h_valuesPR,
                         *numVertices * sizeof(*d_valuesPR),
                         cudaMemcpyHostToDevice));

  } else {
    GPUAssert(cudaMalloc(&d_values, *numVertices * sizeof(*d_values)));
    GPUAssert(cudaMemcpy(d_values, h_values, *numVertices * sizeof(*d_values),
                         cudaMemcpyHostToDevice));
  }

  SetPartitionsConfig();

  GPUAssert(cudaMalloc(&d_partitionsOffsets,
                       (*numPartitions + 1) * sizeof(*d_partitionsOffsets)));
  GPUAssert(cudaMemcpy(d_partitionsOffsets, h_partitionsOffsets.data(),
                       (*numPartitions + 1) * sizeof(*d_partitionsOffsets),
                       cudaMemcpyHostToDevice));

  GPUAssert(
      cudaMalloc(&d_partitionCost, *numPartitions * sizeof(*d_partitionCost)));

  GPUAssert(cudaMemset(d_partitionCost, 0,
                       *numPartitions * sizeof(*d_partitionCost)));

  GPUAssert(cudaHostAlloc((void **)&h_finished, sizeof(*h_finished),
                          cudaHostAllocDefault));

  GPUAssert(cudaMalloc(&d_partitionList,
                       N_TARGET_FILTER_STREAMS * sizeof(*d_partitionList)));

  d_nPartList = new uint32_t *[nNGPUs];

  for (uint32 i = 0; i < nNGPUs; i++) {
    cudaSetDevice(i + 1);
    GPUAssert(cudaMalloc(&d_nPartList[i],
                         N_FILTER_STREAMS * sizeof(*d_nPartList[i])));
  }

  cudaSetDevice(0);

  d_nFilterEdges = new EdgeType **[nNGPUs];

  if (algorithm == SSSP)
    d_nFilterWeights = new EdgeType **[nNGPUs];

  for (uint32 i = 0; i < nNGPUs; i++) {
    cudaSetDevice(i + 1);

    d_nFilterEdges[i] = new EdgeType *[N_FILTER_STREAMS];

    if (algorithm == SSSP)
      d_nFilterWeights[i] = new EdgeType *[N_FILTER_STREAMS];

    for (int stream = 0; stream < N_FILTER_STREAMS; stream++) {
      GPUAssert(cudaMalloc(&d_nFilterEdges[i][stream],
                           maxEdgesInPartition * sizeof(EdgeType)));

      if (algorithm == SSSP)
        GPUAssert(cudaMalloc(&d_nFilterWeights[i][stream],
                             maxEdgesInPartition * sizeof(EdgeType)));
    }
  }

  cudaSetDevice(0);

  for (uint32 i = 0; i < N_TARGET_FILTER_STREAMS; ++i) {
    GPUAssert(cudaMalloc(&d_filterEdges[i],
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    GPUAssert(cudaMemset(d_filterEdges[i], 0,
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    if (algorithm == SSSP) {
      GPUAssert(cudaMalloc(&d_filterWeights[i],
                           maxEdgesInPartition * sizeof(*d_filterWeights)));

      GPUAssert(cudaMemset(d_filterWeights[i], 0,
                           maxEdgesInPartition * sizeof(*d_filterWeights)));
    }
  }

  h_nPartList = new uint32_t *[nNGPUs];

  for (uint32 i = 0; i < nNGPUs; i++) {
    GPUAssert(cudaHostAlloc((void **)&h_nPartList[i],
                            N_FILTER_STREAMS * sizeof(*h_nPartList[i]),
                            cudaHostAllocDefault));
  }

  GPUAssert(cudaHostAlloc((void **)&h_partitionList,
                          N_TARGET_FILTER_STREAMS * sizeof(*h_partitionList),
                          cudaHostAllocDefault));

  GPUAssert(cudaMalloc(&d_inStatic, *numVertices * sizeof(*d_inStatic)));

  SetNumStaticEdges();

  GPUAssert(cudaMemset(d_inStatic, 0, *numVertices * sizeof(*d_inStatic)));

  // Static Edges Array
  GPUAssert(
      cudaMalloc(&d_staticEdges, numStaticEdges * sizeof(*d_staticEdges)));

  // Static Weights Array for SSSP
  if (algorithm == SSSP) {

    for (uint32 i = 0; i < numEdges; i++)
      h_weights[i] = 20;

    GPUAssert(cudaMalloc(&d_staticWeights,
                         numStaticEdges * sizeof(*d_staticWeights)));
  }

  // Thurst Wrappers
  thrustFrontier = thrust::device_ptr<bool>(d_frontier);
  thurstStaticFrontier = thrust::device_ptr<bool>(d_staticFrontier);
  thurstDemandFrontier = thrust::device_ptr<bool>(d_demandFrontier);
  thurstPrefixSum = thrust::device_ptr<EdgeType>(d_prefixSum);

  GPUAssert(cudaPeekAtLastError());

  avgVertPerPart = (double)*numVertices / *numPartitions;

  return;
}

// Make this robust to errors reading from the file
template <class EdgeType>
void CSR<EdgeType>::ReadInputFile(const std::string &filePath,
                                  ALGORITHM_TYPE algo, uint64 sourceVertex,
                                  uint32 numberNGPUs,
                                  std::unordered_map<int, int> numaAffinityMap,
                                  std::string algoType) {
  algorithm = algo;
  type = algoType;
  srcVertex = sourceVertex;
  nNGPUs = numberNGPUs;

  std::set<uint32> uniqueNumaNodes;
  for (const auto &entry : numaAffinityMap) {
    uniqueNumaNodes.insert(entry.second);
  }

  h_edges2 = new EdgeType *[uniqueNumaNodes.size()];

  if (algorithm == SSSP)
    h_weights2 = new EdgeType *[uniqueNumaNodes.size()];

  //   if (algorithm == SSSP) {
  //     h_weights2 =
  //         (uint32 *)numa_alloc_onnode(numEdges * sizeof(*h_weights2), 1);
  //
  //     cudaHostRegister(h_weights2, numEdges * sizeof(*h_weights2),
  //                      cudaHostRegisterDefault);
  //
  //     cudaMemcpy(h_weights2, h_weights, numEdges * sizeof(*h_weights2),
  //                cudaMemcpyHostToHost);
  //   }

  // Convert set to vector
  std::vector<uint32> numaNodesIndexing(uniqueNumaNodes.begin(),
                                        uniqueNumaNodes.end());

  // Create a map from NUMA nodes to their indices
  std::unordered_map<uint32, uint32> numaNodeToIndex;
  for (uint32 i = 0; i < numaNodesIndexing.size(); ++i) {
    numaNodeToIndex[numaNodesIndexing[i]] = i;
  }

  for (const auto &entry : numaAffinityMap) {
    uint32 gpuId = entry.first;
    uint32 numaNode = entry.second;
    GPUAffinityMap[gpuId] = numaNodeToIndex[numaNode];
  }

  std::cout << "GPU ID -> Index Mapping:" << std::endl;
  for (const auto &pair : GPUAffinityMap) {
    std::cout << "GPU" << pair.first << " -> Index " << pair.second
              << std::endl;
  }

  cudaHostAlloc((void **)&numVertices, sizeof(*numVertices),
                cudaHostAllocMapped);

  uint64 nV;

  std::ifstream infile(filePath, std::ios::in | std::ios::binary);

  if (!infile.is_open()) {
    std::cerr << " Error opening file to read" << std::endl;
    exit(0);
  }

  infile.read((char *)&nV, sizeof(uint64));
  infile.read((char *)&numEdges, sizeof(uint64));

  *numVertices = static_cast<EdgeType>(nV);

  std::cout << "Num Vertices: " << *numVertices << std::endl;
  std::cout << "Num Edges: " << numEdges << std::endl;

  h_offsets = new uint64[*numVertices + 1];
  infile.read((char *)h_offsets, (*numVertices) * sizeof(uint64));

  h_offsets[*numVertices] = numEdges;

  std::unordered_map<uint32, bool> allocatedNumaNodes;

  uint32 defaultNumaNode = 0;
  auto it = numaAffinityMap.find(0);
  if (it != numaAffinityMap.end())
    defaultNumaNode = it->second;
  else
    std::cout << "Could not find GPU ID in numaAffinityMap" << std::endl;

  // Allocate h_edges on default Numa Node
  h_edges2[GPUAffinityMap[0]] = (EdgeType *)numa_alloc_onnode(
      numEdges * sizeof(EdgeType), defaultNumaNode);

  // Pin the memory
  cudaHostRegister(h_edges2[GPUAffinityMap[0]], numEdges * sizeof(EdgeType),
                   cudaHostRegisterMapped);

  if (algorithm == SSSP) {
    h_weights2[GPUAffinityMap[0]] = (EdgeType *)numa_alloc_onnode(
        numEdges * sizeof(EdgeType), defaultNumaNode);

    // Pin the memory
    cudaHostRegister(h_weights2[GPUAffinityMap[0]], numEdges * sizeof(EdgeType),
                     cudaHostRegisterMapped);
  }

  // cudaHostGetDevicePointer(&d_edges, h_edges, 0);

  allocatedNumaNodes[defaultNumaNode] = true;
  // cudaMallocManaged((void **)&h_edges, numEdges * sizeof(EdgeType));
  //
  // cudaMemAdvise(h_edges, numEdges * sizeof(EdgeType),
  //               cudaMemAdviseSetAccessedBy, 0);

  infile.read((char *)h_edges2[GPUAffinityMap[0]], numEdges * sizeof(uint32));

  if (algorithm == SSSP) {
    // uint64 *tempWeights = new uint64[numEdges];
    // infile.read((char *)tempWeights, numEdges * sizeof(uint64));

    // Allocate on default Numa Node
    h_weights = (EdgeType *)numa_alloc_onnode(numEdges * sizeof(EdgeType), 0);

    // Pin the memory
    cudaHostRegister(h_weights, numEdges * sizeof(EdgeType),
                     cudaHostRegisterMapped);

    cudaHostGetDevicePointer(&d_weights, h_weights, 0);

    // Replace with file read
    for (int i = 0; i < numEdges; ++i)
      h_weights[i] = 20;

  } else if (algorithm == PR && type == "pull") {
    h_degree = new uint32[*numVertices];

    infile.read((char *)h_degree, *numVertices * sizeof(uint32));
  }

  infile.close();

  for (const auto &pair : numaNodeToIndex) {
    uint32 numaNode = pair.first;
    uint32 index = pair.second;

    // Check if the NUMA node has already been allocated
    if (allocatedNumaNodes.find(numaNode) == allocatedNumaNodes.end()) {
      h_edges2[index] = (EdgeType *)(numa_alloc_onnode(
          numEdges * sizeof(*h_edges2[index]), numaNode));

      // Register memory with CUDA
      cudaHostRegister(h_edges2[index], numEdges * sizeof(*h_edges2[index]),
                       cudaHostRegisterDefault);

      cudaMemcpy(h_edges2[index], h_edges2[GPUAffinityMap[0]],
                 numEdges * sizeof(*h_edges2[index]), cudaMemcpyHostToHost);

      if (algorithm == SSSP) {
        h_weights2[index] = (EdgeType *)(numa_alloc_onnode(
            numEdges * sizeof(*h_weights2[index]), numaNode));

        // Register memory with CUDA
        cudaHostRegister(h_weights2[index],
                         numEdges * sizeof(*h_weights2[index]),
                         cudaHostRegisterDefault);

        cudaMemcpy(h_weights2[index], h_weights2[GPUAffinityMap[0]],
                   numEdges * sizeof(*h_edges2[index]), cudaMemcpyHostToHost);
      }

      allocatedNumaNodes[numaNode] = true;
    }
  }

  return;
}

template <class EdgeType> void CSR<EdgeType>::DumpValues() {

  if (algorithm == PR)
    cudaMemcpy(h_valuesPR, d_valuesPR, *(numVertices) * sizeof(*h_valuesPR),
               cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(h_values, d_values, *(numVertices) * sizeof(*h_values),
               cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  std::string filepath = "results/values.bin";

  std::ofstream file(filepath);
  if (!file.is_open())
    return;

  // std::cout << "Results:" << std::endl;
  if (algorithm == PR) {
    for (EdgeType i = 0; i < *numVertices; ++i)
      file.write(reinterpret_cast<const char *>(&h_valuesPR[i]),
                 sizeof(*h_valuesPR));
    // for (uint32 i = 0; i < 31; ++i)
    //   std::cout << "[" << i << "]= " << h_valuesPR[i] << std::endl;

  } else {
    for (EdgeType i = 0; i < *numVertices; ++i)
      file.write(reinterpret_cast<const char *>(&h_values[i]),
                 sizeof(*h_values));

    // for (uint32 i = 0; i < 31; ++i)
    //   std::cout << "[" << i << "]= " << h_values[i] << std::endl;
  }

  file.close();
}

// TODO: verificar se damos todos os frees
template <class EdgeType> void CSR<EdgeType>::Free() {

  if (algorithm == SSSP)
    cudaFreeHost(h_weights);

  else if (algorithm == PR && type == "pull")
    delete[] h_degree;

  delete[] h_offsets;
  // cudaFreeHost(h_edges);

  if (algorithm == PR) {
    if (type == "push") {
      cudaFree(d_residual);
      cudaFree(d_delta);
    } else {
      cudaFree(d_degree);
      cudaFree(d_sum);
    }

    cudaFree(d_valuesPR);
    delete[] h_valuesPR;
  } else {

    cudaFree(d_values);
    delete[] h_values;
  }

  delete[] h_frontier;

  cudaFreeHost(frontierSize);
  cudaFreeHost(staticSize);
  cudaFreeHost(demandSize);
  cudaFreeHost(numPartitions);

  cudaFree(d_offsets);

  cudaFree(d_demandList);

  cudaFree(d_staticList);

  cudaFree(d_prefixSum);

  cudaFree(d_frontier);

  cudaFree(d_staticFrontier);

  cudaFree(d_demandFrontier);

  cudaFree(d_filterFrontier);

  cudaFree(d_partitionsOffsets);
  cudaFree(d_partitionCost);

  cudaFree(d_filterEdges);

  cudaFree(d_filterWeights);
  cudaFree(d_inStatic);
  cudaFree(d_staticEdges);

  if (algorithm == SSSP) {
    //  cudaFree(d_filterWeights);
    cudaFree(d_staticWeights);
  }

  GPUAssert(cudaPeekAtLastError());

  return;
}

#endif // GRAPH_CUH
