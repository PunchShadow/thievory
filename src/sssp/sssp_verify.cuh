#ifndef SSSP_VERIFY_CUH
#define SSSP_VERIFY_CUH

// Parallel CPU SSSP verification for thievory.
// Runs a label-correcting Bellman-Ford on the host using OpenMP and compares
// against the GPU result.  Uses the same INF sentinel and source distance as
// the GPU path:
//   - INF = numeric_limits<EdgeType>::max() - 1
//   - dist[src] = 0

#include "../common.cuh"

#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

template <typename EdgeType>
static bool cpu_verify_sssp(const uint64 *h_offsets, const EdgeType *h_edges,
                            const EdgeType *h_weights, uint64 numVertices,
                            uint64 numEdges, EdgeType sourceNode,
                            const EdgeType *gpuResult) {
  std::cout << "\n=== CPU SSSP Verification (parallel Bellman-Ford) ===" << std::endl;
  auto t0 = std::chrono::steady_clock::now();

  const EdgeType INF = std::numeric_limits<EdgeType>::max() - 1;

  EdgeType *dist = new EdgeType[numVertices];
#pragma omp parallel for
  for (uint64 i = 0; i < numVertices; i++) dist[i] = INF;
  dist[sourceNode] = 0;

  bool *inNextFrontier = new bool[numVertices];
  std::memset(inNextFrontier, 0, numVertices * sizeof(bool));

  std::vector<EdgeType> frontier;
  frontier.reserve(numVertices);
  frontier.push_back(sourceNode);

  int iter = 0;
  while (!frontier.empty()) {
    iter++;
    std::memset(inNextFrontier, 0, numVertices * sizeof(bool));

#pragma omp parallel for schedule(dynamic, 256)
    for (size_t fi = 0; fi < frontier.size(); fi++) {
      EdgeType v = frontier[fi];
      EdgeType vDist = dist[v];
      if (vDist >= INF) continue;

      const uint64 s = h_offsets[v];
      const uint64 e = h_offsets[v + 1];
      for (uint64 ei = s; ei < e; ei++) {
        EdgeType u = h_edges[ei];
        if (u >= numVertices) continue;
        EdgeType w = h_weights[ei];
        EdgeType newDist = vDist + w;

        // Atomic min via CAS loop.
        EdgeType old = dist[u];
        while (newDist < old) {
          EdgeType prev = __sync_val_compare_and_swap(&dist[u], old, newDist);
          if (prev == old) {
            inNextFrontier[u] = true;
            break;
          }
          old = prev;
        }
      }
    }

    frontier.clear();
    for (uint64 i = 0; i < numVertices; i++) {
      if (inNextFrontier[i]) frontier.push_back(static_cast<EdgeType>(i));
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << "  CPU time: " << cpu_ms << " ms, iterations: " << iter << std::endl;

  uint64 mismatch = 0, rg = 0, rc = 0;
#pragma omp parallel for reduction(+ : mismatch, rg, rc)
  for (uint64 i = 0; i < numVertices; i++) {
    if (gpuResult[i] < INF) rg++;
    if (dist[i] < INF) rc++;
    if (gpuResult[i] != dist[i]) mismatch++;
  }

  std::cout << "  GPU reached: " << rg << " / " << numVertices << std::endl;
  std::cout << "  CPU reached: " << rc << " / " << numVertices << std::endl;
  std::cout << "  Mismatches:  " << mismatch << std::endl;

  if (mismatch > 0) {
    int shown = 0;
    for (uint64 i = 0; i < numVertices && shown < 10; i++) {
      if (gpuResult[i] != dist[i]) {
        std::cout << "    node " << i
                  << ": GPU=" << static_cast<uint64>(gpuResult[i])
                  << " CPU=" << static_cast<uint64>(dist[i]) << std::endl;
        shown++;
      }
    }
  }

  bool pass = (mismatch == 0);
  std::cout << "  SSSP Verification: " << (pass ? "PASSED" : "FAILED") << std::endl;

  delete[] dist;
  delete[] inNextFrontier;
  return pass;
}

#endif // SSSP_VERIFY_CUH
