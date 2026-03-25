# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Thievory is a GPU-accelerated out-of-memory graph processing framework. It processes graphs that exceed GPU memory by partitioning edges and streaming them between host and device, using NUMA-aware allocation and multi-GPU support. It implements four algorithms: BFS, CC, SSSP, and PageRank (push and pull variants). Each algorithm has 32-bit and 64-bit edge type variants.

## Prerequisites

- NVIDIA CUDA toolkit
- NUMA library (`libnuma-dev` or equivalent)
- A `results/` directory must exist in the project root (output is written to `results/values.bin`)

## Build

```bash
make                # builds ./main (parallel by default via -j$(nproc))
make clean          # removes bin/*.o and ./main
```

**Before building**, update the Makefile:
- Change `-arch=sm_80` to match your GPU architecture
- Update `$(HOME)/local/include` and `$(HOME)/local/lib` to your NUMA installation paths

The converter (gitignored) is built separately:
```bash
cd comparer && g++ compare.cpp -o compare
```

## Running

```bash
./main --input <path_to_csr> --algo bfs --source 0 --edgeSize 4 --runs 3 --gpus 0
```

Flags: `--input` (required), `--algo` {bfs,cc,pr,sssp}, `--edgeSize` {4,8}, `--type` {push,pull} (PageRank only), `--source` (BFS/SSSP), `--runs`, `--gpus` (number of neighbor GPUs), `--device`.

## Architecture

**Entry point**: `src/main.cu` — parses CLI args, discovers GPU-NUMA topology via `nvidia-smi`, pins to the correct NUMA node, then dispatches to the chosen algorithm (e.g., `BFS32()`, `SSSP64()`).

**Core abstractions** (all in `src/`):
- `graph.cuh` — `CSR<EdgeType>` template class. Manages the CSR graph representation, host/device memory, edge partitioning, and multi-GPU data. Handles reading CSR binary files, partitioning edges into chunks that fit in GPU memory, and streaming partitions via CUDA streams.
- `common.cuh` — Shared type definitions (`uint32`/`uint64`), CUDA constants (block size, warp size, partition sizes), the `GPUAssert` macro, and common GPU kernels for frontier management (static/demand/filter frontier splitting, partition cost calculation).
- `timer.hpp` — Simple RAII-style CUDA event timer.
- `utils.hpp` — Host-side NUMA topology discovery (calls `nvidia-smi topo`).

**Algorithm modules** (`src/{bfs,cc,sssp,pr}/`): Each has the same structure:
- `<algo>.cuh` — Function declarations (e.g., `BFS32()`, `BFS64()`)
- `<algo>.cu` — Algorithm orchestration: graph loading, memory allocation, iteration loop, timing
- `<algo>_kernels.cuh` / `<algo>_kernels.cu` — CUDA kernel implementations

**Key design pattern**: The framework splits the active frontier into "static" vertices (edges resident in GPU memory) and "demand" vertices (edges that must be streamed from host). A filtering heuristic further classifies demand partitions by active-edge density to decide transfer strategy. This three-way split (static / demand / filter) is central to the processing pipeline.

**Comparer** (`comparer/compare.cpp`): Standalone tool to diff two binary result files (uint32, uint64, or double). Used for correctness validation across frameworks/runs.

## Input Format

Thievory uses binary CSR format. The converter tool (separate from the comparer) converts `.txt` edge lists to CSR and other formats. Converter types: `csr` (Thievory), `csc` (pull PageRank), `wcsr` (weighted), `bel` (EMOGI), `el`/`wel` (Subway), `hwel` (HyTGraph).

## Testing

No automated test suite. Validate by comparing `results/values.bin` output against known-good results using the comparer tool:
```bash
cd comparer && g++ compare.cpp -o compare
./compare --input1 file1.bin --input2 file2.bin --vertices <N> --type 0
```
`--type`: 0 = uint32, 1 = uint64, 2 = double.
